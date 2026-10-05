"""Bound host resources before importing torch; diagnostics never retain tensors."""
from __future__ import annotations

import json
import os
from pathlib import Path


def configure_environment():
    # cuDNN defaults to up to 10,000 cached plans (~2 GiB). Keep a bounded cache.
    # https://docs.pytorch.org/docs/stable/cuda_environment_variables.html
    defaults = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2", "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "128",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def memory_snapshot():
    """Linux process RSS and cgroup headroom (reclaimable file cache excluded)."""
    result = {}
    try:
        rows = Path("/proc/self/status").read_text().splitlines()
        result["rss_gib"] = next(int(r.split()[1]) / 1024**2 for r in rows if r.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        pass
    root = Path("/sys/fs/cgroup")
    for used, limit, stat, cache_key in (
        (root / "memory.current", root / "memory.max", root / "memory.stat", "inactive_file"),
        (root / "memory/memory.usage_in_bytes", root / "memory/memory.limit_in_bytes",
         root / "memory/memory.stat", "total_inactive_file"),
    ):
        try:
            current, maximum = int(used.read_text()), int(limit.read_text())
            if maximum >= 2**60:
                continue
            stats = dict(line.split() for line in stat.read_text().splitlines())
            reclaimable = int(stats.get(cache_key, 0))
            result.update(cgroup_gib=current / 1024**3, limit_gib=maximum / 1024**3,
                          headroom_gib=(maximum - current + reclaimable) / 1024**3)
            break
        except (OSError, ValueError):
            pass
    return result


def check_memory(max_rss_gib=18.0, min_headroom_gib=3.0):
    stats = memory_snapshot()
    if ((max_rss_gib and stats.get("rss_gib", 0) >= max_rss_gib)
            or stats.get("headroom_gib", float("inf")) < min_headroom_gib):
        raise MemoryError(f"Host RAM guard stopped this experiment: {stats}. "
                          "The last completed epoch remains in checkpoint_last.pt; see run.log.")
    return stats


def resource_report():
    import torch
    parts = [f"{key}={value:.2f}" for key, value in memory_snapshot().items()]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            parts.append(f"gpu{i}_alloc={torch.cuda.memory_allocated(i)/1024**3:.2f}GiB "
                         f"reserved={torch.cuda.memory_reserved(i)/1024**3:.2f}GiB")
    return " | ".join(parts) or "RAM stats unavailable on this OS"
