"""Forward-pass latency measurement with warmup and explicit GPU synchronization."""
from __future__ import annotations

import time


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    import numpy as np

    warmup, iters = max(10, int(warmup)), max(50, int(iters))
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return {"p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "mean": float(np.mean(samples)), "n": len(samples),
            "warmup": warmup}


def _prepare_model_input(model, batch_size, img_size, dtype, device):
    import copy
    import torch

    if batch_size < 1 or img_size < 1:
        raise ValueError("batch_size and img_size must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    # Avoid another full model for the normal FP32/AMP path.
    measured_model = copy.deepcopy(model).to(target) if dtype == "fp16" else model.to(target)
    if dtype == "fp16":
        measured_model = measured_model.half()
    measured_model.eval()
    input_dtype = torch.float16 if dtype == "fp16" else torch.float32
    x = torch.randn(batch_size, 3, img_size, img_size, device=target, dtype=input_dtype)
    return measured_model, x, target


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    import torch

    dtype = dtype.lower()
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    measured_model, x, target = _prepare_model_input(model, batch_size, img_size, dtype, device)
    use_amp = dtype == "amp" and target.type == "cuda"

    def forward():
        with torch.inference_mode():
            with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=use_amp):
                measured_model(x)

    sync = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else None
    timing = bench(forward, warmup=warmup, iters=iters, sync=sync)
    timing.update({
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
        "dtype": dtype, "batch": int(batch_size), "img_size": int(img_size),
        "images_per_s": float(batch_size / (timing["p50"] / 1000.0)),
        "torch": torch.__version__, "preprocessing": "excluded", "bn_fused": False,
    })
    return timing


def tta_latency(model, k_views: int, **kw) -> dict:
    """Forward-only K-view microbenchmark; use method_latency for actual TTA/aggregation."""
    import torch

    if int(k_views) < 1:
        raise ValueError("k_views must be positive")
    batch_size = int(kw.get("batch_size", 1))
    img_size = int(kw.get("img_size", 224))
    dtype = str(kw.get("dtype", "fp32")).lower()
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    device = str(kw.get("device", "cuda"))
    warmup, iters = int(kw.get("warmup", 10)), int(kw.get("iters", 100))
    measured_model, x, target = _prepare_model_input(model, batch_size, img_size, dtype, device)
    use_amp = dtype == "amp" and target.type == "cuda"

    def forward_views():
        with torch.inference_mode():
            with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=use_amp):
                for _ in range(int(k_views)):
                    measured_model(x)

    sync = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else None
    timing = bench(forward_views, warmup=warmup, iters=iters, sync=sync)
    timing.update({"gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
                   "dtype": dtype, "batch": batch_size, "img_size": img_size,
                   "images_per_s": float(batch_size / (timing["p50"] / 1000.0)),
                   "torch": torch.__version__, "k_views": int(k_views),
                   "preprocessing": "excluded", "bn_fused": False})
    return timing


def method_latency(model, method="I00", space="prob", temperature=None, batch_size=1,
                   img_size=224, device="cuda", warmup=10, iters=50):
    """Time the actual view creation, forward, aggregation and optional calibration.

    Excludes file decoding, base resize/normalization and host-to-device transfer.
    """
    import torch
    from inference import forward_probabilities
    target = torch.device(device)
    model.eval()
    x = torch.zeros(batch_size, 3, img_size, img_size, device=target)
    def forward():
        with torch.inference_mode():
            return forward_probabilities(model, x, method, space, max(32,img_size-32), temperature)
    timing = bench(forward, warmup, iters,
                   (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else None)
    timing.update(gpu=torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
                  dtype="amp" if method == "I08" else "fp32", batch=batch_size, img_size=img_size,
                  images_per_s=batch_size / (timing["p50"] / 1000), torch=torch.__version__,
                  preprocessing="base preprocessing/H2D excluded; TTA and aggregation included", bn_fused=False)
    return timing
