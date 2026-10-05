"""Run each experiment in a fresh Python process and stream durable text logs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from runtime import atomic_json, configure_environment


def launch(cfg):
    import train
    configure_environment()
    folder = train.run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    config_path = folder / "config.json"
    env_path = folder / "environment.json"
    if result_path.exists() and config_path.exists() and env_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if not train._same_training_config(saved, cfg):
            raise ValueError(f"Saved config differs for {folder}; choose a new output directory")
        if env.get("split_sha256") != train.split_fingerprint(cfg):
            raise ValueError("Split CSV changed; cannot reuse results")
        sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
        if env.get("source_sha256") != sources:
            raise ValueError("Source changed since this experiment; preserve old output and choose a new output directory")
        finished = result.get("epochs_completed", 0) == cfg.epochs
        for key, target in (("checkpoint", folder / "checkpoint_best.pt"),
                            ("curve", Path(cfg.curves_dir)/f"{cfg.exp_id}_seed{cfg.seed}.png"),
                            ("val_predictions", train.pred_path(cfg,"val"))):
            result[key] = str(target)
            finished = finished and target.is_file()
        if cfg.save_test_predictions:
            finished = finished and train.pred_path(cfg,"test").is_file() and bool(result.get("test_metrics"))
            for key in ("inference_method", "aggregate_space", "temperature_scale"):
                if train.pred_path(cfg,"test").exists() and saved.get(key) != getattr(cfg,key):
                    raise ValueError("Final inference configuration already locked by saved test predictions")
        if finished:
            print(f"Reuse {cfg.exp_id}/seed{cfg.seed}: completed artifacts verified", flush=True)
            return result
    request = folder / "request.json"
    atomic_json(request, asdict(cfg))
    command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")), "--config", str(request.resolve())]
    print(f"Start worker {cfg.exp_id}/seed{cfg.seed}; log={folder / 'run.log'}", flush=True)
    # Never buffer the whole stdout in RAM. Each line is persisted before display.
    with (folder / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   env={**os.environ, "PYTHONUNBUFFERED":"1", "PYTHONIOENCODING":"utf-8"})
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if status:
        raise RuntimeError(f"Worker {cfg.exp_id} exited with {status}. See {folder / 'run.log'}. "
                           "Rerun with the same settings to resume the last completed epoch.")
    if not result_path.is_file():
        raise RuntimeError("Worker returned without result.json")
    return json.loads(result_path.read_text(encoding="utf-8"))
