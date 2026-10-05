"""One training implementation for B/T/F; bounded memory, atomic epoch resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args, get_type_hints

from runtime import atomic_json, check_memory, configure_environment, resource_report
configure_environment()  # Before torch/CUDA initialization in the worker process.


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64  # Notebook uses 32 for ALL backbones to fit a T4.
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    data_parallel: bool = True
    convnext_data_parallel: bool = False
    channels_last: bool = True
    num_workers: int = 0
    cpu_threads: int = 1
    max_rss_gib: float = 18.0
    min_headroom_gib: float = 3.0
    validate_data: bool = True
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    resume: bool = True
    save_test_predictions: bool = False
    inference_method: str = "I00"
    aggregate_space: str = "prob"
    temperature_scale: bool = False


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def use_data_parallel(cfg, gpu_count):
    return bool(cfg.data_parallel and gpu_count > 1 and
                (not cfg.backbone.startswith("convnext") or cfg.convnext_data_parallel))


def build_optimizer(model, cfg):
    import torch
    import model as models
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
                             foreach=False)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    import torch
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(total, max(0, round(cfg.warmup_epochs * steps_per_epoch)))
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _unwrap_model(model):
    return getattr(model, "module", model)


class EMA:
    def __init__(self, model, decay):
        import copy
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = decay
        self.model = copy.deepcopy(_unwrap_model(model)).eval().requires_grad_(False)

    def update(self, model):
        import torch
        source = _unwrap_model(model).state_dict()
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                if value.is_floating_point():
                    value.lerp_(source[key].detach(), 1 - self.decay)
                else:
                    value.copy_(source[key])

    def copy_to(self, model):
        _unwrap_model(model).load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    import torch
    import losses
    base = _unwrap_model(model)
    model.train()
    if cfg.init == "frozen":
        base.eval()
        base.get_classifier().train()
    amp = cfg.amp and device.type == "cuda"
    total_loss = torch.zeros((), device=device)
    seen, steps = 0, len(loader)
    started = time.perf_counter()
    last_log = started
    for step, (images, labels, _) in enumerate(loader, 1):
        if step == 1 or step % 25 == 0:
            check_memory(cfg.max_rss_gib, cfg.min_headroom_gib)
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if cfg.mix:
                images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, labels)
        old_scale = scaler.get_scale() if scaler is not None else 1
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= old_scale
        else:
            loss.backward()
            optimizer.step()
            stepped = True
        if stepped:
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(base)
        total_loss.add_(loss.detach().float(), alpha=len(labels))
        seen += len(labels)
        # Text heartbeat at most once per minute, plus first/last batch; no widget spam.
        now = time.perf_counter()
        if step == 1 or step == steps or now - last_log >= 60:
            mean_loss = total_loss.item() / seen
            if not math.isfinite(mean_loss):
                raise FloatingPointError("Nonfinite training loss")
            print(f"{cfg.exp_id} progress={step}/{steps} loss={mean_loss:.4f} "
                  f"elapsed={now-started:.1f}s | {resource_report()}", flush=True)
            last_log = now
        del loss, logits, images, labels
    return {"train_loss": total_loss.item() / seen, "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device, amp=False):
    import numpy as np
    import torch
    model.eval()
    names, ys, outputs = [], [], []
    total_loss = torch.zeros((), device=device)
    with torch.inference_mode():
        for images, labels, batch_names in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
                loss = criterion(logits, labels.to(device, non_blocking=True))
            total_loss.add_(loss.float(), alpha=len(labels))
            outputs.append(logits.float().cpu().numpy())
            ys.append(labels.numpy())
            names.extend(batch_names)
    if not names:
        raise ValueError("Empty evaluation loader")
    return names, np.concatenate(ys), np.concatenate(outputs), total_loss.item() / len(names)


def _probabilities(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    x = np.exp(x - x.max(1, keepdims=True))
    return x / x.sum(1, keepdims=True)


def _predict_method(model, loader, device, cfg):
    import inference
    return inference.predict_method(model, loader, device, cfg.inference_method,
                                    cfg.aggregate_space, max(32, cfg.img_size - 32))


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = [r["epoch"] for r in history]
    for key in ("train_loss", "val_loss"):
        axes[0].plot(xs, [r[key] for r in history], label=key)
    for key in ("macro_f1_val", "top1_val"):
        axes[1].plot(xs, [r[key] for r in history], label=key)
    axes[2].plot(xs, [r["lr"] for r in history], label="backbone LR (end of epoch)")
    for ax, ylabel in zip(axes, ("Loss", "Validation score", "Learning rate")):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _atomic_torch_save(torch, state, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def _rng_state(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(torch, np, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _same_training_config(saved, cfg):
    ignored = {"images_dir", "labels_dir", "out_dir", "pred_dir", "curves_dir", "resume", "validate_data",
               "max_rss_gib", "min_headroom_gib", "save_test_predictions", "inference_method",
               "aggregate_space", "temperature_scale"}
    return {k:v for k,v in saved.items() if k not in ignored} == {
        k:v for k,v in asdict(cfg).items() if k not in ignored}


def split_fingerprint(cfg):
    return {s: hashlib.sha256((Path(cfg.labels_dir) / f"{s}_subset{cfg.fold}.csv").read_bytes()).hexdigest()
            for s in ("train", "val", "test")}


def run(cfg):
    import numpy as np
    import pandas as pd
    import torch
    import dataset
    import model as models
    import losses
    import inference
    import eval as ev
    from importlib.metadata import version

    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    test_path = pred_path(cfg, "test")
    if cfg.save_test_predictions and test_path.exists():
        raise FileExistsError(f"Test predictions already exist; refuse to rerun: {test_path}")
    torch.set_num_threads(cfg.cpu_threads)
    set_seed(cfg.seed)
    check_memory(cfg.max_rss_gib, cfg.min_headroom_gib)
    folder = run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    last_path, best_path = folder / "checkpoint_last.pt", folder / "checkpoint_best.pt"
    fingerprints = split_fingerprint(cfg)
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    previous_env = folder / "environment.json"
    if last_path.exists() and previous_env.exists():
        previous = json.loads(previous_env.read_text(encoding="utf-8"))
        if previous.get("source_sha256") != sources or previous.get("split_sha256") != fingerprints:
            raise ValueError("Source or split changed since checkpoint; preserve old output and use a new directory")
    if (folder / "config.json").exists():
        saved = json.loads((folder / "config.json").read_text())
        if not _same_training_config(saved, cfg):
            raise ValueError(f"Different training config already exists at {folder}; choose a new exp_id/output directory")
        if not cfg.resume and last_path.exists():
            raise FileExistsError("Checkpoint exists but resume=False; choose a new output directory")
    atomic_json(folder / "config.json", asdict(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atomic_json(folder / "environment.json", {"packages": {p:version(p) for p in
        ("torch", "torchvision", "timm", "numpy", "pandas", "Pillow")},
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "split_sha256": fingerprints, "source_sha256": sources})
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    if cfg.validate_data:
        dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size, False, num_workers=0, seed=cfg.seed)
    base = models.build_model(cfg.backbone, pretrained=not last_path.exists(),
                              num_classes=9, drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    if cfg.channels_last:
        base.to(memory_format=torch.channels_last)
    parallel = use_data_parallel(cfg, torch.cuda.device_count())
    model = torch.nn.DataParallel(base) if parallel else base
    print(f"{cfg.exp_id}: backbone={cfg.backbone} | DataParallel={parallel} | batch={cfg.batch_size} "
          f"workers={cfg.num_workers} | channels_last={cfg.channels_last} | {resource_report()}", flush=True)
    kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0).to(device)
        kw.update(weight=weight, alpha=weight)
    criterion = losses.build_criterion(cfg.loss, **kw).to(device)
    optimizer = build_optimizer(base, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(base, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_score, best_epoch, start = [], -1.0, 0, 0
    pretrained_tag = base.pretrained_tag
    if cfg.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if not _same_training_config(state["config"], cfg) or state["split_sha256"] != fingerprints:
            raise ValueError("Resume config or original split CSV differs; refusing checkpoint reuse")
        base.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        if ema is not None:
            ema.model.load_state_dict(state["ema_state"])
        history, best_score, best_epoch = state["history"], state["best_score"], state["best_epoch"]
        pretrained_tag = state["pretrained_tag"]
        start = state["epoch"] + 1
        _restore_rng(torch, np, state["rng_state"])
        train_loader.generator.set_state(state["loader_rng"])
        del state
        print(f"Resume {cfg.exp_id}: {start}/{cfg.epochs} epochs completed", flush=True)
    curve = Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png"
    for epoch in range(start, cfg.epochs):
        started = time.perf_counter()
        print(f"{cfg.exp_id} epoch {epoch+1}/{cfg.epochs}: train", flush=True)
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        train_seconds = time.perf_counter() - started
        print(f"{cfg.exp_id} epoch {epoch+1}: validation", flush=True)
        _, ys, zs, val_loss = evaluate(ema.model if ema else base, val_loader, criterion, device, amp=cfg.amp)
        probs = _probabilities(zs)
        metrics = ev.compute_metrics(ys, probs.argmax(1), probs)
        history.append({"epoch":epoch+1, **stats, "val_loss":val_loss, "macro_f1_val":metrics["macro_f1"],
                        "top1_val":metrics["top1"], "ece_val":metrics["ece"], "train_seconds":train_seconds,
                        "epoch_seconds":time.perf_counter()-started})
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch = float(metrics["macro_f1"]), epoch+1
            _atomic_torch_save(torch, {"model_state":(ema.model if ema else base).state_dict(),
                "best_epoch":best_epoch, "best_macro_f1":best_score, "config":asdict(cfg)}, best_path)
        _atomic_torch_save(torch, {"epoch":epoch, "model_state":base.state_dict(),
            "optimizer_state":optimizer.state_dict(), "scheduler_state":scheduler.state_dict(),
            "scaler_state":scaler.state_dict(), "ema_state":ema.model.state_dict() if ema else None,
            "history":history, "best_score":best_score, "best_epoch":best_epoch,
            "rng_state":_rng_state(torch,np), "loader_rng":train_loader.generator.get_state(),
            "pretrained_tag":pretrained_tag, "config":asdict(cfg), "split_sha256":fingerprints}, last_path)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        plot_curves(history, curve, f"{cfg.exp_id} | {cfg.backbone} | seed={cfg.seed}")
        print(f"{cfg.exp_id} epoch {epoch+1} saved | val_F1={best_score:.4f} best_epoch={best_epoch} "
              f"train={train_seconds:.1f}s | {resource_report()}", flush=True)
    del optimizer, scheduler, scaler, ema, model, train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    base.load_state_dict(state["model_state"])
    del state
    names, ys, logits, val_loss = evaluate(base, val_loader, criterion, device)
    np.save(folder / "val_logits.npy", logits)
    if cfg.inference_method == "I00":
        probs = _probabilities(logits)
    else:
        names, ys, probs = _predict_method(base, val_loader, device, cfg)
    temperature = None
    if cfg.temperature_scale:
        score_logits = np.log(np.clip(probs, 1e-12, 1))
        temperature = inference.fit_temperature(score_logits, ys)
        probs = inference.apply_temperature(score_logits, temperature)
    val_pred = ev.save_predictions(pred_path(cfg,"val"), names, ys, probs)
    np.save(folder / "val_probs.npy", probs)
    metrics = ev.compute_metrics(ys,probs.argmax(1),probs)
    import benchmark
    timing = benchmark.latency_report(base,1,cfg.img_size,dtype="fp32",device=str(device),iters=50)
    result = {"exp_id":cfg.exp_id,"seed":cfg.seed,"backbone":cfg.backbone,"pretrained_tag":pretrained_tag,
        "best_epoch":best_epoch,"checkpoint_macro_f1_val":best_score,"macro_f1_val":float(metrics["macro_f1"]),
        "top1_val":float(metrics["top1"]),"val_loss":val_loss,"epochs_completed":len(history),
        "train_seconds_per_epoch":float(np.mean([h["train_seconds"] for h in history])),
        "params_m":models.count_params(base),"gmacs":models.count_gmacs(base,cfg.img_size),
        "gmac_method":base.gmac_method,"checkpoint":str(best_path),"curve":str(curve),
        "val_predictions":str(val_pred),"temperature":temperature,"test_metrics":None,
        "data_parallel_active":parallel,"latency_p50_batch1_ms":timing["p50"],
        "latency_p95_batch1_ms":timing["p95"],"latency_gpu":timing["gpu"]}
    # Persist full run metadata BEFORE touching test, so a later interruption is recoverable.
    atomic_json(folder / "result.json", result)
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, dataset.build_transforms(False,cfg.img_size),
                                          cfg.batch_size, False, num_workers=0, seed=cfg.seed)
        names_t, ys_t, probs_t = _predict_method(base, test_loader, device, cfg)
        np.save(folder / "test_log_probs.npy", np.log(np.clip(probs_t,1e-12,1)))
        if temperature is not None:
            ev.save_predictions(Path(cfg.pred_dir)/f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv", names_t,ys_t,probs_t)
            probs_t = inference.apply_temperature(np.log(np.clip(probs_t,1e-12,1)),temperature)
        ev.save_predictions(test_path, names_t,ys_t,probs_t)
        test_metrics = ev.compute_metrics(ys_t,probs_t.argmax(1),probs_t)
        result["test_metrics"] = {k:float(test_metrics[k]) for k in ("top1","macro_f1","balanced_acc","ece","nll")}
    atomic_json(folder / "result.json", result)
    print(f"Completed {cfg.exp_id}/seed{cfg.seed}: {result['macro_f1_val']:.4f}",flush=True)
    return result


def parse_overrides(pairs):
    annotations = get_type_hints(Config)
    result = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in annotations:
            raise ValueError(f"Unknown or malformed Config override: {pair}")
        args = get_args(annotations[key])
        target = next((a for a in args if a is not type(None)), annotations[key])
        if raw.lower() in {"none","null"} and type(None) in args:
            value = None
        elif target is bool:
            if raw.lower() not in {"true","false","1","0","yes","no"}:
                raise ValueError(f"Invalid bool: {raw}")
            value = raw.lower() in {"true","1","yes"}
        else:
            value = target(raw)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    values.update(parse_overrides(args.set))
    run(Config(**values))


if __name__ == "__main__":
    main()
