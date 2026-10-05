"""Inference views, aggregation, calibration, and Conv/BatchNorm folding."""
from __future__ import annotations


def forward_probabilities(model, images, method="I00", space="prob", crop=None, temperature=None,
                          return_logits=False):
    """Same tensor path for validation, final prediction and latency measurement."""
    import torch
    import torch.nn.functional as F
    if method not in {"I00", "I01", "I02", "I03", "I08"}:
        raise ValueError(f"Unknown inference method {method}")
    with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                        enabled=method == "I08" and images.device.type == "cuda"):
        if method == "I02":
            # Process views sequentially: only small N x 9 outputs are retained.
            views = views_multicrop(images, crop or max(32, images.shape[-1] - 32))
            logits = [model(F.interpolate(v, size=images.shape[-2:], mode="bilinear", align_corners=False)).float()
                      for v in views]
        elif method in {"I01", "I03"}:
            logits = [model(images).float(), model(view_hflip(images)).float()]
        else:
            logits = [model(images).float()]
    if space == "prob":
        probs = torch.stack([z.softmax(1) for z in logits]).mean(0)
    elif space == "logit":
        probs = torch.stack(logits).mean(0).softmax(1)
    else:
        raise ValueError("space must be prob or logit")
    if temperature is not None:
        probs = (probs.clamp_min(1e-12).log() / temperature).softmax(1)
    return (probs, torch.stack(logits)) if return_logits else probs


def predict_method(model, loader, device, method="I00", space="prob", crop=None, logits_path=None):
    import numpy as np
    import torch
    from runtime import check_memory, resource_report
    model.eval()
    names, labels, predictions, view_logits = [], [], [], []
    with torch.inference_mode():
        for step, (images, target, batch_names) in enumerate(loader, 1):
            if step == 1 or step % 50 == 0:
                check_memory()
            result = forward_probabilities(model, images.to(device, non_blocking=True), method, space, crop,
                                           return_logits=logits_path is not None)
            if logits_path is not None:
                probs, logits = result
                view_logits.append(logits.cpu().numpy())  # K x batch x 9, no feature tensors.
            else:
                probs = result
            predictions.append(probs.cpu().numpy())
            labels.append(target.numpy())
            names.extend(batch_names)
            if step == 1 or step == len(loader):
                print(f"{method} inference {step}/{len(loader)} | {resource_report()}", flush=True)
    if not names:
        raise ValueError("Cannot predict an empty loader")
    if logits_path is not None:
        raw = np.concatenate(view_logits, axis=1)
        np.save(logits_path, raw[0] if len(raw) == 1 else raw)
    return names, np.concatenate(labels), np.concatenate(predictions)


def predict_logits(model, loader, device, view=None, autocast: bool = False):
    import numpy as np
    import torch

    model.eval()
    filenames, labels, logits_all = [], [], []
    use_amp = bool(autocast and torch.device(device).type == "cuda")
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            logits_all.append(logits.float().cpu().numpy())
            labels.append(target.cpu().numpy())
            filenames.extend(list(names))
    if not logits_all:
        raise ValueError("Cannot predict from an empty loader")
    return filenames, np.concatenate(labels), np.concatenate(logits_all)


def predict_multicrop(model, loader, device, crop: int, space: str = "prob"):
    """Evaluate five deterministic crops and return aggregated probabilities."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    filenames, labels, probs_all = [], [], []
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True)
            views = views_multicrop(images, crop)
            views = [F.interpolate(view, size=images.shape[-2:], mode="bilinear", align_corners=False)
                     for view in views]
            logits = [model(view) for view in views]
            probs_all.append(aggregate_views(logits, space))
            labels.append(target.cpu().numpy())
            filenames.extend(list(names))
    if not probs_all:
        raise ValueError("Cannot predict from an empty loader")
    return filenames, np.concatenate(labels), np.concatenate(probs_all)


def view_identity(x):
    return x


def view_hflip(x):
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    if x.ndim != 4:
        raise ValueError("x must have shape NCHW")
    height, width = x.shape[-2:]
    if crop < 1 or crop > min(height, width):
        raise ValueError(f"crop must be between 1 and {min(height, width)}")
    positions = [(0, 0), (0, width - crop), (height - crop, 0),
                 (height - crop, width - crop), ((height - crop) // 2, (width - crop) // 2)]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    import torch.nn.functional as F
    sizes = [int(size) for size in sizes]
    if x.ndim != 4 or not sizes or any(size < 1 for size in sizes):
        raise ValueError("x must be NCHW and sizes must contain positive dimensions")
    return [F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            for size in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    import torch
    import torch.nn.functional as F

    if not logits_per_view:
        raise ValueError("At least one view is required")
    tensors = [torch.as_tensor(value, dtype=torch.float32) for value in logits_per_view]
    if any(value.shape != tensors[0].shape for value in tensors):
        raise ValueError("All view logits must have the same shape")
    if space == "prob":
        probs = torch.stack([F.softmax(value, dim=1) for value in tensors]).mean(dim=0)
    elif space == "logit":
        probs = F.softmax(torch.stack(tensors).mean(dim=0), dim=1)
    else:
        raise ValueError("space must be prob or logit")
    return probs.cpu().numpy()


def ensemble_probs(list_of_probs):
    import numpy as np

    arrays = [np.asarray(value, dtype=np.float64) for value in list_of_probs]
    if not arrays:
        raise ValueError("At least one probability matrix is required")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All probability matrices must have the same shape and row order")
    if any(np.any(array < 0) or not np.allclose(array.sum(axis=1), 1.0, atol=1e-5) for array in arrays):
        raise ValueError("Each input must contain normalized probabilities")
    return np.mean(arrays, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(val_logits, dtype=torch.float64, device="cpu")
    labels = torch.tensor(val_labels, dtype=torch.long, device="cpu")
    if logits.ndim != 2 or len(labels) != len(logits) or len(labels) == 0:
        raise ValueError("val_logits must be (N,K), with one label per row")
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.clamp(-4.0, 4.0).exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().clamp(-4.0, 4.0).exp().item())


def apply_temperature(logits, T: float):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if float(T) <= 0:
        raise ValueError("T must be positive")
    values = torch.as_tensor(logits, dtype=torch.float64)
    return F.softmax(values / float(T), dim=1).cpu().numpy().astype(np.float64)


def fuse_conv_bn(model):
    """Return an eval copy with adjacent Conv2d/BatchNorm2d pairs fused."""
    import copy
    import torch
    from torch import nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    fused = copy.deepcopy(model).eval()
    pairs = [0]

    def fold_children(parent):
        for child in list(parent.children()):
            fold_children(child)
        # Registration order in an arbitrary nn.Module is NOT execution order.
        # Only Sequential guarantees Conv -> BN adjacency; leave other graphs unchanged.
        if not isinstance(parent, nn.Sequential):
            return
        children = list(parent.named_children())
        for (left_name, left), (right_name, right) in zip(children, children[1:]):
            if isinstance(left, nn.Conv2d) and isinstance(right, nn.BatchNorm2d):
                setattr(parent, left_name, fuse_conv_bn_eval(left, right))
                setattr(parent, right_name, nn.Identity())
                pairs[0] += 1

    fold_children(fused)
    if not pairs[0]:
        fused.bn_fusion_error = None
        fused.bn_fused_pairs = 0
        return fused

    parameter = next(fused.parameters())
    cfg = getattr(fused, "pretrained_cfg", None) or getattr(fused, "default_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 224, 224)) if isinstance(cfg, dict) else (3, 224, 224)
    size = int(input_size[-1])
    sample = torch.zeros(1, 3, size, size, device=parameter.device, dtype=parameter.dtype)
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            expected = model(sample)
            actual = fused(sample)
    finally:
        for module, was_training in training_states:
            module.training = was_training
    fused.bn_fusion_error = float((expected - actual).abs().max().item())
    fused.bn_fused_pairs = pairs[0]
    return fused
