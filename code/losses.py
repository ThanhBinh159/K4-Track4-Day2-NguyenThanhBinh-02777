"""Classification losses and batch-level Mixup/CutMix helpers."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    kind = kind.lower()
    weight = kw.get("weight")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1), weight=weight)
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), alpha=kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires weight=class_weights(...)")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross entropy with uniform label smoothing and optional class weights."""

    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        if not 0.0 <= float(smoothing) < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight).detach().clone())

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight,
                               label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma=0 with no alpha is standard cross entropy."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if float(gamma) < 0:
            raise ValueError("gamma must be nonnegative")
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        target = target.long()
        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -((1.0 - log_pt.exp()).clamp_min(0.0) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(device=logits.device, dtype=logits.dtype)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() != 9 or torch.any(counts <= 0):
        raise ValueError("counts must contain 9 positive training-set class counts")
    beta = float(beta)
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1)")
    weights = counts.reciprocal() if beta == 0 else (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return (weights / weights.mean()).to(dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if x.ndim != 4 or len(y) != len(x):
        raise ValueError("x must be NCHW and y must have the same batch length")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be mixup or cutmix")
    if len(x) < 2:
        return x, (y, y, 1.0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[permutation], (y_a, y_b, lam)

    height, width = x.shape[-2:]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + (cut_h + 1) // 2, height)
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + (cut_w + 1) // 2, width)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    lam = float(lam)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
