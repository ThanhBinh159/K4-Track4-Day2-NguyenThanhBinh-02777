"""Model creation and parameter accounting for the DeepWeeds experiments."""
from __future__ import annotations

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Create a timm classifier and record the pretrained tag for reproducibility."""
    import timm

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be scratch, frozen, or finetune")
    model_name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = bool(pretrained and init != "scratch")
    model = timm.create_model(model_name, pretrained=use_pretrained,
                              num_classes=int(num_classes), drop_rate=float(drop_rate))
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    if not isinstance(cfg, dict):
        cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    model.pretrained_tag = ((cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "pretrained")
                            if use_pretrained else "scratch")
    model.model_name = model_name
    if init == "frozen":
        freeze_backbone(model)
    return model


def _classifier_parameters(model) -> set[int]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("This model does not expose a classifier through get_classifier()")
    return {id(param) for param in classifier.parameters()}


def freeze_backbone(model) -> None:
    """Freeze feature parameters while keeping the final classifier trainable."""
    classifier_ids = _classifier_parameters(model)
    if not classifier_ids:
        raise ValueError("The classifier has no parameters to train")
    for param in model.parameters():
        param.requires_grad = id(param) in classifier_ids
    model._freeze_backbone = True
    # Immediately freeze BatchNorm statistics; train.run reapplies this after model.train().
    for module in model.modules():
        if module.__class__.__name__ in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm"}:
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Return separate decay/no-decay groups at backbone and classifier learning rates."""
    classifier_ids = _classifier_parameters(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [],
              "head_decay": [], "head_no_decay": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        if id(param) in classifier_ids:
            groups["head_no_decay" if no_decay else "head_decay"].append(param)
        elif no_decay:
            groups["backbone_no_decay"].append(param)
        else:
            groups["backbone_decay"].append(param)
    result = []
    for key in ("backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"):
        params = groups[key]
        if not params:
            continue
        is_head = key.startswith("head_")
        no_decay = key.endswith("no_decay")
        result.append({"params": params,
                       "lr": float(lr_head if is_head else lr_backbone),
                       "weight_decay": 0.0 if no_decay else float(weight_decay)})
    if not result:
        raise ValueError("Model has no trainable parameters")
    return result


def count_params(model) -> float:
    """Return all parameters, including frozen ones, in millions."""
    return sum(param.numel() for param in model.parameters()) / 1_000_000.0


def count_gmacs(model, img_size: int = 224) -> float:
    """Estimate MACs from Conv2d/Linear layers with temporary forward hooks.

    Includes QK/AV matrix multiplications for timm token-attention modules. Norms,
    activations and pooling are excluded; this is an explicitly labelled MAC estimate.
    """
    import torch
    from torch import nn

    if img_size < 1:
        raise ValueError("img_size must be positive")
    total = [0]
    hooks = []

    def conv_macs(layer, _inputs, output):
        kh, kw = layer.kernel_size
        total[0] += output.numel() * (layer.in_channels // layer.groups) * kh * kw

    def linear_macs(layer, _inputs, output):
        total[0] += output.numel() * layer.in_features

    def attention_macs(layer, inputs, output):
        x = inputs[0]
        if x.ndim == 3:
            b, n, c = x.shape
            total[0] += 2 * b * n * n * c

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_macs))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_macs))
        if hasattr(layer, "qkv") and hasattr(layer, "num_heads"):
            hooks.append(layer.register_forward_hook(attention_macs))
    training_states = [(module, module.training) for module in model.modules()]
    try:
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype))
    finally:
        for hook in hooks:
            hook.remove()
        for module, was_training in training_states:
            module.training = was_training
    model.gmac_method = "Conv2d + Linear + attention QK/AV MACs; excludes norm/activation/pooling"
    return total[0] / 1_000_000_000.0
