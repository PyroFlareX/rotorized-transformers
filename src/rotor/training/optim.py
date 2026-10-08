"""Shared optimizer construction for rotor/scaler parameters (used by converters and training)."""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_LRS = {"angles": 1e-2, "scales": 1e-2, "residual": 1e-3}


def collect_param_groups(modules: list[nn.Module]) -> dict[str, list[nn.Parameter]]:
    """Merge ``param_groups()`` of every ``RotorCascade`` inside ``modules`` (``bias`` goes to ``scales``)."""
    from rotor.math.cascade import RotorCascade, RotorLinear

    groups: dict[str, list[nn.Parameter]] = {"angles": [], "scales": [], "residual": []}
    for root in modules:
        for m in root.modules():
            if isinstance(m, RotorCascade):
                for k, ps in m.param_groups().items():
                    groups[k].extend(ps)
            elif isinstance(m, RotorLinear) and m.bias is not None:
                groups["scales"].append(m.bias)
    return groups


def make_optimizer(
    groups: dict[str, list[nn.Parameter]],
    lrs: dict[str, float] | None = None,
    trainable: tuple[str, ...] = ("angles", "scales", "residual"),
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Adam with one param group per kind. Kinds not in ``trainable`` get ``requires_grad=False``."""
    lrs = {**DEFAULT_LRS, **(lrs or {})}
    param_groups = []
    for kind, params in groups.items():
        on = kind in trainable
        for p in params:
            p.requires_grad_(on)
        if on and params:
            param_groups.append({"params": params, "lr": lrs[kind], "name": kind, "weight_decay": weight_decay})
    if not param_groups:
        raise ValueError("no trainable parameters")
    return torch.optim.Adam(param_groups)
