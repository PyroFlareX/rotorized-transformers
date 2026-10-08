"""Generic module replacement by dotted path (architecture-independent)."""

from __future__ import annotations

from torch import nn


def get_module(model: nn.Module, path: str) -> nn.Module:
    return model.get_submodule(path)


def replace_module(model: nn.Module, path: str, new: nn.Module) -> nn.Module:
    """Swap ``model.<path>`` for ``new``; returns the old module. Raises if the path is missing."""
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    old = getattr(parent, attr)
    if not isinstance(old, nn.Module):
        raise TypeError(f"{path} is not a module")
    if hasattr(old, "in_features") and hasattr(new, "in_features"):
        if (old.in_features, old.out_features) != (new.in_features, new.out_features):
            raise ValueError(
                f"shape mismatch at {path}: {(old.out_features, old.in_features)} "
                f"vs {(new.out_features, new.in_features)}"
            )
    setattr(parent, attr, new)
    return old


def apply_replacements(model: nn.Module, layers: dict[str, nn.Module], dtype=None, device=None) -> dict[str, nn.Module]:
    """Replace every module in ``layers``; returns the originals (for golden comparisons / undo)."""
    originals = {}
    for path, new in layers.items():
        if dtype is not None or device is not None:
            new = new.to(device=device, dtype=dtype)
        originals[path] = replace_module(model, path, new)
    return originals
