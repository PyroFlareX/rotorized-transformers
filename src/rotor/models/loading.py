"""Load a base model, optionally with a ``rotor-scaler`` derived checkpoint applied on top."""

from __future__ import annotations

from pathlib import Path

import torch

from rotor.formats.checkpoint import load_structured_layers
from rotor.formats.manifest import Manifest
from rotor.models import qwen3_5
from rotor.models.replace import apply_replacements

ADAPTERS = {"qwen3_5": qwen3_5}


def adapter_for(config: dict):
    mt = config.get("model_type")
    if mt not in ADAPTERS:
        raise ValueError(f"no adapter for model_type={mt!r}; supported: {sorted(ADAPTERS)}")
    return ADAPTERS[mt]


def load_base(model_dir: Path, dtype: torch.dtype = torch.bfloat16, device=None):
    import json

    config = json.loads((Path(model_dir) / "config.json").read_text())
    return adapter_for(config).load_text_model(model_dir, dtype=dtype, device=device)


def load_converted(
    ckpt_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    device=None,
    base_dir: Path | None = None,
    structured_dtype: torch.dtype = torch.float32,
    rotor_backend: str = "reference",
) -> tuple[torch.nn.Module, Manifest]:
    """Base model from ``manifest.base.path`` (or ``base_dir``) with every manifest layer replaced.

    Structured layers run in ``structured_dtype`` internally and cast back to the activation dtype.
    ``rotor_backend="triton"`` explicitly replaces cascades with the optional fused inference backend.
    """
    if rotor_backend not in {"reference", "triton"}:
        raise ValueError(f"unsupported rotor_backend={rotor_backend!r}; expected 'reference' or 'triton'")

    manifest, layers = load_structured_layers(ckpt_dir, dtype=structured_dtype)
    model = load_base(Path(base_dir or manifest.base.path), dtype=dtype, device=device)
    apply_replacements(model, layers, device=device)
    if rotor_backend == "triton":
        from rotor.runtime import enable_fused_rotors

        enable_fused_rotors(model)
    return model, manifest
