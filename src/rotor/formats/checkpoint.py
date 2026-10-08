"""Export/import of structured checkpoints (``rotor-scaler`` format, see ``rotor.formats.spec``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rotor.formats.manifest import Manifest
from rotor.formats.spec import BASE_METADATA_DIRNAME, MANIFEST_FILENAME, WEIGHTS_FILENAME
from rotor.math.cascade import RotorCascade, RotorLinear

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Small base files copied for provenance (never weights).
BASE_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "LICENSE",
)


def module_tensors(name: str, layer: RotorLinear, storage_dtype: str) -> dict[str, torch.Tensor]:
    dt = _DTYPES[storage_dtype]
    c = layer.cascade
    out = {
        f"{name}.angles": c.angles.detach().to(dt).contiguous(),
        f"{name}.d_in": c.d_in.detach().to(dt).contiguous(),
        f"{name}.d_mid": c.d_mid.detach().to(dt).contiguous(),
        f"{name}.d_out": c.d_out.detach().to(dt).contiguous(),
    }
    if c.residual is not None:
        out[f"{name}.residual.U"] = c.residual.U.detach().to(dt).contiguous()
        out[f"{name}.residual.V"] = c.residual.V.detach().to(dt).contiguous()
    if layer.bias is not None:
        out[f"{name}.bias"] = layer.bias.detach().to(dt).contiguous()
    return out


def write_structured(out_dir: Path, manifest: Manifest, layers: dict[str, RotorLinear], base_dir: Path | None) -> None:
    """Write manifest + tensors into ``out_dir`` (callers pass a temp dir from ``storage.atomic_dir``)."""
    by_module = {e.module: e for e in manifest.layers}
    if set(by_module) != set(layers):
        raise ValueError("manifest layers and provided modules differ")
    tensors: dict[str, torch.Tensor] = {}
    for name, layer in layers.items():
        tensors.update(module_tensors(name, layer, by_module[name].storage_dtype))
    save_file(tensors, str(out_dir / WEIGHTS_FILENAME), metadata={"format": manifest.format})
    (out_dir / MANIFEST_FILENAME).write_text(manifest.to_json())
    if base_dir is not None:
        meta = out_dir / BASE_METADATA_DIRNAME
        meta.mkdir()
        for fname in BASE_METADATA_FILES:
            src = Path(base_dir) / fname
            if src.is_file():
                shutil.copy2(src, meta / fname)


def read_manifest(ckpt_dir: Path) -> Manifest:
    return Manifest.from_json((Path(ckpt_dir) / MANIFEST_FILENAME).read_text())


def load_structured_layers(
    ckpt_dir: Path, dtype: torch.dtype = torch.float32, device=None
) -> tuple[Manifest, dict[str, RotorLinear]]:
    """Rebuild every ``RotorLinear`` described by the manifest (parameters cast to ``dtype``)."""
    manifest = read_manifest(ckpt_dir)
    tensors = load_file(str(Path(ckpt_dir) / WEIGHTS_FILENAME), device="cpu")
    layers: dict[str, RotorLinear] = {}
    for e in manifest.layers:
        casc = RotorCascade(e.spec, dtype=dtype, device=device)
        state = {
            "angles": tensors[f"{e.module}.angles"],
            "d_in": tensors[f"{e.module}.d_in"],
            "d_mid": tensors[f"{e.module}.d_mid"],
            "d_out": tensors[f"{e.module}.d_out"],
        }
        if e.spec.residual_rank > 0:
            state["residual.U"] = tensors[f"{e.module}.residual.U"]
            state["residual.V"] = tensors[f"{e.module}.residual.V"]
        casc.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
        bias = tensors.get(f"{e.module}.bias") if e.has_bias else None
        layers[e.module] = RotorLinear(casc, bias.to(dtype) if bias is not None else None).to(device)
    return manifest, layers
