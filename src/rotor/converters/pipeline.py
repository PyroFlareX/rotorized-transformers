"""End-to-end conversion: base checkpoint -> allocate byte budget -> fit every projection -> ``rotor-scaler`` dir.

Never downloads anything. If the base checkpoint is not on disk, ``convert`` raises ``FileNotFoundError`` telling
the user which Hugging Face repo/revision to place at which path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from rotor.converters.budget import allocate, dense_bytes, structured_bytes
from rotor.converters.calibration import collect_inputs
from rotor.converters.fit import fit_activation, fit_weight
from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.formats.checkpoint import write_structured
from rotor.formats.manifest import BaseProvenance, LayerEntry, Manifest
from rotor.math.cascade import CascadeSpec, RotorLinear
from rotor.models import qwen3_5
from rotor.models.discovery import TensorInfo, local_tensors
from rotor.storage import Storage, StorageError, atomic_dir, load_storage, model_entry, sha256_file
from rotor.training.layerwise import FitSchedule

OBJECTIVES = ("weight", "activation")
RANDOM_CALIBRATION_NOTE = "synthetic random-token calibration; not representative of natural text"
_TOP_LEVEL_KEYS = ("name", "model", "families", "layers", "budget", "cascade", "fit", "calibration")


@dataclass
class ConversionConfig:
    name: str
    model: str
    families: list[str] = field(default_factory=lambda: ["ffn.gate", "ffn.up", "ffn.down"])
    layers: list[int] | None = None
    budget: dict = field(default_factory=lambda: {"byte_ratio": 0.25, "storage_dtype": "bfloat16"})
    cascade: dict = field(
        default_factory=lambda: {
            "topology": {"kind": "butterfly", "seed": 0},
            "stages_per_rotor": "auto",
            "residual_rank": 0,
            "adapter": "pad_truncate",
        }
    )
    fit: dict = field(
        default_factory=lambda: {
            "objective": "weight",
            "seed": 0,
            "device": "cpu",
            "init_angle_std": 0.01,
            "schedule": {},
        }
    )
    calibration: dict = field(
        default_factory=lambda: {
            "source": "random",
            "path": None,
            "seq_len": 256,
            "num_windows": 8,
            "seed": 0,
            "max_rows": 4096,
        }
    )

    def __post_init__(self) -> None:
        if self.fit.get("objective", "weight") not in OBJECTIVES:
            raise ValueError(f"fit.objective must be one of {OBJECTIVES}, got {self.fit.get('objective')!r}")
        if self.calibration.get("source", "random") not in ("random", "text_file"):
            raise ValueError(f"calibration.source must be random|text_file, got {self.calibration.get('source')!r}")

    @classmethod
    def from_dict(cls, d: dict) -> ConversionConfig:
        unknown = sorted(set(d) - set(_TOP_LEVEL_KEYS))
        if unknown:
            raise ValueError(f"unknown conversion config keys {unknown}; expected {list(_TOP_LEVEL_KEYS)}")
        for required in ("name", "model"):
            if required not in d:
                raise ValueError(f"conversion config is missing required key {required!r}")
        defaults = cls(name=d["name"], model=d["model"])
        kwargs: dict = {"name": d["name"], "model": d["model"]}
        if d.get("families") is not None:
            kwargs["families"] = list(d["families"])
        if d.get("layers") is not None:
            kwargs["layers"] = [int(i) for i in d["layers"]]
        for key in ("budget", "cascade", "fit", "calibration"):
            if d.get(key) is not None:
                kwargs[key] = {**getattr(defaults, key), **d[key]}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConversionConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "families": list(self.families),
            "layers": None if self.layers is None else list(self.layers),
            "budget": json.loads(json.dumps(self.budget)),
            "cascade": json.loads(json.dumps(self.cascade)),
            "fit": json.loads(json.dumps(self.fit)),
            "calibration": json.loads(json.dumps(self.calibration)),
        }


def _policy(config: ConversionConfig) -> dict:
    return {
        "byte_ratio": config.budget["byte_ratio"],
        "storage_dtype": config.budget.get("storage_dtype", "bfloat16"),
        "topology": config.cascade.get("topology", {"kind": "butterfly", "seed": 0}),
        "stages_per_rotor": config.cascade.get("stages_per_rotor", "auto"),
        "residual_rank": config.cascade.get("residual_rank", 0),
        "adapter": config.cascade.get("adapter", "pad_truncate"),
    }


def _resolve_base_dir(config: ConversionConfig, storage: Storage, base_dir: Path | None) -> Path:
    _, entry = model_entry(config.model)
    target = Path(base_dir) if base_dir is not None else storage.base_path(entry["local_path"])
    if not target.is_dir():
        raise FileNotFoundError(
            f"base checkpoint for {entry['hf_id']} (revision {entry['revision']}) not found at {target}. "
            f"No download is performed: place the files of that exact repository revision at {target} "
            f"(e.g. `huggingface-cli download {entry['hf_id']} --revision {entry['revision']} --local-dir {target}`)."
        )
    return target


def _dense_weight(base_dir: Path, tensors: dict[str, TensorInfo], name: str) -> torch.Tensor:
    with safe_open(str(base_dir / tensors[name].shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float32)


def _dense_bias(base_dir: Path, tensors: dict[str, TensorInfo], weight_name: str) -> torch.Tensor | None:
    bias_name = weight_name[: -len("weight")] + "bias"
    if bias_name not in tensors:
        return None
    return _dense_weight(base_dir, tensors, bias_name)


def _layer_plan(specs: dict[str, CascadeSpec], projections: list[qwen3_5.Projection], storage_dtype: str) -> list[dict]:
    return [
        {
            "module": p.module,
            "family": p.family,
            "in": p.in_features,
            "out": p.out_features,
            "num_rotors": specs[p.module].num_rotors,
            "stages_per_rotor": specs[p.module].stages_per_rotor,
            "params": specs[p.module].param_count(),
            "structured_bytes": structured_bytes(specs[p.module], storage_dtype),
            "dense_bytes": dense_bytes(p),
        }
        for p in projections
    ]


def _provenance(base_dir: Path, entry: dict, config_json: dict) -> BaseProvenance:
    files = sorted(base_dir.glob("*.safetensors")) + [base_dir / "config.json"]
    tc = qwen3_5.text_config(config_json)
    tied = config_json.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
    return BaseProvenance(
        hf_id=entry.get("hf_id"),
        revision=entry.get("revision"),
        path=os.path.abspath(base_dir),
        file_sha256={f.name: sha256_file(f) for f in files},
        tied_word_embeddings=None if tied is None else bool(tied),
        architectures=list(config_json.get("architectures", [])),
    )


def _calibration_windows(config: ConversionConfig, base_dir: Path, config_json: dict) -> torch.Tensor:
    cal = config.calibration
    seq_len, num_windows, seed = int(cal["seq_len"]), int(cal["num_windows"]), int(cal.get("seed", 0))
    if cal.get("source", "random") == "random":
        vocab = int(qwen3_5.text_config(config_json)["vocab_size"])
        return random_windows(vocab, seq_len, num_windows, seed)
    if not cal.get("path"):
        raise ValueError("calibration.source=text_file requires calibration.path")
    tokenizer = qwen3_5.load_tokenizer(base_dir)
    ids = tokenize_text_file(tokenizer, Path(cal["path"]))
    return windows_from_ids(ids, seq_len, num_windows, seed)


def convert(
    config: ConversionConfig,
    storage: Storage | None = None,
    base_dir: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> Path | dict:
    """Convert the configured projections of a local base checkpoint.

    Returns the output directory, or (``dry_run=True``) the byte plan computed from safetensors headers only.
    """
    storage = storage or load_storage()
    _, entry = model_entry(config.model)
    base = _resolve_base_dir(config, storage, base_dir)
    config_json = json.loads((base / "config.json").read_text())
    tensor_list = local_tensors(base)
    tensors = {t.name: t for t in tensor_list}
    projections = qwen3_5.select_projections(
        qwen3_5.discover_projections(tensor_list, config_json), config.families, config.layers
    )
    if not projections:
        raise ValueError(f"no projections of families {config.families} (layers={config.layers}) found in {base}")
    storage_dtype = config.budget.get("storage_dtype", "bfloat16")
    specs = allocate(projections, _policy(config))
    plan = _layer_plan(specs, projections, storage_dtype)
    dense_total = sum(r["dense_bytes"] for r in plan)
    struct_total = sum(r["structured_bytes"] for r in plan)

    if dry_run:
        return {
            "model": config.model,
            "base_dir": os.path.abspath(base),
            "layers": plan,
            "totals": {
                "dense_bytes": dense_total,
                "structured_bytes": struct_total,
                "byte_ratio": struct_total / dense_total,
            },
        }

    dest = Path(out_dir) if out_dir is not None else storage.derived_path(config.name)
    storage.assert_writable(dest)
    if dest.exists():
        raise StorageError(f"refusing to overwrite existing output {dest}")

    objective = config.fit.get("objective", "weight")
    seed = int(config.fit.get("seed", 0))
    device = config.fit.get("device", "cpu")
    init_angle_std = float(config.fit.get("init_angle_std", 0.01))
    schedule = FitSchedule.from_dict(config.fit.get("schedule") or {})
    conversion_config = config.to_dict()
    notes: list[str] = []

    inputs: dict[str, torch.Tensor] = {}
    if objective == "activation":
        windows = _calibration_windows(config, base, config_json)
        conversion_config["windows_fingerprint"] = windows_fingerprint(windows)
        if config.calibration.get("source", "random") == "random":
            notes.append(RANDOM_CALIBRATION_NOTE)
        notes.append("calibration inputs captured from the dense base model (not propagated through converted layers)")
        log(f"calibration: {tuple(windows.shape)} token windows, capturing inputs of {len(projections)} modules")
        model = qwen3_5.load_text_model(base, dtype=torch.bfloat16, device="cpu")
        try:
            inputs = collect_inputs(
                model,
                [p.module for p in projections],
                windows,
                max_rows=int(config.calibration.get("max_rows", 4096)),
                seed=int(config.calibration.get("seed", 0)),
            )
        finally:
            del model

    layer_entries: list[LayerEntry] = []
    layers: dict[str, RotorLinear] = {}
    for i, p in enumerate(projections):
        spec = specs[p.module]
        W = _dense_weight(base, tensors, p.tensor)
        if objective == "weight":
            cascade, metrics = fit_weight(W, spec, schedule, seed, init_angle_std, device, log)
        else:
            cascade, metrics = fit_activation(
                W, inputs[p.module], spec, schedule, seed, init_angle_std, None, device, log
            )
        bias = _dense_bias(base, tensors, p.tensor)  # dense bias is kept verbatim; excluded from both byte counts
        layers[p.module] = RotorLinear(cascade.to("cpu"), bias)
        layer_entries.append(
            LayerEntry(
                module=p.module,
                source_tensor=p.tensor,
                family=p.family,
                spec=spec,
                storage_dtype=storage_dtype,
                has_bias=bias is not None,
                dense_params=p.in_features * p.out_features,
                structured_params=spec.param_count(),
                dense_bytes=dense_bytes(p),
                structured_bytes=structured_bytes(spec, storage_dtype),
                metrics=metrics,
            )
        )
        log(
            f"[{i + 1}/{len(projections)}] {p.module} ({p.family}) rotors={spec.num_rotors} "
            f"rel_fro={metrics['weight']['rel_fro']:.4g} final_loss={metrics['final_loss']:.4g}"
        )

    manifest = Manifest(
        base=_provenance(base, entry, config_json),
        layers=layer_entries,
        conversion_config=conversion_config,
        notes=notes,
    )
    with atomic_dir(dest) as tmp:
        write_structured(tmp, manifest, layers, base)
    log(f"wrote {dest}: {manifest.totals()}")
    return dest
