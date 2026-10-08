"""End-to-end conversion: base checkpoint -> allocate byte budget -> fit every projection -> ``rotor-scaler`` dir.

Never downloads anything. If the base checkpoint is not on disk, ``convert`` raises ``FileNotFoundError`` telling
the user which Hugging Face repo/revision to place at which path.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from rotor.converters.budget import (
    allocate,
    dense_bytes,
    parse_target,
    resolve_byte_ratio,
    structured_bytes,
    target_name,
)
from rotor.converters.calibration import collect_inputs
from rotor.converters.fit import fit_activation, fit_weight
from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.environment import ResourceMonitor
from rotor.formats.checkpoint import layer_accounting, write_structured
from rotor.formats.manifest import BaseProvenance, LayerEntry, Manifest
from rotor.formats.spec import bytes_per_param
from rotor.math.cascade import CascadeSpec, RotorLinear
from rotor.math.topology import TOPOLOGY_KINDS
from rotor.models import qwen3_5
from rotor.models.discovery import TensorInfo, local_tensors
from rotor.reporting import ExperimentRun, make_run_id, markdown_table
from rotor.storage import Storage, StorageError, atomic_dir, load_storage, model_entry, sha256_file
from rotor.training.layerwise import FitSchedule

OBJECTIVES = ("weight", "activation")
RANDOM_CALIBRATION_NOTE = "synthetic random-token calibration; not representative of natural text"
_TOP_LEVEL_KEYS = (
    "name",
    "model",
    "families",
    "layers",
    "budget",
    "cascade",
    "fit",
    "calibration",
    "baselines",
    "targets",
    "topologies",
    "residual_ranks",
)
_BASELINES_DICT_KEYS = ("methods", "targets")


def _validate_target_list(targets: object, what: str) -> list[str]:
    if not isinstance(targets, list | tuple) or not targets:
        raise ValueError(f"{what} must be a non-empty list of compression targets, got {targets!r}")
    for t in targets:
        parse_target(t)  # raises ValueError
    return [str(t) for t in targets]


def _validate_baselines(baselines: object) -> list[str] | dict:
    """``baselines`` is a list of method names or ``{methods: [...], targets: [...]}``."""
    if isinstance(baselines, dict):
        unknown = sorted(set(baselines) - set(_BASELINES_DICT_KEYS))
        if unknown:
            raise ValueError(f"unknown baselines keys {unknown}; expected {list(_BASELINES_DICT_KEYS)}")
        out: dict = {}
        if baselines.get("methods") is not None:
            out["methods"] = _validate_method_names(baselines["methods"])
        if baselines.get("targets") is not None:
            out["targets"] = _validate_target_list(baselines["targets"], "baselines.targets")
        return out
    return _validate_method_names(baselines)


def _validate_method_names(methods: object) -> list[str]:
    if not isinstance(methods, list | tuple) or not methods or not all(isinstance(m, str) and m for m in methods):
        raise ValueError(f"baseline methods must be a non-empty list of method names, got {methods!r}")
    return list(methods)


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
    # Optional sweep/compare keys: they drive scripts/sweep_compression.py and scripts/compare_baselines.py and are
    # ignored by ``convert`` itself.
    baselines: list[str] | dict | None = None
    targets: list[str] | None = None
    topologies: list[str] | None = None
    residual_ranks: list[int] | None = None

    def __post_init__(self) -> None:
        if self.fit.get("objective", "weight") not in OBJECTIVES:
            raise ValueError(f"fit.objective must be one of {OBJECTIVES}, got {self.fit.get('objective')!r}")
        if self.calibration.get("source", "random") not in ("random", "text_file"):
            raise ValueError(f"calibration.source must be random|text_file, got {self.calibration.get('source')!r}")
        resolve_byte_ratio(self.budget)  # exactly one of target / byte_ratio, valid value
        if self.baselines is not None:
            self.baselines = _validate_baselines(self.baselines)
        if self.targets is not None:
            self.targets = _validate_target_list(self.targets, "targets")
        if self.topologies is not None:
            if not isinstance(self.topologies, list | tuple) or not self.topologies:
                raise ValueError(f"topologies must be a non-empty list, got {self.topologies!r}")
            bad = [t for t in self.topologies if t not in TOPOLOGY_KINDS]
            if bad:
                raise ValueError(f"unknown topologies {bad}; expected a subset of {TOPOLOGY_KINDS}")
            self.topologies = list(self.topologies)
        if self.residual_ranks is not None:
            ranks = self.residual_ranks
            if not isinstance(ranks, list | tuple) or not ranks:
                raise ValueError(f"residual_ranks must be a non-empty list of ints, got {ranks!r}")
            if any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in ranks):
                raise ValueError(f"residual_ranks must be ints >= 0, got {ranks!r}")
            self.residual_ranks = list(ranks)

    def byte_ratio(self) -> float:
        """Structured/dense byte ratio from the ``budget`` mapping (``target`` preset or ``byte_ratio``)."""
        return resolve_byte_ratio(self.budget)

    def target_label(self) -> str:
        """Preset name (``b025``) or ``bNNN`` for the configured byte ratio."""
        if self.budget.get("target") is not None and str(self.budget["target"]).startswith("b"):
            return str(self.budget["target"])
        return target_name(self.byte_ratio())

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
        for key in ("cascade", "fit", "calibration"):
            if d.get(key) is not None:
                kwargs[key] = {**getattr(defaults, key), **d[key]}
        if d.get("budget") is not None:
            # a user-given target/byte_ratio replaces (never merges with) the default byte_ratio
            budget = {k: v for k, v in defaults.budget.items() if k not in ("byte_ratio", "target")}
            budget.update(d["budget"])
            if "byte_ratio" not in budget and "target" not in budget:
                budget["byte_ratio"] = defaults.budget["byte_ratio"]
            kwargs["budget"] = budget
        for key in ("baselines", "targets", "topologies", "residual_ranks"):
            if d.get(key) is not None:
                kwargs[key] = json.loads(json.dumps(d[key]))
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConversionConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "model": self.model,
            "families": list(self.families),
            "layers": None if self.layers is None else list(self.layers),
            "budget": json.loads(json.dumps(self.budget)),
            "cascade": json.loads(json.dumps(self.cascade)),
            "fit": json.loads(json.dumps(self.fit)),
            "calibration": json.loads(json.dumps(self.calibration)),
        }
        for key in ("baselines", "targets", "topologies", "residual_ranks"):
            value = getattr(self, key)
            if value is not None:
                out[key] = json.loads(json.dumps(value))
        return out


def _policy(config: ConversionConfig) -> dict:
    budget = config.budget
    policy: dict = {"storage_dtype": budget.get("storage_dtype", "bfloat16")}
    if budget.get("target") is not None:
        policy["target"] = budget["target"]
    else:
        policy["byte_ratio"] = budget["byte_ratio"]
    policy.update(
        {
            "topology": config.cascade.get("topology", {"kind": "butterfly", "seed": 0}),
            "stages_per_rotor": config.cascade.get("stages_per_rotor", "auto"),
            "residual_rank": config.cascade.get("residual_rank", 0),
            "adapter": config.cascade.get("adapter", "pad_truncate"),
        }
    )
    return policy


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


def _bias_numel(tensors: dict[str, TensorInfo], weight_name: str) -> int:
    info = tensors.get(weight_name[: -len("weight")] + "bias")
    if info is None:
        return 0
    n = 1
    for s in info.shape:
        n *= s
    return n


def _layer_plan(
    specs: dict[str, CascadeSpec],
    projections: list[qwen3_5.Projection],
    storage_dtype: str,
    tensors: dict[str, TensorInfo],
) -> list[dict]:
    """Per-layer plan. ``structured_bytes`` equals what ``write_structured`` serializes for the layer
    (cascade parameters plus the dense bias, if the projection has one, stored in ``storage_dtype``)."""
    plan = []
    for p in projections:
        spec = specs[p.module]
        bias_bytes = _bias_numel(tensors, p.tensor) * bytes_per_param(storage_dtype)
        plan.append(
            {
                "module": p.module,
                "family": p.family,
                "in": p.in_features,
                "out": p.out_features,
                "dense_dtype": p.dtype,
                "dense_params": p.in_features * p.out_features,
                "num_rotors": spec.num_rotors,
                "stages_per_rotor": spec.stages_per_rotor,
                "params": spec.param_count(),
                "bias_bytes": bias_bytes,
                "structured_bytes": structured_bytes(spec, storage_dtype) + bias_bytes,
                "dense_bytes": dense_bytes(p),
            }
        )
    return plan


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


def _unique_run_id(run: ExperimentRun) -> None:
    """Run ids have second resolution; bump the timestamp if a run with the same id was already published."""
    while (run.out_root / run.run_id).exists():
        run.created += timedelta(seconds=1)
        run.run_id = make_run_id(run.model, run.method, run.target, run.created)


def _layer_rows(manifest: Manifest, fit_seconds: dict[str, float]) -> list[dict]:
    rows = []
    for e in manifest.layers:
        row = {
            "module": e.module,
            "family": e.family,
            "in": e.spec.in_features,
            "out": e.spec.out_features,
            "dense_dtype": e.dense_dtype,
            "dense_params": e.dense_params,
            "dense_bytes": e.dense_bytes,
            "num_rotors": e.spec.num_rotors,
            "structured_bytes": e.structured_bytes,
            "byte_ratio": e.structured_bytes / e.dense_bytes if e.dense_bytes else None,
            "angles": e.learned_scalars.get("angles"),
            "scales": e.learned_scalars.get("scales"),
            "residual": e.learned_scalars.get("residual"),
            "bias": e.learned_scalars.get("bias"),
            "rel_fro": e.metrics.get("weight", {}).get("rel_fro"),
            "fit_wall_seconds": fit_seconds.get(e.module),
        }
        act = e.metrics.get("activation")
        if act is not None:
            row["act_rel_fro"] = act.get("rel_fro")
            row["act_mean_row_cosine"] = act.get("mean_row_cosine")
        rows.append(row)
    return rows


def _fill_run(
    run: ExperimentRun,
    config: ConversionConfig,
    manifest: Manifest,
    dest: Path,
    original_checkpoint_bytes: int,
    resources: dict,
    write_seconds: float,
    fit_seconds: dict[str, float],
) -> None:
    totals = manifest.totals()
    ser = manifest.serialized
    rows = _layer_rows(manifest, fit_seconds)
    derived_bytes = ser["file_bytes"] + ser["manifest_bytes"]
    run.metrics.update(
        {
            "storage": {
                "original_checkpoint_bytes": original_checkpoint_bytes,
                "converted_dense_bytes": totals["dense_bytes"],
                "structured_tensor_bytes": totals["structured_bytes"],
                "layer_byte_ratio": totals["byte_ratio"],
                "derived_weights_file_bytes": ser["file_bytes"],
                "derived_manifest_bytes": ser["manifest_bytes"],
                "derived_checkpoint_bytes": derived_bytes,
                "file_compression_ratio": ser["file_bytes"] / totals["dense_bytes"],
                "safetensors_header_bytes": ser["header_bytes"],
                "padding_bytes": ser["padding_bytes"],
                "learned_scalar_count": totals["learned_scalar_count"],
                "index_bytes": totals["index_bytes"],
                "quantization_metadata_bytes": totals["quantization_metadata_bytes"],
            },
            "resources": resources,
            "write_wall_seconds": write_seconds,
            "layers": rows,
        }
    )
    rel = [r["rel_fro"] for r in rows if r["rel_fro"] is not None]
    if rel:
        run.metrics["quality"] = {
            "mean_weight_rel_fro": sum(rel) / len(rel),
            "max_weight_rel_fro": max(rel),
            "min_weight_rel_fro": min(rel),
        }
    run.estimates.update(
        {
            "whole_model_bytes_if_unconverted_tensors_kept": (
                original_checkpoint_bytes - totals["dense_bytes"] + derived_bytes
            ),
            "whole_model_ratio_if_unconverted_tensors_kept": (
                (original_checkpoint_bytes - totals["dense_bytes"] + derived_bytes) / original_checkpoint_bytes
            ),
            "note": (
                "arithmetic on measured byte counts: original checkpoint with the converted tensors replaced by the "
                "derived file; not a measured file"
            ),
            "flops_per_token_total": sum(e.spec.flops_per_token() for e in manifest.layers),
            "flops_per_token_dense_total": sum(2 * e.dense_params for e in manifest.layers),
        }
    )
    run.summary = (
        f"Converted {len(rows)} projections ({', '.join(sorted({e.family for e in manifest.layers}))}) of "
        f"`{config.model}` at target `{run.target}` (byte ratio {config.byte_ratio():.4g}). Serialized weights file "
        f"{ser['file_bytes']} B vs {totals['dense_bytes']} B dense converted weights "
        f"(file ratio {ser['file_bytes'] / totals['dense_bytes']:.4f}). Output: `{dest}`."
    )
    run.interpretation = (
        "Only weight-space (and, for the activation objective, layer-input) reconstruction errors of the fitted "
        "projections are reported here; no model-quality (perplexity/task) result is implied. Compare against "
        "equal-byte baselines (truncated SVD etc.) before attributing any quality to the rotor structure."
    )
    run.add_section(
        "Per-layer conversion",
        markdown_table(
            rows,
            [
                "module",
                "family",
                "in",
                "out",
                "dense_dtype",
                "dense_bytes",
                "num_rotors",
                "structured_bytes",
                "byte_ratio",
                "angles",
                "scales",
                "residual",
                "rel_fro",
                *(["act_rel_fro"] if any("act_rel_fro" in r for r in rows) else []),
                "fit_wall_seconds",
            ],
        ),
    )
    run.add_section(
        "Byte accounting",
        markdown_table(
            [
                {"item": "original checkpoint (sum of *.safetensors)", "bytes": original_checkpoint_bytes},
                {"item": "converted tensors, dense (checkpoint dtype)", "bytes": totals["dense_bytes"]},
                {"item": "converted tensors, serialized tensor payload", "bytes": totals["structured_bytes"]},
                {"item": "rotor_weights.safetensors on disk", "bytes": ser["file_bytes"]},
                {"item": "  safetensors header", "bytes": ser["header_bytes"]},
                {"item": "  alignment padding", "bytes": ser["padding_bytes"]},
                {"item": "rotor_manifest.json", "bytes": ser["manifest_bytes"]},
                {"item": "index bytes (deterministic topology)", "bytes": totals["index_bytes"]},
                {"item": "quantization metadata (float storage)", "bytes": totals["quantization_metadata_bytes"]},
            ],
            ["item", "bytes"],
        ),
    )


def convert(
    config: ConversionConfig,
    storage: Storage | None = None,
    base_dir: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    report: bool = True,
    report_root: Path | None = None,
    run_info: dict | None = None,
) -> Path | dict:
    """Convert the configured projections of a local base checkpoint.

    Returns the output directory, or (``dry_run=True``) the byte plan computed from safetensors headers only.
    Unless ``report=False`` the conversion is also published as an ``ExperimentRun`` (kind ``convert``) under
    ``report_root`` (default ``reports/experiments``); its directory is stored in ``run_info["run_dir"]`` if a dict
    is given.
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
    plan = _layer_plan(specs, projections, storage_dtype, tensors)
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
    original_checkpoint_bytes = sum(f.stat().st_size for f in sorted(base.glob("*.safetensors")))
    conversion_config["original_checkpoint_bytes"] = original_checkpoint_bytes

    run: ExperimentRun | None = None
    if report:
        topo_kind = config.cascade.get("topology", {"kind": "butterfly"})["kind"]
        run = ExperimentRun(
            "convert",
            model=config.model,
            method=f"rotor-{topo_kind}-{objective}",
            target=config.target_label(),
            config={**config.to_dict(), "base_dir": os.path.abspath(base), "output_dir": str(dest)},
            out_root=report_root,
        )
        _unique_run_id(run)

    def emit(msg: str) -> None:
        if run is not None:
            run.log(msg, echo=False)
        log(msg)

    with contextlib.ExitStack() as stack:
        if run is not None:
            stack.enter_context(run)

        inputs: dict[str, torch.Tensor] = {}
        layer_entries: list[LayerEntry] = []
        layers: dict[str, RotorLinear] = {}
        fit_seconds: dict[str, float] = {}
        with ResourceMonitor() as monitor:
            if objective == "activation":
                windows = _calibration_windows(config, base, config_json)
                conversion_config["windows_fingerprint"] = windows_fingerprint(windows)
                if config.calibration.get("source", "random") == "random":
                    notes.append(RANDOM_CALIBRATION_NOTE)
                notes.append(
                    "calibration inputs captured from the dense base model (not propagated through converted layers)"
                )
                emit(
                    f"calibration: {tuple(windows.shape)} token windows, capturing inputs of {len(projections)} modules"
                )
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

            plan_by_module = {r["module"]: r for r in plan}
            for i, p in enumerate(projections):
                spec = specs[p.module]
                W = _dense_weight(base, tensors, p.tensor)
                t0 = time.perf_counter()
                if objective == "weight":
                    cascade, metrics = fit_weight(W, spec, schedule, seed, init_angle_std, device, emit)
                else:
                    cascade, metrics = fit_activation(
                        W, inputs[p.module], spec, schedule, seed, init_angle_std, None, device, emit
                    )
                fit_seconds[p.module] = time.perf_counter() - t0
                metrics["fit_wall_seconds"] = fit_seconds[p.module]
                bias = _dense_bias(base, tensors, p.tensor)  # dense bias is stored verbatim in the storage dtype
                layer = RotorLinear(cascade.to("cpu"), bias)
                layers[p.module] = layer
                serialized_bytes, scalars = layer_accounting(p.module, layer, storage_dtype)
                if serialized_bytes != plan_by_module[p.module]["structured_bytes"]:
                    raise RuntimeError(
                        f"{p.module}: serialized tensor bytes {serialized_bytes} != planned "
                        f"{plan_by_module[p.module]['structured_bytes']}"
                    )
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
                        structured_bytes=serialized_bytes,
                        metrics=metrics,
                        learned_scalars=scalars,
                        dense_dtype=p.dtype,
                    )
                )
                emit(
                    f"[{i + 1}/{len(projections)}] {p.module} ({p.family}) rotors={spec.num_rotors} "
                    f"rel_fro={metrics['weight']['rel_fro']:.4g} final_loss={metrics['final_loss']:.4g} "
                    f"fit={fit_seconds[p.module]:.2f}s"
                )

            manifest = Manifest(
                base=_provenance(base, entry, config_json),
                layers=layer_entries,
                conversion_config=conversion_config,
                notes=notes,
            )
        conversion_config["resources"] = {
            **monitor.as_dict(),
            "scope": "base-shard hashing, calibration capture and all layer fits; excludes the final write",
            "peak_rss_note": "process high-water mark (getrusage), an upper bound for this region",
        }
        t0 = time.perf_counter()
        with atomic_dir(dest) as tmp:
            write_structured(tmp, manifest, layers, base)
        write_seconds = time.perf_counter() - t0
        emit(f"wrote {dest}: {manifest.totals()}")
        if run is not None:
            _fill_run(
                run,
                config,
                manifest,
                dest,
                original_checkpoint_bytes,
                conversion_config["resources"],
                write_seconds,
                fit_seconds,
            )
    if run is not None and run_info is not None:
        run_info["run_dir"] = run.path
    return dest
