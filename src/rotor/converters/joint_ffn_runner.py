"""End-to-end Phase 4B joint FFN fitting from approved local calibration and held-out text."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from rotor.converters.budget import auto_stages, resolve_byte_ratio, target_name
from rotor.converters.fit import fit_joint_ffn
from rotor.converters.snapshots import cpu_copy
from rotor.data import tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.environment import ResourceMonitor
from rotor.formats.checkpoint import layer_accounting, write_structured
from rotor.formats.manifest import BaseProvenance, LayerEntry, Manifest
from rotor.formats.spec import bytes_per_param
from rotor.math.cascade import CascadeSpec
from rotor.math.topology import TOPOLOGY_KINDS, Topology
from rotor.models import qwen3_5
from rotor.models.discovery import TensorInfo, local_tensors
from rotor.models.qwen3_5 import Projection
from rotor.reporting import ExperimentRun, markdown_table
from rotor.storage import Storage, StorageError, atomic_dir, load_storage, model_entry, sha256_file
from rotor.training.layerwise import FitSchedule

_ALLOWED_TOP = {"name", "model", "layer", "budget", "cascade", "fit", "calibration", "heldout"}
_ALLOWED_DATA = {"path", "seq_len", "num_windows", "seed", "max_rows"}
_ALLOWED_FIT = {"device", "seed", "init_angle_std", "schedule"}
_ALLOWED_CASCADE = {"topology", "stages_per_rotor", "residual_rank", "adapter"}
_ALLOWED_BUDGET = {"target", "byte_ratio", "storage_dtype"}
_FAMILIES = {"ffn.gate": "gate", "ffn.up": "up", "ffn.down": "down"}
_MODULE_SUFFIX = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
_SNAPSHOT_SCHEMA = 1
_SNAPSHOT_FORMAT = "rotor-joint-ffn-snapshot"
_SNAPSHOT_OWNER_FORMAT = "rotor-joint-ffn-snapshot-owner"
_SNAPSHOT_NAME = re.compile(r"^joint-snapshot-(\d+)-([0-9a-f]{32})$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _section(config: dict, name: str, allowed: set[str], defaults: dict) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys {unknown}; expected {sorted(allowed)}")
    return {**defaults, **value}


def load_joint_config(path: str | Path) -> dict:
    """Load, validate, and resolve a joint-fit YAML file and its config-relative text paths."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("joint FFN config must be a YAML mapping")
    unknown = sorted(set(raw) - _ALLOWED_TOP)
    if unknown:
        raise ValueError(f"unknown joint FFN config keys {unknown}; expected {sorted(_ALLOWED_TOP)}")
    missing = sorted({"name", "model", "layer"} - set(raw))
    if missing:
        raise ValueError(f"joint FFN config is missing required keys {missing}")
    config = copy.deepcopy(raw)
    if not isinstance(config["name"], str) or not _SAFE_NAME.fullmatch(config["name"]):
        raise ValueError(
            "name must be a simple output directory name containing letters, digits, dot, underscore, or dash"
        )
    if not isinstance(config["model"], str) or not config["model"]:
        raise ValueError("model must be a non-empty key in configs/models.yaml")
    layer = config["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("layer must be a non-negative integer")

    budget_value = config.get("budget", {})
    if not isinstance(budget_value, dict):
        raise ValueError("budget must be a mapping")
    unknown_budget = sorted(set(budget_value) - _ALLOWED_BUDGET)
    if unknown_budget:
        raise ValueError(f"unknown budget keys {unknown_budget}; expected {sorted(_ALLOWED_BUDGET)}")
    if "target" in budget_value and "byte_ratio" in budget_value:
        raise ValueError("budget must set exactly one of target or byte_ratio")
    budget = {"byte_ratio": 0.25, "storage_dtype": "bfloat16"}
    if "target" in budget_value:
        budget.pop("byte_ratio")
    budget.update(budget_value)
    ratio = resolve_byte_ratio(budget)
    if not math.isfinite(ratio) or ratio > 1:
        raise ValueError("budget byte_ratio must be finite and in (0, 1]")
    bytes_per_param(budget["storage_dtype"])
    config["budget"] = budget

    cascade = _section(
        config,
        "cascade",
        _ALLOWED_CASCADE,
        {
            "topology": {"kind": "butterfly", "seed": 0},
            "stages_per_rotor": "auto",
            "residual_rank": 0,
            "adapter": "pad_truncate",
        },
    )
    topology = cascade["topology"]
    if not isinstance(topology, dict) or set(topology) - {"kind", "seed"}:
        raise ValueError("cascade.topology must contain only kind and optional seed")
    if topology.get("kind", "butterfly") not in TOPOLOGY_KINDS:
        raise ValueError(f"cascade.topology.kind must be one of {TOPOLOGY_KINDS}")
    topology.setdefault("kind", "butterfly")
    topology.setdefault("seed", 0)
    if isinstance(topology["seed"], bool) or not isinstance(topology["seed"], int):
        raise ValueError("cascade.topology.seed must be an integer")
    stages = cascade["stages_per_rotor"]
    if stages != "auto" and (isinstance(stages, bool) or not isinstance(stages, int) or stages < 0):
        raise ValueError("cascade.stages_per_rotor must be 'auto' or an integer >= 0")
    if (
        isinstance(cascade["residual_rank"], bool)
        or not isinstance(cascade["residual_rank"], int)
        or cascade["residual_rank"] < 0
    ):
        raise ValueError("cascade.residual_rank must be an integer >= 0")
    if cascade["adapter"] != "pad_truncate":
        raise ValueError("cascade.adapter must be 'pad_truncate'")
    config["cascade"] = cascade

    fit = _section(
        config,
        "fit",
        _ALLOWED_FIT,
        {"device": "cpu", "seed": 0, "init_angle_std": 0.01, "schedule": {}},
    )
    if not isinstance(fit["seed"], int) or isinstance(fit["seed"], bool):
        raise ValueError("fit.seed must be an integer")
    if (
        not isinstance(fit["init_angle_std"], (int, float))
        or not math.isfinite(fit["init_angle_std"])
        or fit["init_angle_std"] < 0
    ):
        raise ValueError("fit.init_angle_std must be finite and >= 0")
    if not isinstance(fit["device"], str) or not fit["device"]:
        raise ValueError("fit.device must be a torch device string")
    if not isinstance(fit["schedule"], dict):
        raise ValueError("fit.schedule must be a mapping")
    FitSchedule.from_dict(fit["schedule"])
    config["fit"] = fit

    for section_name in ("calibration", "heldout"):
        section = _section(
            config,
            section_name,
            _ALLOWED_DATA,
            {"path": None, "seq_len": 256, "num_windows": 8, "seed": 0, "max_rows": 4096},
        )
        if not isinstance(section["path"], str) or not section["path"].strip():
            raise ValueError(
                f"{section_name}.path must name an existing, user-approved local text file; "
                "there is no random-token fallback"
            )
        text_path = Path(section["path"])
        if not text_path.is_absolute():
            text_path = path.parent / text_path
        text_path = text_path.resolve()
        if not text_path.is_file():
            raise FileNotFoundError(f"{section_name}.path does not exist: {text_path}")
        section["path"] = str(text_path)
        for key in ("seq_len", "num_windows", "max_rows"):
            if isinstance(section[key], bool) or not isinstance(section[key], int) or section[key] < 1:
                raise ValueError(f"{section_name}.{key} must be an integer >= 1")
        if isinstance(section["seed"], bool) or not isinstance(section["seed"], int):
            raise ValueError(f"{section_name}.seed must be an integer")
        config[section_name] = section
    if config["calibration"]["seq_len"] != config["heldout"]["seq_len"]:
        raise ValueError("calibration and heldout seq_len must match so token windows can be checked for overlap")
    return config


def _load_joint_weights(base: Path, projections: list[Projection], infos: dict[str, TensorInfo]):
    weights: dict[str, torch.Tensor] = {}
    biases: dict[str, torch.Tensor | None] = {}
    projection_by_key: dict[str, Projection] = {}
    for projection in projections:
        key = _FAMILIES[projection.family]
        projection_by_key[key] = projection
        info = infos[projection.tensor]
        with safe_open(str(base / info.shard), framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(projection.tensor).contiguous()
            bias_name = projection.tensor.removesuffix("weight") + "bias"
            bias = tensors.get_tensor(bias_name).contiguous() if bias_name in infos else None
        weights[key], biases[key] = weight, bias
    if set(weights) != {"gate", "up", "down"}:
        raise ValueError(f"layer {projections[0].layer} must have exactly one gate, up, and down projection")
    return weights, biases, projection_by_key


def _read_windows(tokenizer, section: dict) -> torch.Tensor:
    ids = tokenize_text_file(tokenizer, Path(section["path"]))
    return windows_from_ids(ids, int(section["seq_len"]), int(section["num_windows"]), int(section["seed"]))


def _assert_windows_disjoint(train: torch.Tensor, heldout: torch.Tensor) -> None:
    if train.shape[1] != heldout.shape[1]:
        raise ValueError("calibration and heldout seq_len must match so token windows can be checked for overlap")
    train_rows = {row.contiguous().numpy().tobytes() for row in train}
    if any(row.contiguous().numpy().tobytes() in train_rows for row in heldout):
        raise ValueError("calibration and held-out token windows overlap; use disjoint windows/text")


@torch.no_grad()
def _capture_mlp_inputs(
    model: torch.nn.Module, module_path: str, windows: torch.Tensor, max_rows: int, seed: int
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError(f"module {module_path} did not receive a tensor input")
        value = args[0]
        captured.append(value.detach().reshape(-1, value.shape[-1]).to(device="cpu", dtype=torch.float32))

    module = model.get_submodule(module_path)
    handle = module.register_forward_pre_hook(hook)
    was_training = model.training
    model.eval()
    try:
        for window in windows:
            model(input_ids=window.unsqueeze(0).to(next(model.parameters()).device))
    finally:
        model.train(was_training)
        handle.remove()
    if not captured:
        raise RuntimeError(f"no teacher inputs were captured from {module_path}")
    rows = torch.cat(captured, dim=0).contiguous()
    if rows.shape[0] > max_rows:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(rows.shape[0], generator=generator)[:max_rows].sort().values
        rows = rows[indices].contiguous()
    return rows


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rows_digest(rows: torch.Tensor) -> str:
    return _tensor_digest(rows.to(torch.float32))


def _joint_specs(
    projections: dict[str, Projection],
    weights: dict[str, torch.Tensor],
    biases: dict[str, torch.Tensor | None],
    budget: dict,
    cascade: dict,
):
    ratio = resolve_byte_ratio(budget)
    storage_dtype = budget["storage_dtype"]
    bpp = bytes_per_param(storage_dtype)
    dense_bytes = {
        key: weights[key].numel() * weights[key].element_size()
        + (0 if biases[key] is None else biases[key].numel() * biases[key].element_size())
        for key in ("gate", "up", "down")
    }
    dense_total = sum(dense_bytes.values())
    byte_budget = math.floor(ratio * dense_total)
    topology = Topology.from_dict(cascade["topology"])
    specs: dict[str, CascadeSpec] = {}
    fixed_bias_bytes = sum(0 if biases[key] is None else biases[key].numel() * bpp for key in biases)
    for key in ("gate", "up", "down"):
        projection = projections[key]
        stages_value = cascade["stages_per_rotor"]
        stages = (
            auto_stages(topology.kind, max(projection.in_features, projection.out_features))
            if stages_value == "auto"
            else stages_value
        )
        specs[key] = CascadeSpec(
            in_features=projection.in_features,
            out_features=projection.out_features,
            num_rotors=1,
            stages_per_rotor=stages,
            topology=topology,
            adapter=cascade["adapter"],
            residual_rank=cascade["residual_rank"],
        )
    minimum_bytes = fixed_bias_bytes + sum(spec.param_count() * bpp for spec in specs.values())
    if minimum_bytes > byte_budget:
        raise ValueError(
            f"combined budget {byte_budget} bytes cannot fit one rotor for each projection plus stored biases "
            f"({minimum_bytes} bytes minimum; dense bytes={dense_total})"
        )
    remaining = byte_budget - minimum_bytes
    # Add one rotor at a time to the least-filled projection, measured against dense weight parameters.
    # This preserves a useful budget share for every projection while accounting for unequal widths exactly.
    active = {key for key in specs}
    while active:
        key = min(active, key=lambda name: (specs[name].param_count() / max(weights[name].numel(), 1), name))
        old = specs[key]
        extra = CascadeSpec(
            in_features=old.in_features,
            out_features=old.out_features,
            num_rotors=old.num_rotors + 1,
            stages_per_rotor=old.stages_per_rotor,
            topology=old.topology,
            adapter=old.adapter,
            residual_rank=old.residual_rank,
        )
        cost = (extra.param_count() - old.param_count()) * bpp
        if cost <= remaining:
            specs[key] = extra
            remaining -= cost
        else:
            active.remove(key)
    combined_bytes = fixed_bias_bytes + sum(spec.param_count() * bpp for spec in specs.values())
    if combined_bytes > byte_budget:
        raise RuntimeError("joint FFN byte allocator exceeded its combined budget")
    return specs, {
        "dense_bytes": dense_total,
        "byte_budget": byte_budget,
        "combined_minimum_bytes": minimum_bytes,
        "combined_structured_bytes": combined_bytes,
        "per_projection_dense_bytes": dense_bytes,
        "per_projection_structured_bytes": {
            key: specs[key].param_count() * bpp + (0 if biases[key] is None else biases[key].numel() * bpp)
            for key in specs
        },
        "target_ratio": ratio,
        "storage_dtype": storage_dtype,
    }


def _snapshot_owner(identity: dict) -> str:
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    marker = {
        "format": _SNAPSHOT_OWNER_FORMAT,
        "schema_version": _SNAPSHOT_SCHEMA,
        "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
    }
    return json.dumps(marker, sort_keys=True) + "\n"


def _publish_snapshot(storage: Storage, root: Path, state: dict, keep: int) -> Path:
    storage.assert_writable(root)
    if root.is_symlink():
        raise StorageError(f"refusing to use symlinked joint FFN snapshot root {root}")
    root.mkdir(parents=True, exist_ok=True)
    now_ns = time.time_ns()
    dest = root / f"joint-snapshot-{now_ns}-{uuid.uuid4().hex}"
    with atomic_dir(dest) as temporary:
        owner_marker = _snapshot_owner(state["identity"])
        (temporary / "owner.json").write_text(owner_marker, encoding="utf-8")
        with (temporary / "state.pt").open("wb") as stream:
            torch.save(cpu_copy(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
    owned = []
    for path in root.iterdir():
        if _SNAPSHOT_NAME.fullmatch(path.name) is None or path.is_symlink() or not path.is_dir():
            continue
        try:
            if (path / "owner.json").read_text(encoding="utf-8") == owner_marker and (path / "state.pt").is_file():
                owned.append(path)
        except OSError:
            continue
    owned.sort(key=lambda path: path.name)
    for old in owned[:-keep]:
        shutil.rmtree(old)
    return dest


def _load_snapshot(storage: Storage, path: Path, expected_root: Path, identity: dict) -> dict:
    if path.is_symlink() or not path.is_dir() or _SNAPSHOT_NAME.fullmatch(path.name) is None:
        raise StorageError(f"invalid joint FFN snapshot directory {path}")
    storage.assert_under_nas(path)
    try:
        if path.parent.resolve(strict=True) != expected_root.resolve(strict=False):
            raise StorageError(f"joint FFN snapshot must be inside {expected_root}")
    except OSError as exc:
        raise StorageError(f"cannot resolve joint FFN snapshot path {path}: {exc}") from exc
    if (path / "owner.json").is_symlink() or (path / "owner.json").read_text(encoding="utf-8") != _snapshot_owner(
        identity
    ):
        raise StorageError(f"joint FFN snapshot is not owned by this runner: {path}")
    state_path = path / "state.pt"
    if state_path.is_symlink() or not state_path.is_file():
        raise StorageError(f"joint FFN snapshot has no regular state.pt: {path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"format", "schema_version", "identity", "fit_state"}:
        raise ValueError("invalid joint FFN snapshot fields")
    if state["format"] != _SNAPSHOT_FORMAT or state["schema_version"] != _SNAPSHOT_SCHEMA:
        raise ValueError("unsupported joint FFN snapshot format or schema")
    if state["identity"] != identity:
        raise ValueError(
            "joint FFN snapshot identity differs from model, config, weights, or fixed training/held-out rows"
        )
    if not isinstance(state["fit_state"], dict):
        raise ValueError("joint FFN snapshot fit_state must be a mapping")
    return state["fit_state"]


def _provenance(base: Path, model_info: dict, config_json: dict) -> BaseProvenance:
    files = sorted(base.glob("*.safetensors")) + [base / "config.json"]
    text_config = qwen3_5.text_config(config_json)
    tied = config_json.get("tie_word_embeddings", text_config.get("tie_word_embeddings"))
    return BaseProvenance(
        hf_id=model_info.get("hf_id"),
        revision=model_info.get("revision"),
        path=os.path.abspath(base),
        file_sha256={file.name: sha256_file(file) for file in files},
        tied_word_embeddings=None if tied is None else bool(tied),
        architectures=list(config_json.get("architectures", [])),
    )


def _unique_run_id(run: ExperimentRun) -> None:
    from datetime import timedelta

    while (run.out_root / run.run_id).exists():
        run.created += timedelta(seconds=1)
        from rotor.reporting import make_run_id

        run.run_id = make_run_id(run.model, run.method, run.target, run.created)


def run_joint_ffn(
    config: dict,
    *,
    storage: Storage | None = None,
    config_path: str | Path | None = None,
    device: str | None = None,
    steps_scale: float | None = None,
    resume_from: Path | None = None,
    snapshot_interval_seconds: float = 1800,
    snapshot_keep: int = 3,
    report_root: Path | None = None,
    log=print,
) -> Path:
    """Capture fixed teacher rows, jointly fit one FFN, and atomically publish its standard rotor-scaler checkpoint."""
    config = copy.deepcopy(config)
    if device is not None:
        config["fit"]["device"] = device
    if steps_scale is not None:
        if not math.isfinite(steps_scale) or steps_scale <= 0:
            raise ValueError("steps_scale must be finite and > 0")
        schedule_config = config["fit"].setdefault("schedule", {})
        for phase in schedule_config.get("phases", []):
            phase["steps"] = max(1, int(round(int(phase["steps"]) * steps_scale)))
    if not math.isfinite(snapshot_interval_seconds) or snapshot_interval_seconds <= 0:
        raise ValueError("snapshot_interval_seconds must be finite and > 0")
    if isinstance(snapshot_keep, bool) or not isinstance(snapshot_keep, int) or snapshot_keep < 1:
        raise ValueError("snapshot_keep must be an integer >= 1")
    schedule = FitSchedule.from_dict(config["fit"].get("schedule") or {})
    storage = storage or load_storage()
    canonical_model, model_info = model_entry(config["model"])
    base = storage.base_path(model_info["local_path"])
    if not base.is_dir():
        raise FileNotFoundError(
            f"base checkpoint for {canonical_model} is not present at {base}; "
            "place the pinned model there first (no downloads)"
        )
    config_json = json.loads((base / "config.json").read_text(encoding="utf-8"))
    infos_list = local_tensors(base)
    infos = {info.name: info for info in infos_list}
    all_projections = qwen3_5.discover_projections(infos_list, config_json)
    projections = [p for p in all_projections if p.layer == config["layer"] and p.family in _FAMILIES]
    if len(projections) != 3 or {p.family for p in projections} != set(_FAMILIES):
        raise ValueError(f"base checkpoint layer {config['layer']} does not contain exactly gate/up/down FFN weights")
    weights, biases, by_key = _load_joint_weights(base, projections, infos)
    source_dtypes = {weights[key].dtype for key in ("gate", "up", "down")}
    if len(source_dtypes) != 1:
        raise ValueError("the gate, up, and down source weights must share one dtype for teacher activation capture")
    source_dtype = next(iter(source_dtypes))
    specs, budget_plan = _joint_specs(by_key, weights, biases, config["budget"], config["cascade"])
    storage_dtype = config["budget"]["storage_dtype"]
    module_names = {key: f"model.layers.{config['layer']}.mlp.{_MODULE_SUFFIX[key]}" for key in _FAMILIES.values()}
    target = target_name(budget_plan["target_ratio"])
    dest = storage.derived_path(config["name"])
    storage.assert_writable(dest)
    if dest.exists():
        raise StorageError(f"refusing to overwrite existing joint FFN output {dest}")
    snapshot_root = dest.with_name(f"{dest.name}.snapshots")
    storage.assert_writable(snapshot_root)

    resolved_config = copy.deepcopy(config)
    resolved_config["model"] = canonical_model
    resolved_config["base_dir"] = str(base.resolve())
    resolved_config["output_dir"] = str(dest)
    resolved_config["budget_plan"] = budget_plan
    resolved_config["modules"] = module_names
    resolved_config["config_path"] = None if config_path is None else str(Path(config_path).resolve())
    identity_config = copy.deepcopy(config)
    identity_config["fit"]["device"] = config["fit"]["device"]
    identity_config["fit"]["schedule"] = copy.deepcopy(config["fit"].get("schedule") or {})
    identity_config["fit"]["steps_scale"] = steps_scale
    identity_config["config_path"] = None
    resolved_config["fit"]["schedule"] = copy.deepcopy(config["fit"].get("schedule") or {})
    run = ExperimentRun(
        "fit",
        model=canonical_model,
        method=f"rotor-joint-ffn-{config['cascade']['topology']['kind']}",
        target=target,
        config=resolved_config,
        title=f"Joint FFN output fit: {canonical_model} layer {config['layer']}",
        out_root=report_root,
    )
    _unique_run_id(run)

    def emit(message: str) -> None:
        run.log(message)

    def execute() -> None:
        with ResourceMonitor() as monitor:
            tokenizer = qwen3_5.load_tokenizer(base)
            train_windows = _read_windows(tokenizer, config["calibration"])
            heldout_windows = _read_windows(tokenizer, config["heldout"])
            _assert_windows_disjoint(train_windows, heldout_windows)
            mlp_path = f"model.layers.{config['layer']}.mlp"
            model = qwen3_5.load_text_model(base, dtype=source_dtype, device=config["fit"]["device"])
            try:
                train_rows = _capture_mlp_inputs(
                    model, mlp_path, train_windows, config["calibration"]["max_rows"], config["calibration"]["seed"]
                )
                heldout_rows = _capture_mlp_inputs(
                    model, mlp_path, heldout_windows, config["heldout"]["max_rows"], config["heldout"]["seed"]
                )
            finally:
                del model
            train_window_sha = windows_fingerprint(train_windows)
            heldout_window_sha = windows_fingerprint(heldout_windows)
            train_rows_sha = _rows_digest(train_rows)
            heldout_rows_sha = _rows_digest(heldout_rows)
            provenance = _provenance(base, model_info, config_json)
            identity = {
                "config": identity_config,
                "base": asdict(provenance),
                "projections": {key: asdict(by_key[key]) for key in ("gate", "up", "down")},
                "specs": {key: specs[key].to_dict() for key in specs},
                "source_tensors": {
                    key: {
                        "weight": _tensor_digest(weights[key]),
                        "bias": None if biases[key] is None else _tensor_digest(biases[key]),
                    }
                    for key in weights
                },
                "train_windows_sha256": train_window_sha,
                "heldout_windows_sha256": heldout_window_sha,
                "train_rows_sha256": train_rows_sha,
                "heldout_rows_sha256": heldout_rows_sha,
            }
            active_resume = None
            if resume_from is not None:
                active_resume = _load_snapshot(storage, Path(resume_from), snapshot_root, identity)
            fit_start = time.monotonic()
            last_snapshot = time.monotonic()

            def on_progress(progress: dict) -> None:
                nonlocal last_snapshot
                now = time.monotonic()
                if now - last_snapshot < snapshot_interval_seconds:
                    return
                state = {
                    "format": _SNAPSHOT_FORMAT,
                    "schema_version": _SNAPSHOT_SCHEMA,
                    "identity": identity,
                    "fit_state": progress,
                }
                saved = _publish_snapshot(storage, snapshot_root, state, snapshot_keep)
                last_snapshot = now
                emit(f"published resumable joint FFN snapshot {saved}")

            modules, fit_metrics = fit_joint_ffn(
                train_rows,
                weights,
                specs,
                schedule,
                biases=biases,
                heldout_X=heldout_rows,
                seed=int(config["fit"]["seed"]),
                init_angle_std=float(config["fit"]["init_angle_std"]),
                device=config["fit"]["device"],
                log=emit,
                resume_state=active_resume,
                progress_callback=on_progress,
                storage_dtype=storage_dtype,
                combined_byte_budget=budget_plan["byte_budget"],
            )
            fit_seconds = time.monotonic() - fit_start
            layers = {module_names[key]: modules[key].to(device="cpu", dtype=torch.float32) for key in modules}
            entries: list[LayerEntry] = []
            for key in ("gate", "up", "down"):
                projection = by_key[key]
                module = module_names[key]
                serialized_bytes, scalars = layer_accounting(module, layers[module], storage_dtype)
                if serialized_bytes != fit_metrics["per_projection_bytes"][key]:
                    raise RuntimeError(
                        f"{module}: serialized payload {serialized_bytes} differs from fit accounting "
                        f"{fit_metrics['per_projection_bytes'][key]}"
                    )
                metric = {
                    "objective": "joint_ffn_output",
                    "train_output_relative_error": fit_metrics["train_output_relative_error"],
                    "heldout_output_relative_error": fit_metrics["heldout_output_relative_error"],
                    "joint_combined_bytes": fit_metrics["combined_bytes"],
                    "joint_combined_byte_budget": fit_metrics["combined_byte_budget"],
                    "projection_key": key,
                    "fit_wall_seconds_total": fit_seconds,
                }
                entry = LayerEntry(
                    module=module,
                    source_tensor=projection.tensor,
                    family=projection.family,
                    spec=specs[key],
                    storage_dtype=storage_dtype,
                    has_bias=layers[module].bias is not None,
                    dense_params=weights[key].numel(),
                    structured_params=specs[key].param_count(),
                    dense_bytes=weights[key].numel() * weights[key].element_size(),
                    structured_bytes=serialized_bytes,
                    metrics=metric,
                    learned_scalars=scalars,
                    dense_dtype=projection.dtype,
                )
                entries.append(entry)
            if sum(entry.structured_bytes for entry in entries) != fit_metrics["combined_bytes"]:
                raise RuntimeError("serialized three-projection payload does not match combined fit byte accounting")
            conversion_config = {
                **resolved_config,
                "windows_fingerprint": {"calibration": train_window_sha, "heldout": heldout_window_sha},
                "input_rows_sha256": {"calibration": train_rows_sha, "heldout": heldout_rows_sha},
                "fit_metrics": fit_metrics,
            }
            manifest_notes = [
                "Joint FFN budget denominator includes dense weight and present bias bytes "
                "for gate, up, and down.",
                "Manifest per-layer dense_bytes keep the standard weight-only convention; "
                "see fit_metrics.combined_dense_bytes.",
            ]
            manifest = Manifest(
                base=provenance,
                layers=entries,
                conversion_config=conversion_config,
                notes=[
                    "Joint Phase 4B fit of gate, up, and down projections against normalized "
                    "composed SwiGLU output error.",
                    "Calibration and held-out hidden-state rows were captured from "
                    "disjoint fixed real-text windows.",
                    "FP32 master parameters were exported in the configured storage dtype.",
                    *manifest_notes,
                ],
            )
            with atomic_dir(dest) as temporary:
                write_structured(temporary, manifest, layers, base)
        def finalize():
            monitor_metrics = monitor.as_dict()
            run.metrics.update(
                {
                    "fit": fit_metrics,
                    "budget_plan": budget_plan,
                    "inputs": {
                        "calibration_path": config["calibration"]["path"],
                        "heldout_path": config["heldout"]["path"],
                        "calibration_windows": list(train_windows.shape),
                        "heldout_windows": list(heldout_windows.shape),
                        "calibration_windows_sha256": train_window_sha,
                        "heldout_windows_sha256": heldout_window_sha,
                        "calibration_rows": int(train_rows.shape[0]),
                        "heldout_rows": int(heldout_rows.shape[0]),
                        "calibration_rows_sha256": train_rows_sha,
                        "heldout_rows_sha256": heldout_rows_sha,
                        "selection": "seeded randperm sorted by original row index; fixed once before fitting",
                    },
                    "storage": {
                        "output_dir": str(dest),
                        "serialized": manifest.serialized,
                        "manifest_totals": manifest.totals(),
                    },
                    "resources": monitor_metrics,
                    "fit_wall_seconds": fit_seconds,
                }
            )
            run.summary = (
                f"Jointly fitted gate/up/down FFN projections for `{canonical_model}` layer {config['layer']}; "
                f"held-out normalized output error {fit_metrics['heldout_output_relative_error']:.6g}; "
                f"serialized payload {fit_metrics['combined_bytes']} / "
                f"{budget_plan['byte_budget']} B. "
                f"Output: `{dest}`."
            )
            run.interpretation = (
                "This is a fixed-row teacher FFN output fitting result, not an end-to-end language-model quality "
                "evaluation. Calibration rows were used for optimization; disjoint held-out rows were used only for "
                "selection/reporting. The output checkpoint stores three projections only."
            )
            run.add_section(
                "Joint FFN output metrics",
                markdown_table(
                    [
                        {"set": "calibration", "normalized_squared_error": fit_metrics["train_output_relative_error"]},
                        {"set": "held-out", "normalized_squared_error": fit_metrics["heldout_output_relative_error"]},
                    ],
                    ["set", "normalized_squared_error"],
                ),
            )
            run.add_section(
                "Combined byte budget",
                markdown_table(
                    [
                        {
                            "projection": key,
                            "dense_bytes": budget_plan["per_projection_dense_bytes"][key],
                            "structured_payload_bytes": fit_metrics["per_projection_bytes"][key],
                            "num_rotors": specs[key].num_rotors,
                        }
                        for key in ("gate", "up", "down")
                    ]
                    + [
                        {
                            "projection": "combined",
                            "dense_bytes": fit_metrics["combined_dense_bytes"],
                            "structured_payload_bytes": fit_metrics["combined_bytes"],
                            "num_rotors": "-",
                        }
                    ],
                    ["projection", "dense_bytes", "structured_payload_bytes", "num_rotors"],
                ),
            )
            emit(f"wrote joint FFN checkpoint {dest}")
        finalize()

    with run:
        execute()
    return dest
