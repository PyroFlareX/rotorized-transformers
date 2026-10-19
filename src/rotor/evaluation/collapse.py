"""Bounded reference-backend diagnostics for stored rotor projections and replacement ablations."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rotor.converters.snapshots import load_snapshot
from rotor.data import tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.evaluation.generation import greedy_decode
from rotor.evaluation.lm import evaluate_lm
from rotor.evaluation.prompts import load_prompts
from rotor.formats.checkpoint import load_structured_layers, read_manifest
from rotor.formats.manifest import LayerEntry
from rotor.math.cascade import CascadeSpec, RotorCascade, RotorLinear
from rotor.models.loading import load_base
from rotor.models.qwen3_5 import load_tokenizer
from rotor.models.replace import replace_module
from rotor.reporting import ExperimentRun
from rotor.storage import load_storage, model_entry, sha256_file


def _serialized_parameter_report(ckpt: Path, module: str, layer: RotorLinear) -> dict[str, Any]:
    """Check safetensors finiteness and exact loaded-parameter equality after storage casting."""
    from safetensors import safe_open

    expected = {key: value.detach().cpu() for key, value in layer.state_dict().items()}
    rows = {}
    with safe_open(str(ckpt / "rotor_weights.safetensors"), framework="pt", device="cpu") as file:
        for state_name, value in expected.items():
            tensor_name = state_name.removeprefix("cascade.")
            key = f"{module}.{tensor_name}"
            stored = file.get_tensor(key)
            rows[state_name] = {
                "tensor_name": key,
                "dtype": str(stored.dtype),
                "elements": stored.numel(),
                "finite_values": int(torch.isfinite(stored).sum()),
                "loaded_equals_serialized_cast": bool(torch.equal(value.to(stored.dtype), stored)),
            }
    return {
        "all_finite": all(row["finite_values"] == row["elements"] for row in rows.values()),
        "all_loaded_equal_serialized_cast": all(row["loaded_equals_serialized_cast"] for row in rows.values()),
        "parameters": rows,
    }


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _visible_whitespace(text: str) -> str:
    visible = {" ": "[SP]", "\n": "[NL]\n", "\r": "[CR]", "\t": "[TAB]"}
    return "".join(visible.get(char, f"[U+{ord(char):04X}]" if char.isspace() else char) for char in text)


def _generation_metrics(token_ids: list[int], text: str, tokenizer) -> dict[str, Any]:
    count = len(token_ids)
    specials = set(getattr(tokenizer, "all_special_ids", []))
    return {
        "tokens": count,
        "special_token_fraction": sum(token_id in specials for token_id in token_ids) / count if count else 0.0,
        "repeated_token_fraction": (count - len(set(token_ids))) / count if count else 0.0,
        "whitespace_character_fraction": sum(char.isspace() for char in text) / len(text) if text else 0.0,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None}
    q = torch.quantile(finite, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=finite.dtype))
    return {key: float(value) for key, value in zip(("min", "p50", "p90", "p99", "max"), q, strict=True)}


def _tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Scalar error, finiteness, RMS, cosine, and compact row/channel distributions."""
    if reference.shape != actual.shape:
        raise ValueError(f"metric shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    ref2 = reference.detach().to(device="cpu", dtype=torch.float64)
    got2 = actual.detach().to(device="cpu", dtype=torch.float64)
    ref, got = ref2.reshape(-1), got2.reshape(-1)
    finite = torch.isfinite(ref) & torch.isfinite(got)
    result: dict[str, Any] = {
        "elements": ref.numel(),
        "finite_reference": int(torch.isfinite(ref).sum()),
        "finite_actual": int(torch.isfinite(got).sum()),
        "finite_pairs": int(finite.sum()),
    }
    if not finite.any():
        result.update({"max_abs_error": None, "relative_l2_error": None, "cosine_similarity": None})
        return result
    ref, got = ref[finite], got[finite]
    delta = got - ref
    nonzero = ref != 0
    ref_norm = ref.norm()
    result.update(
        {
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "max_relative_error_nonzero_reference": float((delta[nonzero].abs() / ref[nonzero].abs()).max())
            if nonzero.any()
            else None,
            "relative_l2_error": float(delta.norm() / ref_norm) if ref_norm else None,
            "cosine_similarity": float(
                F.cosine_similarity(ref[None], got[None], dim=1, eps=torch.finfo(torch.float64).tiny)[0]
            ),
            "reference_rms": float(ref.square().mean().sqrt()),
            "actual_rms": float(got.square().mean().sqrt()),
        }
    )
    if ref2.ndim >= 2:
        # These compact channel profiles avoid emitting every model-width value.
        ref_matrix = ref2.reshape(-1, ref2.shape[-1])
        got_matrix = got2.reshape(-1, got2.shape[-1])
        result.update(
            {
                "row_rms_reference": _quantiles(ref_matrix.square().mean(dim=1).sqrt()),
                "row_rms_actual": _quantiles(got_matrix.square().mean(dim=1).sqrt()),
                "channel_rms_reference": _quantiles(ref_matrix.square().mean(dim=0).sqrt()),
                "channel_rms_actual": _quantiles(got_matrix.square().mean(dim=0).sqrt()),
                "nonfinite_reference": int((~torch.isfinite(ref_matrix)).sum()),
                "nonfinite_actual": int((~torch.isfinite(got_matrix)).sum()),
            }
        )
    return result


def _independent_dense_columns(cascade: RotorCascade, start: int, stop: int) -> torch.Tensor:
    spec = cascade.spec
    if stop <= start or stop - start > spec.in_features:
        raise ValueError("dense oracle chunk must contain between 1 and in_features columns")
    device, dtype = cascade.angles.device, cascade.angles.dtype
    width = spec.width
    indices = torch.arange(start, stop, device=device)
    basis = torch.zeros((stop - start, spec.in_features), dtype=dtype, device=device)
    basis[torch.arange(stop - start, device=device), indices] = 1
    hidden = basis * cascade.d_in
    if width > spec.in_features:
        hidden = F.pad(hidden, (0, width - spec.in_features))
    stage_pairs = [spec.topology.pairs(width, stage) for stage in range(spec.stages_per_rotor)]
    stage_offsets: list[int] = []
    offset = 0
    for pairs in stage_pairs:
        stage_offsets.append(offset)
        offset += len(pairs)
    for rotor in range(spec.num_rotors - 1, -1, -1):
        for stage_index in range(len(stage_pairs) - 1, -1, -1):
            pairs = stage_pairs[stage_index]
            if not pairs:
                continue
            pair_index = torch.tensor(pairs, dtype=torch.long, device=device)
            a, b = pair_index[:, 0], pair_index[:, 1]
            angle_start = stage_offsets[stage_index]
            theta = cascade.angles[rotor, angle_start : angle_start + len(pairs)]
            cosine, sine = torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)
            xa, xb = hidden.index_select(-1, a), hidden.index_select(-1, b)
            changed = hidden.clone()
            changed.index_copy_(-1, a, cosine * xa - sine * xb)
            changed.index_copy_(-1, b, sine * xa + cosine * xb)
            hidden = changed
        if rotor:
            hidden = hidden * cascade.d_mid[rotor - 1]
    columns = hidden[..., : spec.out_features] * cascade.d_out
    if cascade.residual is not None:
        columns = columns + (basis @ cascade.residual.V.T) @ cascade.residual.U.T
    return columns


def bounded_dense_action_parity(
    layer: RotorLinear,
    probes: torch.Tensor,
    *,
    basis_chunk: int = 64,
    dense_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare reference forward to independently streamed dense-column actions; never constructs identity/full W."""
    if probes.ndim != 2 or probes.shape[1] != layer.in_features:
        raise ValueError(f"probes must have shape (rows, {layer.in_features}), got {tuple(probes.shape)}")
    if basis_chunk < 1:
        raise ValueError("basis_chunk must be >= 1")
    cascade = layer.cascade
    device, dtype = cascade.angles.device, cascade.angles.dtype
    x = probes.to(device=device, dtype=dtype)
    dense_action = torch.zeros((x.shape[0], layer.out_features), dtype=torch.float64, device=device)
    weight_error_sq = torch.zeros((), dtype=torch.float64, device=device)
    weight_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    weight_max_abs = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, layer.in_features, basis_chunk):
            stop = min(start + basis_chunk, layer.in_features)
            columns = _independent_dense_columns(cascade, start, stop).to(torch.float64)
            dense_action.add_(x[:, start:stop].to(torch.float64) @ columns)
            if dense_weight is not None:
                expected = dense_weight[:, start:stop].to(device=device, dtype=torch.float64).T
                delta = columns - expected
                weight_error_sq += delta.square().sum()
                weight_norm_sq += expected.square().sum()
                weight_max_abs = torch.maximum(weight_max_abs, delta.abs().max())
        if layer.bias is not None:
            dense_action.add_(layer.bias.to(device=device, dtype=torch.float64))
        direct = layer(x).to(torch.float64)
    metrics = _tensor_metrics(direct, dense_action)
    metrics["dense_basis_chunk"] = basis_chunk
    metrics["dense_max_temporary_basis_elements"] = min(basis_chunk, layer.in_features) * layer.in_features
    if dense_weight is not None:
        metrics["weight_relative_frobenius_error"] = (
            float(weight_error_sq.sqrt() / weight_norm_sq.sqrt()) if weight_norm_sq else None
        )
        metrics["weight_max_abs_error"] = float(weight_max_abs)
        metrics["weight_reference_frobenius"] = float(weight_norm_sq.sqrt())
    return metrics


def _capture_teacher(model, paths: list[str], input_ids: torch.Tensor, device: torch.device):
    """One teacher pass captures projection inputs/outputs, FFN outputs, block outputs, and logits."""
    inputs: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
    outputs: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
    blocks = sorted({f"model.layers.{path.split('.')[2]}" for path in paths})
    block_outputs: dict[str, list[torch.Tensor]] = {path: [] for path in blocks}
    ffns = sorted({f"model.layers.{path.split('.')[2]}.mlp" for path in paths})
    ffn_outputs: dict[str, list[torch.Tensor]] = {path: [] for path in ffns}
    handles = []
    for path in paths:

        def pre_hook(_module, args, target=path):
            inputs[target].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu().float())

        def post_hook(_module, _args, result, target=path):
            value = result[0] if isinstance(result, tuple) else result
            outputs[target].append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

        module = model.get_submodule(path)
        handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))
    for path in blocks:

        def block_hook(_module, _args, result, target=path):
            value = result[0] if isinstance(result, tuple) else result
            block_outputs[target].append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

        handles.append(model.get_submodule(path).register_forward_hook(block_hook))
    for path in ffns:

        def ffn_hook(_module, _args, result, target=path):
            value = result[0] if isinstance(result, tuple) else result
            ffn_outputs[target].append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

        handles.append(model.get_submodule(path).register_forward_hook(ffn_hook))
    try:
        with torch.no_grad():
            result = model(input_ids=input_ids.to(device))
    finally:
        for handle in handles:
            handle.remove()
    if any(not inputs[path] or not outputs[path] for path in paths):
        raise RuntimeError("one or more stored projection modules were not executed by the prompt")
    return (
        {path: torch.cat(values) for path, values in inputs.items()},
        {path: torch.cat(values) for path, values in outputs.items()},
        {path: torch.cat(values) for path, values in ffn_outputs.items()},
        {path: torch.cat(values) for path, values in block_outputs.items()},
        result.logits.detach().cpu().float(),
    )


def _load_master_layers(snapshot_path: Path, manifest, storage) -> tuple[dict[str, RotorLinear], dict[str, Any]]:
    path = snapshot_path.expanduser().resolve()
    storage.assert_under_nas(path)
    state = load_snapshot(path)
    identity = state["identity"]
    base_identity = identity.get("base", {})
    for key in ("path", "hf_id", "revision", "file_sha256"):
        if base_identity.get(key) != getattr(manifest.base, key):
            raise ValueError(f"FP32 snapshot base identity differs from checkpoint manifest at {key}")
    identity_specs = identity.get("specs")
    if not isinstance(identity_specs, dict):
        raise ValueError("FP32 snapshot lacks its conversion CascadeSpec identities")
    entries = {entry.module: entry for entry in manifest.layers}
    masters: dict[str, RotorLinear] = {}
    for module, record in state["layers"].items():
        if module not in entries:
            continue
        entry = LayerEntry.from_dict(record["entry"])
        stored = entries[module]
        if entry.module != module or entry.source_tensor != stored.source_tensor or entry.spec != stored.spec:
            raise ValueError(f"snapshot projection identity differs from checkpoint for {module}")
        if identity_specs.get(module) != stored.spec.to_dict():
            raise ValueError(f"snapshot identity CascadeSpec differs from checkpoint for {module}")
        state_dict = record["state_dict"]
        layer = RotorLinear(RotorCascade(entry.spec, dtype=torch.float32), state_dict.get("bias"))
        layer.load_state_dict(state_dict, strict=True)
        if any(parameter.dtype != torch.float32 for parameter in layer.parameters()):
            raise ValueError(f"snapshot projection {module} is not an FP32 master")
        masters[module] = layer
    return masters, {
        "path": str(path),
        "identity_sha256": _sha256_json(identity),
        "completed_matching_projections": sorted(masters),
    }


def _cast_layer(layer: RotorLinear, dtype: torch.dtype) -> RotorLinear:
    """Round stored parameters to dtype, then evaluate them with FP32 arithmetic."""
    bias = layer.bias.detach().cpu().to(dtype).float() if layer.bias is not None else None
    cast = RotorLinear(RotorCascade(layer.cascade.spec, dtype=torch.float32), bias)
    cast.load_state_dict(
        {key: value.detach().cpu().to(dtype).float() for key, value in layer.state_dict().items()}, strict=True
    )
    return cast


def _kl_metrics(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> dict[str, Any]:
    teacher = teacher_logits.detach().cpu().float()
    student = student_logits.detach().cpu().float()
    if teacher.shape != student.shape:
        raise ValueError(f"teacher/student logits differ in shape: {teacher.shape} != {student.shape}")
    probs = teacher.softmax(dim=-1)
    row_kl = (probs * (teacher.log_softmax(dim=-1) - student.log_softmax(dim=-1))).sum(dim=-1)
    return {
        "teacher_to_student_kl_last_token": float(row_kl[:, -1].mean()),
        "teacher_to_student_kl_all_positions_mean": float(row_kl.mean()),
        "top_token_agreement": float((teacher.argmax(dim=-1) == student.argmax(dim=-1)).float().mean()),
        "teacher_top_token": int(teacher[:, -1].argmax(dim=-1)[0]),
        "student_top_token": int(student[:, -1].argmax(dim=-1)[0]),
        "finite_teacher_logits": int(torch.isfinite(teacher).sum()),
        "finite_student_logits": int(torch.isfinite(student).sum()),
    }


@contextmanager
def _replacement_arm(model, replacements: dict[str, RotorLinear]):
    originals = {}
    try:
        for path, layer in replacements.items():
            originals[path] = replace_module(model, path, layer)
        yield
    finally:
        for path, original in originals.items():
            replace_module(model, path, original)


def _capture_intervention(model, input_ids: torch.Tensor, paths: list[str], device: torch.device):
    block_output: list[torch.Tensor] = []
    block_path = f"model.layers.{paths[0].split('.')[2]}"
    projection_outputs: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
    mlp_path = f"model.layers.{paths[0].split('.')[2]}.mlp"
    ffn_output: list[torch.Tensor] = []
    handles = []
    for path in paths:

        def projection_hook(_module, _args, result, target=path):
            value = result[0] if isinstance(result, tuple) else result
            projection_outputs[target].append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

        handles.append(model.get_submodule(path).register_forward_hook(projection_hook))

    def ffn_hook(_module, _args, result):
        value = result[0] if isinstance(result, tuple) else result
        ffn_output.append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

    handles.append(model.get_submodule(mlp_path).register_forward_hook(ffn_hook))

    def block_hook(_module, _args, result):
        value = result[0] if isinstance(result, tuple) else result
        block_output.append(value.detach().reshape(-1, value.shape[-1]).cpu().float())

    handles.append(model.get_submodule(block_path).register_forward_hook(block_hook))
    try:
        with torch.no_grad():
            result = model(input_ids=input_ids.to(device))
    finally:
        for handle in handles:
            handle.remove()
    return (
        {path: torch.cat(values) for path, values in projection_outputs.items()},
        ffn_output[0],
        block_output[0],
        result,
    )


def _resolve_paths(args) -> tuple[Path, Path, str]:
    storage = load_storage()
    base_dir = storage.base_path(args.base_dir.expanduser().resolve()) if args.base_dir else None
    model_label = args.model or "local-model"
    if args.model:
        model_label, catalog_entry = model_entry(args.model)
        base_dir = base_dir or storage.base_path(catalog_entry["local_path"])
    ckpt = storage.assert_under_nas(args.ckpt.expanduser().resolve())
    manifest = read_manifest(ckpt)
    base_dir = base_dir or storage.assert_under_nas(Path(manifest.base.path).expanduser().resolve())
    if not base_dir.is_dir():
        raise FileNotFoundError(f"local base checkpoint missing: {base_dir}; downloads are never attempted")
    if model_label == "local-model":
        model_label = manifest.base.hf_id.rsplit("/", 1)[-1] if manifest.base.hf_id else base_dir.name
    return base_dir, ckpt, model_label


def _append_jsonl(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _run_smoke(run: ExperimentRun, seed: int) -> None:
    """Known-correct tiny rectangular cascade with residual and bias; CPU only."""
    spec = CascadeSpec(7, 5, 3, 4, residual_rank=2)
    cascade = RotorCascade(spec, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        cascade.angles.normal_(0, 0.2, generator=gen)
        cascade.d_in.normal_(1, 0.1, generator=gen)
        cascade.d_mid.normal_(1, 0.1, generator=gen)
        cascade.d_out.normal_(1, 0.1, generator=gen)
        cascade.residual.U.normal_(0, 0.1, generator=gen)
        cascade.residual.V.normal_(0, 0.1, generator=gen)
    layer = RotorLinear(cascade, torch.arange(5, dtype=torch.float64) / 7)
    probes = torch.randn(9, 7, generator=gen, dtype=torch.float64)
    parity = bounded_dense_action_parity(layer, probes, basis_chunk=3)
    parity["harness"] = "known-correct-rectangular-cascade-residual-bias"
    parity["tolerance"] = {
        "rtol": 1e-10,
        "atol": 1e-10,
        "source": "tests/test_math.py forward-vs-dense cascade contract",
    }
    if parity["max_abs_error"] > 1e-10:
        raise AssertionError(f"tiny CPU parity failed max_abs_error={parity['max_abs_error']}")
    run.metrics["smoke"] = parity
    run.summary = "Tiny CPU-only, known-correct rectangular rotor dense-action parity harness passed."
    run.interpretation = "Synthetic harness validation only; it does not measure model or language quality."


def run_collapse_diagnostics(args) -> Path:
    """Run stored projection parity, optional FP32/storage comparisons, and prompt replacement ablations."""
    label = "tiny-cascade-smoke" if args.smoke else (args.model or args.ckpt.name)
    run = ExperimentRun(
        "collapse-diagnosis",
        model=label,
        method="reference-rotor-diagnostics",
        target="smoke" if args.smoke else "stored-checkpoint",
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        title=f"Collapse diagnosis: {label}",
    )
    if args.smoke:
        with run:
            _run_smoke(run, args.seed)
        return run.path

    with run:
        base_dir, ckpt, label = _resolve_paths(args)
        storage = load_storage()
        manifest, stored_layers = load_structured_layers(ckpt, dtype=torch.float32, device="cpu")
        entries = {entry.module: entry for entry in manifest.layers}
        if not stored_layers or set(stored_layers) != set(entries):
            raise ValueError("checkpoint manifest and loaded structured projection sets differ or are empty")
        device = torch.device(args.device)
        model = load_base(base_dir, dtype=DTYPES[args.dtype], device=device).eval()
        tokenizer = load_tokenizer(base_dir)
        prompt_path = args.prompts.expanduser().resolve()
        prompts = load_prompts(prompt_path)[: args.prompt_limit]
        if not prompts:
            raise ValueError("--prompt-limit selected no prompts")
        prompt_ids = []
        for row in prompts:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            if not ids:
                raise ValueError(f"prompt {row['id']} tokenizes to no tokens")
            prompt_ids.append(torch.tensor([ids], dtype=torch.long))
        paths = sorted(stored_layers)
        prompt_inputs: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
        prompt_outputs: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
        ffn_outputs: dict[str, list[torch.Tensor]] = {}
        teacher_logits: dict[str, torch.Tensor] = {}
        for path in paths:
            ffn_outputs.setdefault(f"model.layers.{path.split('.')[2]}.mlp", [])
        block_outputs: dict[str, list[torch.Tensor]] = {}
        prompt_block_outputs: dict[str, dict[str, torch.Tensor]] = {}
        prompt_ffn_outputs: dict[str, dict[str, torch.Tensor]] = {}
        prompt_outputs_by_prompt: dict[str, dict[str, torch.Tensor]] = {}
        for prompt, ids in zip(prompts, prompt_ids, strict=True):
            captured_inputs, captured_outputs, captured_ffns, captured_blocks, logits = _capture_teacher(
                model, paths, ids, device
            )
            for path in paths:
                prompt_inputs[path].append(captured_inputs[path])
                prompt_outputs[path].append(captured_outputs[path])
            prompt_outputs_by_prompt[prompt["id"]] = captured_outputs
            for path, value in captured_ffns.items():
                ffn_outputs[path].append(value)
            for path, value in captured_blocks.items():
                block_outputs.setdefault(path, []).append(value)
            prompt_block_outputs[prompt["id"]] = captured_blocks
            prompt_ffn_outputs[prompt["id"]] = captured_ffns
            teacher_logits[prompt["id"]] = logits
        prompt_inputs = {path: torch.cat(rows) for path, rows in prompt_inputs.items()}
        prompt_outputs = {path: torch.cat(rows) for path, rows in prompt_outputs.items()}
        ffn_outputs = {path: torch.cat(rows) for path, rows in ffn_outputs.items()}

        masters, snapshot_info = ({}, None)
        if args.snapshot:
            masters, snapshot_info = _load_master_layers(args.snapshot, manifest, storage)
        prompt_fingerprints = {
            row["id"]: hashlib.sha256(json.dumps(ids[0].tolist(), separators=(",", ":")).encode()).hexdigest()
            for row, ids in zip(prompts, prompt_ids, strict=True)
        }
        run.metrics["provenance"] = {
            "checkpoint": str(ckpt),
            "checkpoint_manifest_sha256": sha256_file(ckpt / "rotor_manifest.json"),
            "base_model_path": str(base_dir),
            "base_config_sha256": sha256_file(base_dir / "config.json"),
            "base_hf_id": manifest.base.hf_id,
            "base_revision": manifest.base.revision,
            "tokenizer_sha256": sha256_file(base_dir / "tokenizer.json")
            if (base_dir / "tokenizer.json").is_file()
            else None,
            "prompts_path": str(prompt_path),
            "prompts_sha256": sha256_file(prompt_path),
            "prompt_token_fingerprints": prompt_fingerprints,
            "seed": args.seed,
            "device": str(device),
            "activation_dtype": args.dtype,
            "reference_backend": "RotorLinear",
            "snapshot": snapshot_info,
        }
        run.metrics["corpus_status"] = (
            (
                f"user-supplied held-out text: {args.held_out_text}; "
                "approval is caller-provided and not independently verified"
            )
            if args.held_out_text
            else "no approved held-out corpus configured; diagnostic prompts are not held-out benchmark data"
        )
        run.metrics["stored_projection_count"] = len(stored_layers)
        run.metrics["fp32_master_projection_count"] = len(masters)
        run.metrics["snapshot"] = snapshot_info or {"status": "not supplied; FP32 master comparisons unavailable"}
        run.metrics["heldout_nll"] = {"status": "not measured: no approved corpus supplied"}
        run.log(f"loaded {len(stored_layers)} stored projections, reference backend, device={device}")
        run.log(f"diagnostic prompt SHA-256 {run.metrics['provenance']['prompts_sha256']}; {len(prompts)} prompt(s)")
        if not args.held_out_text:
            run.log("no approved corpus configured; fixed diagnostic prompts are not used as held-out NLL")

        records = (
            args.output.expanduser().resolve()
            if args.output
            else run.out_root.parent / "prompt-evaluations" / f"{run.run_id}.jsonl"
        )
        records.parent.mkdir(parents=True, exist_ok=True)
        if records.exists():
            raise FileExistsError(f"diagnostic JSONL already exists: {records}")
        run.add_log_reference(str(records))
        generator = torch.Generator().manual_seed(args.seed)
        run.metrics["operator_parity"] = {}
        heldout_windows = None
        if args.held_out_text:
            heldout_ids = tokenize_text_file(tokenizer, args.held_out_text)
            heldout_windows = windows_from_ids(heldout_ids, args.heldout_seq_len, args.heldout_num_windows, args.seed)
            run.metrics["heldout_nll"] = {
                "corpus_path": str(args.held_out_text.resolve()),
                "corpus_sha256": sha256_file(args.held_out_text),
                "windows": int(heldout_windows.shape[0]),
                "seq_len": int(heldout_windows.shape[1]),
                "windows_fingerprint": windows_fingerprint(heldout_windows),
                "base": evaluate_lm(model, heldout_windows, batch_size=1, device=device),
            }

        with records.open("x", encoding="utf-8") as output:
            for index, path in enumerate(paths):
                entry, layer = entries[path], stored_layers[path].to(device)
                random_inputs = torch.randn(args.probe_rows, entry.spec.in_features, generator=generator, device="cpu")
                prompt_x = prompt_inputs[path]
                combined = torch.cat((random_inputs, prompt_x), dim=0)
                parity = bounded_dense_action_parity(layer, combined, basis_chunk=args.basis_chunk)
                parity["serialized_parameters"] = _serialized_parameter_report(ckpt, path, stored_layers[path])
                actual_prompt_output = layer(prompt_x.to(device)).detach().cpu().float()
                teacher_error = _tensor_metrics(prompt_outputs[path], actual_prompt_output)
                parity.update(
                    {
                        "module": path,
                        "family": entry.family,
                        "layer": entry.module.split(".")[2],
                        "spec": entry.spec.to_dict(),
                        "storage_dtype": entry.storage_dtype,
                        "random_probe_rows": args.probe_rows,
                        "random_probe_fingerprint": hashlib.sha256(random_inputs.numpy().tobytes()).hexdigest(),
                        "teacher_activation_rows": int(prompt_x.shape[0]),
                        "teacher_activation_error": teacher_error,
                        "teacher_activation_input_distribution": {
                            "rms": _tensor_metrics(prompt_x, torch.zeros_like(prompt_x))["reference_rms"],
                            "rms_quantiles_by_token": _quantiles(prompt_x.double().square().mean(dim=-1).sqrt()),
                        },
                        "teacher_dense_output_distribution": {
                            "rms": _tensor_metrics(prompt_outputs[path], torch.zeros_like(prompt_outputs[path]))[
                                "reference_rms"
                            ],
                            "rms_quantiles_by_token": _quantiles(
                                prompt_outputs[path].double().square().mean(dim=-1).sqrt()
                            ),
                        },
                    }
                )
                if path in masters:
                    master = masters[path]
                    master_output = master(prompt_x).detach().cpu().float()
                    storage = {}
                    for dtype_name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                        cast = _cast_layer(master, dtype)
                        storage[dtype_name] = _tensor_metrics(
                            master_output, cast(prompt_x.float()).detach().cpu().float()
                        )
                    storage["serialized_vs_fp32_master"] = _tensor_metrics(
                        master_output, layer(prompt_x.to(device)).detach().cpu().float()
                    )
                    parity["serialization"] = storage
                    _append_jsonl(output, {"kind": "serialization", "module": path, "metrics": storage})
                run.metrics["operator_parity"][path] = parity
                _append_jsonl(output, {"kind": "projection-parity", "module": path, "metrics": parity})
                if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(paths):
                    run.log(f"parity {index + 1}/{len(paths)} {path}: max abs {parity['max_abs_error']:.4g}")

            run.metrics["teacher_traces"] = {
                "projection_inputs": {
                    path: {
                        "rows": int(prompt_inputs[path].shape[0]),
                        "rms": _tensor_metrics(prompt_inputs[path], torch.zeros_like(prompt_inputs[path]))[
                            "reference_rms"
                        ],
                        "rms_quantiles_per_token": _quantiles(
                            prompt_inputs[path].double().square().mean(dim=-1).sqrt()
                        ),
                    }
                    for path in paths
                },
                "projection_outputs": {
                    path: {
                        "rows": int(prompt_outputs[path].shape[0]),
                        "rms": _tensor_metrics(prompt_outputs[path], torch.zeros_like(prompt_outputs[path]))[
                            "reference_rms"
                        ],
                        "rms_quantiles_per_token": _quantiles(
                            prompt_outputs[path].double().square().mean(dim=-1).sqrt()
                        ),
                    }
                    for path in paths
                },
                "ffn_outputs": {
                    path: {
                        "rows": int(ffn_outputs[path].shape[0]),
                        "rms": _tensor_metrics(ffn_outputs[path], torch.zeros_like(ffn_outputs[path]))["reference_rms"],
                        "rms_quantiles_per_token": _quantiles(ffn_outputs[path].double().square().mean(dim=-1).sqrt()),
                    }
                    for path in ffn_outputs
                },
                "block_outputs": {
                    path: {
                        "rows": int(torch.cat(rows).shape[0]),
                        "rms": _tensor_metrics(torch.cat(rows), torch.zeros_like(torch.cat(rows)))["reference_rms"],
                        "rms_quantiles_per_token": _quantiles(torch.cat(rows).double().square().mean(dim=-1).sqrt()),
                    }
                    for path, rows in block_outputs.items()
                },
            }
            parity_rows = list(run.metrics["operator_parity"].values())
            run.metrics["operator_parity_summary"] = {
                "projections": len(parity_rows),
                "all_outputs_finite": all(row["finite_pairs"] == row["elements"] for row in parity_rows),
                "max_dense_action_abs_error": max(row["max_abs_error"] or 0 for row in parity_rows),
                "FP64_reference_tolerance": {
                    "rtol": 1e-10,
                    "atol": 1e-10,
                    "source": "tests/test_math.py RotorCascade forward/dense contract",
                },
                "model_precision_note": "FP32 action discrepancies are reported, not judged with a newly "
                "invented tolerance.",
            }

            if not args.parity_only:
                run.metrics["prompt_ablations"] = {}
                selected_layers = sorted({int(value) for value in args.layers.split(",") if value.strip()})
                for prompt, ids in zip(prompts, prompt_ids, strict=True):
                    baseline_logits = teacher_logits[prompt["id"]]
                    baseline_ids = greedy_decode(model, ids, args.max_new_tokens, device)
                    baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=False)
                    _append_jsonl(
                        output,
                        {
                            "kind": "generation",
                            "prompt_id": prompt["id"],
                            "arm": "dense-base",
                            "prompt_text": prompt["prompt"],
                            "input_ids": ids[0].tolist(),
                            "input_ids_sha256": prompt_fingerprints[prompt["id"]],
                            "output_ids": baseline_ids,
                            "output_text": baseline_text,
                            "output_text_visible": _visible_whitespace(baseline_text),
                            "generation_metrics": _generation_metrics(baseline_ids, baseline_text, tokenizer),
                        },
                    )
                    run.metrics["prompt_ablations"][prompt["id"]] = {
                        "dense_base_generation": _generation_metrics(baseline_ids, baseline_text, tokenizer)
                    }
                    available_layers = {int(entries[path].module.split(".")[2]) for path in paths}
                    for layer_index in selected_layers:
                        if layer_index not in available_layers:
                            continue
                        family_path = {
                            entries[path].family: path
                            for path in paths
                            if int(entries[path].module.split(".")[2]) == layer_index
                        }
                        arms: list[tuple[str, dict[str, RotorLinear], list[str]]] = []
                        for family in ("ffn.gate", "ffn.up", "ffn.down"):
                            if family in family_path:
                                path = family_path[family]
                                arms.append(
                                    (f"layer-{layer_index}-{family}", {path: stored_layers[path].to(device)}, [path])
                                )
                        families = ("ffn.gate", "ffn.up", "ffn.down")
                        if all(family in family_path for family in families):
                            ffn_paths = [family_path[family] for family in families]
                            arms.append(
                                (
                                    f"layer-{layer_index}-complete-ffn",
                                    {path: stored_layers[path].to(device) for path in ffn_paths},
                                    ffn_paths,
                                )
                            )
                            if all(path in masters for path in ffn_paths):
                                master_paths = {path: masters[path].to(device) for path in ffn_paths}
                                arms.append((f"layer-{layer_index}-complete-ffn-fp32-master", master_paths, ffn_paths))
                        for arm_name, replacements, changed_paths in arms:
                            heldout = None
                            with _replacement_arm(model, replacements):
                                changed_projection_outputs, changed_ffn_output, changed_block_output, changed_model = (
                                    _capture_intervention(model, ids, changed_paths, device)
                                )
                                generated_ids = greedy_decode(model, ids, args.max_new_tokens, device)
                                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                                if heldout_windows is not None:
                                    heldout = evaluate_lm(model, heldout_windows, batch_size=1, device=device)
                            mlp_path = f"model.layers.{layer_index}.mlp"
                            report = {
                                "logits": _kl_metrics(baseline_logits, changed_model.logits),
                                "ffn_output": _tensor_metrics(
                                    prompt_ffn_outputs[prompt["id"]][mlp_path], changed_ffn_output
                                ),
                                "block_hidden_state": _tensor_metrics(
                                    prompt_block_outputs[prompt["id"]][f"model.layers.{layer_index}"],
                                    changed_block_output,
                                ),
                                "projection_outputs": {
                                    path: _tensor_metrics(
                                        prompt_outputs_by_prompt[prompt["id"]][path], changed_projection_outputs[path]
                                    )
                                    for path in changed_paths
                                },
                                "heldout_nll": heldout,
                            }
                            run.metrics["prompt_ablations"][prompt["id"]][arm_name] = report
                            _append_jsonl(
                                output,
                                {
                                    "kind": "generation",
                                    "prompt_id": prompt["id"],
                                    "arm": arm_name,
                                    "prompt_text": prompt["prompt"],
                                    "input_ids": ids[0].tolist(),
                                    "input_ids_sha256": prompt_fingerprints[prompt["id"]],
                                    "output_ids": generated_ids,
                                    "output_text": generated_text,
                                    "output_text_visible": _visible_whitespace(generated_text),
                                    "generation_metrics": _generation_metrics(generated_ids, generated_text, tokenizer),
                                    **report,
                                },
                            )
                            run.log(
                                f"ablation {prompt['id']} {arm_name}: "
                                f"KL {report['logits']['teacher_to_student_kl_last_token']:.5g}; "
                                f"top agreement {report['logits']['top_token_agreement']:.3f}"
                            )

        run.metrics["diagnostic_jsonl"] = str(records)
        run.metrics["diagnostic_jsonl_sha256"] = sha256_file(records)
        run.summary = (
            f"Checked dense-action parity for {len(stored_layers)} stored projections; "
            f"progressive records: `{records}`."
        )
        run.interpretation = (
            "Reference-backend operator parity, serialization comparisons, and fixed-prompt "
            "interventions are diagnostics, not causal proof. "
            "Without supplied approved text, no held-out NLL or language-quality claim is made."
        )
        run.add_section(
            "Progressive diagnostic records",
            f"- JSONL: `{records}` (each record flushed and fsynced)\n"
            f"- Prompt SHA-256: `{run.metrics['provenance']['prompts_sha256']}`\n"
            f"- Stored projections: {len(stored_layers)}\n- Matched FP32 snapshot masters: {len(masters)}",
        )
        run.add_section(
            "Evidence boundaries",
            "- Backend: reference PyTorch RotorLinear only; no fused backend.\n"
            "- Dense-action oracle independently applies the serialized cascade stage equations "
            "with bounded basis chunks.\n"
            "- Fixed diagnostic prompts are not calibration/fitting data or held-out benchmark data.\n"
            "- FP16/BF16 interventions use the same FP32 snapshot master parameters and captured teacher activations.",
        )
    print(f"published run {run.path}; diagnostics {records}")
    return run.path
