"""Layer-local projection fitting and joint SiLU-gated FFN output fitting."""

from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from rotor.evaluation.reconstruction import activation_errors, weight_errors
from rotor.formats.spec import bytes_per_param
from rotor.math.cascade import CascadeSpec, RotorCascade, RotorLinear
from rotor.training.layerwise import FitSchedule, optimize_cascade
from rotor.training.optim import collect_param_groups, make_optimizer

_TAIL = 10

_JOINT_FFN_ROW_BATCH = 256


def _prepare(
    W: torch.Tensor,
    spec: CascadeSpec,
    seed: int,
    init_angle_std: float,
    init: RotorCascade | None,
    device,
) -> tuple[RotorCascade, torch.Tensor]:
    W = W.detach().to(device=device, dtype=torch.float32)
    if tuple(W.shape) != (spec.out_features, spec.in_features):
        raise ValueError(f"W shape {tuple(W.shape)} does not match spec {(spec.out_features, spec.in_features)}")
    if init is not None:
        if init.spec != spec:
            raise ValueError("init cascade spec differs from the requested spec")
        cascade = copy.deepcopy(init).to(device=device, dtype=torch.float32)
    else:
        cascade = RotorCascade(spec, dtype=torch.float32, device=device)
        gen = torch.Generator(device=torch.device(device)).manual_seed(seed)
        cascade.init_from_dense(W, angle_std=init_angle_std, generator=gen)
    cascade.grad_checkpoint = True
    return cascade, W


def _run(
    cascade: RotorCascade,
    loss_fn: Callable[[RotorCascade], torch.Tensor],
    schedule: FitSchedule,
    log: Callable[[str], None] | None,
    *,
    resume_state: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[list[float], float]:
    history = optimize_cascade(
        cascade,
        loss_fn,
        schedule,
        log,
        resume_state=resume_state,
        progress_callback=progress_callback,
    )
    if history:
        final = history[-1]
    else:
        with torch.no_grad():
            final = float(loss_fn(cascade))
    return history, final


def fit_weight(
    W: torch.Tensor,
    spec: CascadeSpec,
    schedule: FitSchedule,
    seed: int = 0,
    init_angle_std: float = 0.01,
    device="cpu",
    log: Callable[[str], None] | None = None,
    *,
    resume_state: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[RotorCascade, dict]:
    """Minimize ``||W - W_hat||_F^2 / ||W||_F^2`` over float32 master parameters."""
    cascade, Wd = _prepare(W, spec, seed, init_angle_std, None, device)
    denom = Wd.pow(2).sum().clamp_min(torch.finfo(torch.float32).tiny)

    def loss_fn(c: RotorCascade) -> torch.Tensor:
        return (Wd - c.dense_weight()).pow(2).sum() / denom

    history, final = _run(
        cascade, loss_fn, schedule, log, resume_state=resume_state, progress_callback=progress_callback
    )
    with torch.no_grad():
        metrics = {
            "objective": "weight",
            "final_loss": final,
            "loss_history_tail": history[-_TAIL:],
            "weight": weight_errors(Wd, cascade.dense_weight()),
            "steps": len(history),
        }
    return cascade, metrics


def fit_activation(
    W: torch.Tensor,
    X: torch.Tensor,
    spec: CascadeSpec,
    schedule: FitSchedule,
    seed: int = 0,
    init_angle_std: float = 0.01,
    init: RotorCascade | None = None,
    device="cpu",
    log: Callable[[str], None] | None = None,
    *,
    resume_state: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[RotorCascade, dict]:
    """Minimize ``||X W^T - X W_hat^T||_F^2 / ||X W^T||_F^2`` (``X``: calibration rows x in)."""
    cascade, Wd = _prepare(W, spec, seed, init_angle_std, init, device)
    Xd = X.detach().to(device=device, dtype=torch.float32)
    if Xd.ndim != 2 or Xd.shape[1] != spec.in_features:
        raise ValueError(f"X must be (rows, {spec.in_features}), got {tuple(Xd.shape)}")
    Y = Xd @ Wd.T
    denom = Y.pow(2).sum().clamp_min(torch.finfo(torch.float32).tiny)

    def loss_fn(c: RotorCascade) -> torch.Tensor:
        return (c(Xd) - Y).pow(2).sum() / denom

    history, final = _run(
        cascade, loss_fn, schedule, log, resume_state=resume_state, progress_callback=progress_callback
    )
    with torch.no_grad():
        W_hat = cascade.dense_weight()
        metrics = {
            "objective": "activation",
            "final_loss": final,
            "loss_history_tail": history[-_TAIL:],
            "weight": weight_errors(Wd, W_hat),
            "activation": activation_errors(Wd, W_hat, Xd),
            "steps": len(history),
        }
    return cascade, metrics


_JOINT_FFN_KEYS = ("gate", "up", "down")


def joint_ffn_output(
    x: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    biases: Mapping[str, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Evaluate the dense SiLU-gated FFN, preserving checkpoint-dtype linear/activation semantics.

    ``weights`` use PyTorch linear layout (out_features, in_features) and keys ``gate``, ``up``, and ``down``.
    Inputs are cast to each projection's checkpoint dtype and biases are applied there. Output rows are promoted to
    FP32 for loss accumulation; evaluation is chunked to bound intermediate activation memory.
    """
    if set(weights) != set(_JOINT_FFN_KEYS):
        raise ValueError(f"weights must contain exactly {_JOINT_FFN_KEYS}")
    if biases is not None and not set(biases).issubset(_JOINT_FFN_KEYS):
        raise ValueError(f"biases may contain only {_JOINT_FFN_KEYS}")
    device = x.device
    gate_weight = weights["gate"].detach().to(device=device)
    up_weight = weights["up"].detach().to(device=device)
    down_weight = weights["down"].detach().to(device=device)
    gate_bias = None if biases is None or biases.get("gate") is None else biases["gate"].detach().to(device=device)
    up_bias = None if biases is None or biases.get("up") is None else biases["up"].detach().to(device=device)
    down_bias = None if biases is None or biases.get("down") is None else biases["down"].detach().to(device=device)
    outputs = []
    for start in range(0, x.shape[0], _JOINT_FFN_ROW_BATCH):
        rows = x[start : start + _JOINT_FFN_ROW_BATCH]
        gate = F.linear(rows.to(dtype=gate_weight.dtype), gate_weight, gate_bias)
        up = F.linear(rows.to(dtype=up_weight.dtype), up_weight, up_bias)
        hidden = F.silu(gate) * up
        output = F.linear(hidden.to(dtype=down_weight.dtype), down_weight, down_bias)
        outputs.append(output.to(dtype=torch.float32))
    return torch.cat(outputs, dim=0)


def joint_ffn_student_output(x: torch.Tensor, modules: Mapping[str, RotorLinear]) -> torch.Tensor:
    """Compose fitted gate/up/down ``RotorLinear`` modules with exact SiLU-gate semantics."""
    if set(modules) != set(_JOINT_FFN_KEYS):
        raise ValueError(f"modules must contain exactly {_JOINT_FFN_KEYS}")
    return _joint_ffn_rows(x, modules)


def _validate_joint_ffn_inputs(
    X_train: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    specs: Mapping[str, CascadeSpec],
    biases: Mapping[str, torch.Tensor | None] | None,
    heldout_X: torch.Tensor | None,
) -> None:
    if set(weights) != set(_JOINT_FFN_KEYS) or set(specs) != set(_JOINT_FFN_KEYS):
        raise ValueError(f"weights and specs must contain exactly {_JOINT_FFN_KEYS}")
    if biases is not None and not set(biases).issubset(_JOINT_FFN_KEYS):
        raise ValueError(f"biases may contain only {_JOINT_FFN_KEYS}")
    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError("X_train must be a non-empty (rows, hidden) tensor")
    for key in _JOINT_FFN_KEYS:
        weight, spec = weights[key], specs[key]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"weights[{key!r}] must be a rank-2 tensor")
        if tuple(weight.shape) != (spec.out_features, spec.in_features):
            raise ValueError(
                f"weights[{key!r}] shape {tuple(weight.shape)} does not match spec "
                f"{(spec.out_features, spec.in_features)}"
            )
        bias = None if biases is None else biases.get(key)
        if bias is not None and (
            not isinstance(bias, torch.Tensor) or bias.ndim != 1 or bias.numel() != spec.out_features
        ):
            raise ValueError(f"biases[{key!r}] must have shape ({spec.out_features},)")
    if specs["gate"].in_features != X_train.shape[1] or specs["up"].in_features != X_train.shape[1]:
        raise ValueError("gate and up input widths must equal X_train.shape[1]")
    if specs["gate"].out_features != specs["up"].out_features:
        raise ValueError("gate and up output widths must match")
    if specs["down"].in_features != specs["gate"].out_features:
        raise ValueError("down input width must equal gate/up output width")
    if heldout_X is not None:
        if heldout_X.ndim != 2 or heldout_X.shape[0] == 0 or heldout_X.shape[1] != X_train.shape[1]:
            raise ValueError(f"heldout_X must be a non-empty (rows, {X_train.shape[1]}) tensor")


def _check_disjoint_rows(train: torch.Tensor, heldout: torch.Tensor) -> None:
    """Reject exact fixed-row overlap without materializing an N x M x width comparison tensor."""
    train_cpu = train.detach().to(device="cpu", dtype=torch.float32).contiguous()
    heldout_cpu = heldout.detach().to(device="cpu", dtype=torch.float32).contiguous()
    fingerprints: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(train_cpu):
        fingerprints[hashlib.blake2b(memoryview(row.numpy()), digest_size=16).digest()].append(index)
    for row in heldout_cpu:
        digest = hashlib.blake2b(memoryview(row.numpy()), digest_size=16).digest()
        if any(torch.equal(row, train_cpu[index]) for index in fingerprints.get(digest, ())):
            raise ValueError("heldout_X contains rows also present in X_train; fit and held-out rows must be disjoint")


def _joint_ffn_objective(
    x: torch.Tensor,
    target: torch.Tensor,
    modules: Mapping[str, RotorLinear],
    *,
    backward: bool = False,
) -> float:
    """Evaluate normalized squared error in row batches; optionally accumulate gradients batch by batch."""
    denominator = target.square().sum().clamp_min(torch.finfo(torch.float32).tiny)
    numerator_total = torch.zeros((), dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[0], _JOINT_FFN_ROW_BATCH):
        rows = x[start : start + _JOINT_FFN_ROW_BATCH]
        expected = target[start : start + _JOINT_FFN_ROW_BATCH]
        prediction = _joint_ffn_rows(rows, modules)
        numerator = (prediction - expected).square().sum()
        numerator_total.add_(numerator.detach())
        if backward:
            (numerator / denominator).backward()
    return float(numerator_total / denominator)


def _joint_ffn_rows(
    x: torch.Tensor,
    modules: Mapping[str, RotorLinear],
    row_batch: int = _JOINT_FFN_ROW_BATCH,
) -> torch.Tensor:
    """Compose rows in bounded batches to cap activation memory on real calibration sets."""
    outputs = []
    for start in range(0, x.shape[0], row_batch):
        rows = x[start : start + row_batch]
        gate = modules["gate"](rows)
        up = modules["up"](rows)
        outputs.append(modules["down"](F.silu(gate) * up))
    return torch.cat(outputs, dim=0)


def _validate_joint_resume_state(resume_state: dict, schedule: FitSchedule) -> tuple[int, int, list[float]]:
    required = {"phase_index", "next_step", "history", "module_states", "optimizer_state"}
    if not isinstance(resume_state, dict) or set(resume_state) != required:
        raise ValueError(
            "joint FFN resume state must contain phase_index, next_step, history, module_states, optimizer_state"
        )
    phase, step, history = resume_state["phase_index"], resume_state["next_step"], resume_state["history"]
    if (
        isinstance(phase, bool)
        or not isinstance(phase, int)
        or not 0 <= phase < len(schedule.phases)
        or isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= max(schedule.phases[phase][1], 0)
    ):
        raise ValueError("joint FFN resume phase_index or next_step is outside the schedule")
    if not isinstance(history, list) or not all(
        isinstance(value, int | float) and not isinstance(value, bool) for value in history
    ):
        raise ValueError("joint FFN resume history must be a list of numeric losses")
    if not isinstance(resume_state["module_states"], dict) or set(resume_state["module_states"]) != set(
        _JOINT_FFN_KEYS
    ):
        raise ValueError("joint FFN resume module_states must contain gate, up, and down")
    if not isinstance(resume_state["optimizer_state"], dict) or set(resume_state["optimizer_state"]) != {
        "state",
        "param_groups",
    }:
        raise ValueError("joint FFN resume optimizer_state is invalid")
    return phase, step, list(history)


def _joint_ffn_byte_accounting(
    weights: Mapping[str, torch.Tensor],
    specs: Mapping[str, CascadeSpec],
    biases: Mapping[str, torch.Tensor | None] | None,
    storage_dtype: str,
    combined_byte_budget: int | None,
) -> dict:
    storage_bpp = bytes_per_param(storage_dtype)
    per_projection_params = {}
    per_projection_bytes = {}
    per_projection_dense_bytes = {}
    for key in _JOINT_FFN_KEYS:
        bias = None if biases is None else biases.get(key)
        bias_count = 0 if bias is None else bias.numel()
        per_projection_params[key] = {**specs[key].param_breakdown(), "bias": bias_count}
        per_projection_bytes[key] = (specs[key].param_count() + bias_count) * storage_bpp
        per_projection_dense_bytes[key] = weights[key].numel() * weights[key].element_size() + (
            0 if bias is None else bias.numel() * bias.element_size()
        )
    combined_bytes = sum(per_projection_bytes.values())
    combined_dense_bytes = sum(per_projection_dense_bytes.values())
    if combined_byte_budget is not None and combined_bytes > combined_byte_budget:
        raise ValueError(
            f"joint structured payload is {combined_bytes} bytes, exceeding "
            f"combined_byte_budget={combined_byte_budget}; "
            f"per-projection bytes={per_projection_bytes}"
        )
    return {
        "per_projection_params": per_projection_params,
        "per_projection_bytes": per_projection_bytes,
        "per_projection_dense_bytes": per_projection_dense_bytes,
        "combined_bytes": combined_bytes,
        "combined_dense_bytes": combined_dense_bytes,
        "combined_byte_budget": combined_byte_budget,
        "combined_byte_ratio": combined_bytes / combined_dense_bytes if combined_dense_bytes else None,
        "storage_dtype": storage_dtype,
    }


def fit_joint_ffn(
    X_train: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    specs: Mapping[str, CascadeSpec],
    schedule: FitSchedule,
    *,
    biases: Mapping[str, torch.Tensor | None] | None = None,
    heldout_X: torch.Tensor | None = None,
    seed: int = 0,
    init_angle_std: float = 0.01,
    init: Mapping[str, RotorCascade | RotorLinear] | None = None,
    device="cpu",
    log: Callable[[str], None] | None = None,
    resume_state: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    storage_dtype: str = "bfloat16",
    combined_byte_budget: int | None = None,
) -> tuple[dict[str, RotorLinear], dict]:
    """Jointly fit gate/up/down cascades to normalized composed FFN output error.

    The teacher follows source projection checkpoint dtypes, captured FP32 inputs are cast for each linear, and
    output error is accumulated in FP32. Student cascades and biases remain FP32 masters. ``heldout_X`` is
    evaluation-only and exact duplicate rows across train and held-out sets are rejected. Progress callbacks receive
    all three module states and the shared optimizer state, suitable for weights-only snapshots.

    ``combined_byte_budget`` applies to combined tensor-payload bytes at ``storage_dtype`` (including biases). It
    excludes safetensors header and manifest metadata, matching existing converter budgeting.
    """
    _validate_joint_ffn_inputs(X_train, weights, specs, biases, heldout_X)
    if init is not None:
        if not set(init).issubset(_JOINT_FFN_KEYS):
            raise ValueError(f"init may contain only {_JOINT_FFN_KEYS}")
        if any(not isinstance(module, (RotorCascade, RotorLinear)) for module in init.values()):
            raise TypeError("init values must be RotorCascade or RotorLinear instances")
    if combined_byte_budget is not None and (
        isinstance(combined_byte_budget, bool) or not isinstance(combined_byte_budget, int) or combined_byte_budget < 0
    ):
        raise ValueError("combined_byte_budget must be a non-negative integer or None")
    accounting = _joint_ffn_byte_accounting(weights, specs, biases, storage_dtype, combined_byte_budget)
    device = torch.device(device)
    train_x = X_train.detach().to(device=device, dtype=torch.float32)
    heldout_x = None if heldout_X is None else heldout_X.detach().to(device=device, dtype=torch.float32)
    if heldout_x is not None:
        _check_disjoint_rows(train_x, heldout_x)

    source_weights = {key: weights[key].detach().to(device=device) for key in _JOINT_FFN_KEYS}
    source_biases = {
        key: (None if biases is None or biases.get(key) is None else biases[key].detach().to(device=device))
        for key in _JOINT_FFN_KEYS
    }
    teacher_weights = {key: source_weights[key].to(dtype=torch.float32) for key in _JOINT_FFN_KEYS}
    teacher_biases = {
        key: (None if source_biases[key] is None else source_biases[key].to(dtype=torch.float32))
        for key in _JOINT_FFN_KEYS
    }
    with torch.no_grad():
        teacher_train = joint_ffn_output(train_x, source_weights, source_biases)
        teacher_heldout = None if heldout_x is None else joint_ffn_output(heldout_x, source_weights, source_biases)

    modules: dict[str, RotorLinear] = {}
    for offset, key in enumerate(_JOINT_FFN_KEYS):
        start = None if init is None else init.get(key)
        if isinstance(start, RotorLinear):
            if start.cascade.spec != specs[key]:
                raise ValueError(f"init cascade for {key!r} has a different CascadeSpec")
            cascade = copy.deepcopy(start.cascade).to(device=device, dtype=torch.float32)
        elif isinstance(start, RotorCascade):
            if start.spec != specs[key]:
                raise ValueError(f"init cascade for {key!r} has a different CascadeSpec")
            cascade = copy.deepcopy(start).to(device=device, dtype=torch.float32)
        else:
            cascade = RotorCascade(specs[key], dtype=torch.float32, device=device)
            generator = torch.Generator(device=device).manual_seed(int(seed) + offset)
            cascade.init_from_dense(teacher_weights[key], angle_std=init_angle_std, generator=generator)
        cascade.grad_checkpoint = True
        bias = teacher_biases[key]
        modules[key] = RotorLinear(cascade, bias).to(device=device, dtype=torch.float32)

    if resume_state is not None:
        resume_phase, resume_step, history = _validate_joint_resume_state(resume_state, schedule)
        for key in _JOINT_FFN_KEYS:
            modules[key].load_state_dict(resume_state["module_states"][key], strict=True)
    else:
        resume_phase, resume_step, history = 0, 0, []

    groups = collect_param_groups([modules[key] for key in _JOINT_FFN_KEYS])
    for phase_index, (trainable, phase_steps) in enumerate(schedule.phases):
        if phase_index < resume_phase:
            continue
        kinds = tuple(kind for kind in trainable if kind in groups)
        if not kinds or phase_steps <= 0:
            continue
        start_step = resume_step if phase_index == resume_phase else 0
        if start_step >= phase_steps:
            continue
        optimizer = make_optimizer(groups, schedule.lrs, trainable=kinds)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        if resume_state is not None and phase_index == resume_phase:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        for step in range(start_step, phase_steps):
            if schedule.cosine_decay:
                fraction = 0.5 * (1 + math.cos(math.pi * step / phase_steps))
                for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base_lr * (0.05 + 0.95 * fraction)
            optimizer.zero_grad(set_to_none=True)
            loss = _joint_ffn_objective(train_x, teacher_train, modules, backward=True)
            optimizer.step()
            history.append(loss)
            if log and schedule.log_every and step % schedule.log_every == 0:
                log(f"[joint:{'+'.join(kinds)}] step {step}/{phase_steps} loss {history[-1]:.6g}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase_index": phase_index,
                        "next_step": step + 1,
                        "history": history,
                        "module_states": {key: modules[key].state_dict() for key in _JOINT_FFN_KEYS},
                        "optimizer_state": optimizer.state_dict(),
                    }
                )
    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(True)

    train_error = _joint_ffn_objective(train_x, teacher_train, modules)
    heldout_error = None if heldout_x is None else _joint_ffn_objective(heldout_x, teacher_heldout, modules)

    # Tensor payload accounting is derived before optimization from the exact specs and source tensor dtypes.
    metrics = {
        "objective": "joint_ffn_output",
        "final_loss": float(train_error),
        "train_output_relative_error": float(train_error),
        "heldout_output_relative_error": None if heldout_error is None else float(heldout_error),
        "loss_history_tail": history[-_TAIL:],
        "steps": len(history),
        **accounting,
    }
    return modules, metrics
