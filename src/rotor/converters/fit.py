"""Layer-local fitting of a ``RotorCascade`` to a dense weight (weight space) or to its activations."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from rotor.evaluation.reconstruction import activation_errors, weight_errors
from rotor.math.cascade import CascadeSpec, RotorCascade
from rotor.training.layerwise import FitSchedule, optimize_cascade

_TAIL = 10


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
    return cascade, W


def _run(
    cascade: RotorCascade,
    loss_fn: Callable[[RotorCascade], torch.Tensor],
    schedule: FitSchedule,
    log: Callable[[str], None] | None,
) -> tuple[list[float], float]:
    history = optimize_cascade(cascade, loss_fn, schedule, log)
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
) -> tuple[RotorCascade, dict]:
    """Minimize ``||W - W_hat||_F^2 / ||W||_F^2`` over float32 master parameters."""
    cascade, Wd = _prepare(W, spec, seed, init_angle_std, None, device)
    denom = Wd.pow(2).sum().clamp_min(torch.finfo(torch.float32).tiny)

    def loss_fn(c: RotorCascade) -> torch.Tensor:
        return (Wd - c.dense_weight()).pow(2).sum() / denom

    history, final = _run(cascade, loss_fn, schedule, log)
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

    history, final = _run(cascade, loss_fn, schedule, log)
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
