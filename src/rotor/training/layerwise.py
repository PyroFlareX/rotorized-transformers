"""Layer-local optimization loop shared by the converters (rotor-angle, scale, residual fitting)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from rotor.math.cascade import RotorCascade
from rotor.training.optim import make_optimizer


@dataclass
class FitSchedule:
    """``phases`` is a list of ``(trainable_kinds, steps)``; e.g. scales-only warmup, then everything."""

    phases: list[tuple[tuple[str, ...], int]] = field(
        default_factory=lambda: [(("scales",), 100), (("angles", "scales", "residual"), 900)]
    )
    lrs: dict[str, float] = field(default_factory=dict)
    cosine_decay: bool = True
    log_every: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> FitSchedule:
        phases = [(tuple(p["train"]), int(p["steps"])) for p in d.get("phases", [])] or cls().phases
        return cls(phases=phases, lrs=dict(d.get("lrs", {})), cosine_decay=bool(d.get("cosine_decay", True)),
                   log_every=int(d.get("log_every", 0)))


def optimize_cascade(
    cascade: RotorCascade,
    loss_fn: Callable[[RotorCascade], torch.Tensor],
    schedule: FitSchedule,
    log: Callable[[str], None] | None = None,
) -> list[float]:
    """Run ``schedule`` minimizing ``loss_fn(cascade)``. Returns the per-step loss history (measured)."""
    history: list[float] = []
    groups = cascade.param_groups()
    for trainable, steps in schedule.phases:
        kinds = tuple(k for k in trainable if k in groups)
        if not kinds or steps <= 0:
            continue
        opt = make_optimizer(groups, schedule.lrs, trainable=kinds)
        base = [g["lr"] for g in opt.param_groups]
        for step in range(steps):
            if schedule.cosine_decay:
                f = 0.5 * (1 + math.cos(math.pi * step / steps))
                for g, b in zip(opt.param_groups, base, strict=True):
                    g["lr"] = b * (0.05 + 0.95 * f)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(cascade)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
            if log and schedule.log_every and step % schedule.log_every == 0:
                log(f"[{'+'.join(kinds)}] step {step}/{steps} loss {history[-1]:.6g}")
    for ps in groups.values():
        for p in ps:
            p.requires_grad_(True)
    return history
