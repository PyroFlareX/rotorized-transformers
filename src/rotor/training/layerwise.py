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
        return cls(
            phases=phases,
            lrs=dict(d.get("lrs", {})),
            cosine_decay=bool(d.get("cosine_decay", True)),
            log_every=int(d.get("log_every", 0)),
        )


def optimize_cascade(
    cascade: RotorCascade,
    loss_fn: Callable[[RotorCascade], torch.Tensor],
    schedule: FitSchedule,
    log: Callable[[str], None] | None = None,
    *,
    resume_state: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[float]:
    """Run a schedule, optionally resuming and reporting each completed optimizer step."""
    if resume_state is None:
        history: list[float] = []
        resume_phase = 0
        resume_step = 0
    else:
        required = {"phase_index", "next_step", "history", "cascade_state", "optimizer_state"}
        missing = required.difference(resume_state)
        if missing:
            raise ValueError(f"resume state is missing fields: {', '.join(sorted(missing))}")
        resume_phase = resume_state["phase_index"]
        resume_step = resume_state["next_step"]
        if (
            isinstance(resume_phase, bool)
            or not isinstance(resume_phase, int)
            or not 0 <= resume_phase < len(schedule.phases)
        ):
            raise ValueError("resume state phase_index is outside the schedule")
        if isinstance(resume_step, bool) or not isinstance(resume_step, int):
            raise ValueError("resume state next_step must be an integer")
        phase_steps = schedule.phases[resume_phase][1]
        if not 0 <= resume_step <= max(phase_steps, 0):
            raise ValueError("resume state next_step is outside the phase")
        if not isinstance(resume_state["history"], list):
            raise ValueError("resume state history must be a list")
        history = resume_state["history"]
        cascade.load_state_dict(resume_state["cascade_state"])

    groups = cascade.param_groups()
    for phase_index, (trainable, steps) in enumerate(schedule.phases):
        if phase_index < resume_phase:
            continue
        kinds = tuple(k for k in trainable if k in groups)
        if not kinds or steps <= 0:
            continue
        start_step = resume_step if phase_index == resume_phase else 0
        if start_step >= steps:
            continue
        opt = make_optimizer(groups, schedule.lrs, trainable=kinds)
        base = [g["lr"] for g in opt.param_groups]
        if resume_state is not None and phase_index == resume_phase:
            opt.load_state_dict(resume_state["optimizer_state"])
        for step in range(start_step, steps):
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
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase_index": phase_index,
                        "next_step": step + 1,
                        "history": history,
                        "cascade_state": cascade.state_dict(),
                        "optimizer_state": opt.state_dict(),
                    }
                )
    for ps in groups.values():
        for p in ps:
            p.requires_grad_(True)
    return history
