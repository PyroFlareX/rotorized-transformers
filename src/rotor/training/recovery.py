"""Recovery fine-tuning: KL distillation of a converted student towards its dense teacher.

Only parameters of ``RotorLinear`` modules are trained; everything else is frozen.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rotor.formats.checkpoint import read_manifest, write_structured
from rotor.math.cascade import RotorLinear
from rotor.storage import Storage
from rotor.training.optim import collect_param_groups, make_optimizer

DEFAULT_CONFIG = {
    "steps": 100,
    "batch_size": 1,
    "temperature": 1.0,
    "ce_weight": 0.0,
    "grad_clip": 1.0,
    "log_every": 10,
    "seed": 0,
    "lrs": {},
    "trainable": ["angles", "scales", "residual"],
    "weight_decay": 0.0,
}


def _logits(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids).logits


def train_recovery(
    student: nn.Module,
    teacher: nn.Module,
    windows: torch.Tensor,
    config: dict,
    log: Callable[[str], None] = print,
) -> dict:
    """Minimize ``T^2 * KL(teacher || student)`` (+ ``ce_weight`` * next-token CE) over ``windows``.

    Window order is a pure function of ``config['seed']``. Returns measured history
    ``[{step, loss, kl, ce}]`` (``ce`` is always measured; it only enters the loss if ``ce_weight > 0``).
    """
    cfg = {**DEFAULT_CONFIG, **config}
    steps, bs = int(cfg["steps"]), int(cfg["batch_size"])
    temp, ce_w = float(cfg["temperature"]), float(cfg["ce_weight"])
    if windows.ndim != 2 or windows.shape[0] == 0:
        raise ValueError("windows must be a non-empty (n, seq_len) tensor")

    for p in student.parameters():
        p.requires_grad_(False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    rotor_mods = [m for m in student.modules() if isinstance(m, RotorLinear)]
    if not rotor_mods:
        raise ValueError("student has no RotorLinear modules to train")
    groups = collect_param_groups(rotor_mods)
    opt = make_optimizer(groups, cfg["lrs"], tuple(cfg["trainable"]), float(cfg["weight_decay"]))
    params = [p for g in opt.param_groups for p in g["params"]]
    trainable = sum(p.numel() for p in params)

    device = next(student.parameters()).device
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    n = windows.shape[0]
    order: list[int] = []
    student.train()
    teacher.eval()
    history: list[dict] = []
    for step in range(1, steps + 1):
        while len(order) < bs:
            order.extend(torch.randperm(n, generator=gen).tolist())
        idx, order = order[:bs], order[bs:]
        ids = windows[idx].to(device)
        with torch.no_grad():
            t_logits = _logits(teacher, ids).float()
        s_logits = _logits(student, ids).float()
        v = s_logits.shape[-1]
        kl = F.kl_div(
            F.log_softmax(s_logits / temp, dim=-1).reshape(-1, v),
            F.log_softmax(t_logits / temp, dim=-1).reshape(-1, v),
            reduction="batchmean",
            log_target=True,
        ) * (temp**2)
        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, v), ids[:, 1:].reshape(-1))
        loss = kl + ce_w * ce if ce_w > 0 else kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
        opt.step()
        rec = {"step": step, "loss": float(loss), "kl": float(kl), "ce": float(ce)}
        history.append(rec)
        le = int(cfg["log_every"])
        if le and (step % le == 0 or step == 1 or step == steps):
            log(f"step {step}/{steps} loss={rec['loss']:.5f} kl={rec['kl']:.5f} ce={rec['ce']:.5f}")
    student.eval()
    student._recovery_config = {  # consumed by export_recovered
        **{
            k: cfg[k]
            for k in ("steps", "batch_size", "temperature", "ce_weight", "grad_clip", "seed", "lrs", "trainable")
        },
        "windows": int(n),
        "seq_len": int(windows.shape[1]),
        "final_loss": history[-1]["loss"] if history else None,
    }
    return {"history": history, "steps": steps, "trainable_params": trainable}


def export_recovered(student: nn.Module, src_ckpt_dir: Path, out_dir: Path, storage: Storage) -> Path:
    """Write a NEW derived checkpoint with the student's current RotorLinear tensors; never overwrites the source."""
    from rotor.storage import atomic_dir

    src_ckpt_dir, out_dir = Path(src_ckpt_dir), Path(out_dir)
    if out_dir.resolve() == src_ckpt_dir.resolve():
        raise ValueError("out_dir must differ from the source checkpoint")
    storage.assert_writable(out_dir)
    manifest = read_manifest(src_ckpt_dir)
    modules = dict(student.named_modules())
    layers: dict[str, RotorLinear] = {}
    for e in manifest.layers:
        m = modules.get(e.module)
        if not isinstance(m, RotorLinear):
            raise ValueError(f"student module {e.module!r} is not a RotorLinear")
        layers[e.module] = m
    new = copy.deepcopy(manifest)
    new.notes = [*new.notes, "recovery fine-tuned"]
    new.conversion_config = {**new.conversion_config, "recovery": getattr(student, "_recovery_config", {})}
    base_dir = Path(manifest.base.path)
    with atomic_dir(out_dir) as tmp:
        write_structured(tmp, new, layers, base_dir if base_dir.is_dir() else None)
    return out_dir
