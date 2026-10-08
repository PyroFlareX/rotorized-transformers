#!/usr/bin/env python
"""Recovery fine-tuning of a converted checkpoint by KL distillation from the dense base (RotorLinear params only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.environment import ResourceMonitor
from rotor.formats import read_manifest
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
from rotor.reporting import ExperimentRun, markdown_table
from rotor.storage import load_storage
from rotor.training.recovery import export_recovered, train_recovery


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path, help="derived rotor-scaler checkpoint directory")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", type=Path, help="text used for training windows")
    src.add_argument("--random-tokens", action="store_true", help="uniform random tokens (SMOKE ONLY)")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-windows", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr-angles", type=float, default=1e-3)
    p.add_argument("--lr-scales", type=float, default=1e-3)
    p.add_argument("--lr-residual", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, help="output name (under derived root) or directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = load_storage()
    ckpt = Path(args.ckpt)
    manifest = read_manifest(ckpt)
    base_dir = Path(manifest.base.path)
    out = Path(args.out)
    out_dir = out if out.is_absolute() or len(out.parts) > 1 else storage.derived_path(args.out)
    storage.assert_writable(out_dir)
    if out_dir.exists():
        print(f"error: output {out_dir} already exists; recovery never overwrites", file=sys.stderr)
        return 2

    budget = manifest.conversion_config.get("budget", {})
    if budget.get("target"):
        target = str(budget["target"])
    elif budget.get("byte_ratio") is not None:
        target = f"b{round(float(budget['byte_ratio']) * 100):03d}"
    else:
        target = "na"
    config = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "grad_clip": args.grad_clip,
        "log_every": args.log_every,
        "seed": args.seed,
        "lrs": {"angles": args.lr_angles, "scales": args.lr_scales, "residual": args.lr_residual},
    }
    run_config = {
        **config,
        "ckpt": str(ckpt),
        "data": "random-tokens" if args.random_tokens else str(args.text_file),
        "seq_len": args.seq_len,
        "num_windows": args.num_windows,
        "device": args.device,
        "output": str(out_dir),
        "source_conversion_config": manifest.conversion_config,
    }

    with ExperimentRun(
        "recovery", model=manifest.base.hf_id, method="recovery-kd", target=target, config=run_config
    ) as run:
        dev = torch.device(args.device)
        teacher = load_base(base_dir, dtype=torch.float32, device=dev)
        student, _ = load_converted(ckpt, dtype=torch.float32, device=dev)
        vocab = teacher.config.vocab_size
        if args.random_tokens:
            run.log("WARNING: --random-tokens is a smoke test only; results say nothing about language quality.")
            windows = random_windows(vocab, args.seq_len, args.num_windows, args.seed)
        else:
            tok = load_tokenizer(base_dir)
            ids = tokenize_text_file(tok, args.text_file)
            windows = windows_from_ids(ids, args.seq_len, args.num_windows, args.seed)
        fingerprint = windows_fingerprint(windows)
        run.log(f"windows: {tuple(windows.shape)} sha256={fingerprint[:16]}")

        with ResourceMonitor() as mon:
            result = train_recovery(student, teacher, windows, config, log=run.log)
            export_recovered(student, ckpt, out_dir, storage)
        h = result["history"]
        run.metrics["training"] = {
            "steps": result["steps"],
            "trainable_params": result["trainable_params"],
            "kl_first": h[0]["kl"],
            "kl_last": h[-1]["kl"],
            "ce_first": h[0]["ce"],
            "ce_last": h[-1]["ce"],
            "loss_last": h[-1]["loss"],
            "history": h,
        }
        run.metrics["windows"] = {"shape": list(windows.shape), "sha256": fingerprint}
        run.metrics["resources"] = mon.as_dict()
        run.metrics["output"] = str(out_dir)
        stride = max(1, len(h) // 20)
        run.add_section(
            "Distillation loss history (measured; every ~1/20 of the steps plus the last)",
            markdown_table([r for i, r in enumerate(h) if i % stride == 0 or i == len(h) - 1]),
        )
        run.summary = (
            f"KL distillation from the dense base into {ckpt.name}: kl {h[0]['kl']:.5f} -> {h[-1]['kl']:.5f} "
            f"over {result['steps']} steps (measured on the training windows themselves, not a held-out set); "
            f"only RotorLinear parameters ({result['trainable_params']:,}) were trained."
        )
        run.add_log_reference(f"recovered checkpoint: `{out_dir}`")

    print(
        f"kl: first={h[0]['kl']:.5f} last={h[-1]['kl']:.5f} (measured); trainable params={result['trainable_params']}"
    )
    print(f"wrote {out_dir}\nrun record {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
