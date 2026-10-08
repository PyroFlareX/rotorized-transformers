#!/usr/bin/env python
"""Recovery fine-tuning of a converted checkpoint by KL distillation from the dense base (RotorLinear params only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.evaluation.records import write_run_record
from rotor.formats import read_manifest
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
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

    dev = torch.device(args.device)
    teacher = load_base(base_dir, dtype=torch.float32, device=dev)
    student, _ = load_converted(ckpt, dtype=torch.float32, device=dev)
    vocab = teacher.config.vocab_size
    if args.random_tokens:
        print("WARNING: --random-tokens is a smoke test only; results say nothing about language quality.")
        windows = random_windows(vocab, args.seq_len, args.num_windows, args.seed)
        data_src = "random-tokens"
    else:
        tok = load_tokenizer(base_dir)
        ids = tokenize_text_file(tok, args.text_file)
        windows = windows_from_ids(ids, args.seq_len, args.num_windows, args.seed)
        data_src = str(args.text_file)
    print(f"windows: {tuple(windows.shape)} sha256={windows_fingerprint(windows)[:16]}")

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
    result = train_recovery(student, teacher, windows, config)
    export_recovered(student, ckpt, out_dir, storage)
    rec = write_run_record(
        "recovery",
        out_dir.name,
        {
            "config": {
                **config,
                "ckpt": str(ckpt),
                "data": data_src,
                "seq_len": args.seq_len,
                "num_windows": args.num_windows,
                "device": args.device,
            },
            "windows_sha256": windows_fingerprint(windows),
            "result": result,
            "output": str(out_dir),
        },
    )
    h = result["history"]
    print(
        f"kl: first={h[0]['kl']:.5f} last={h[-1]['kl']:.5f} (measured); trainable params={result['trainable_params']}"
    )
    print(f"wrote {out_dir}\nrun record {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
