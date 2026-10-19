#!/usr/bin/env python
"""Run bounded rotor parity, serialization, activation-trace, and fixed-prompt ablation diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rotor.evaluation.collapse import run_collapse_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="existing rotor-scaler checkpoint directory")
    parser.add_argument("--model", help="model catalog key or HF ID; no downloads are attempted")
    parser.add_argument("--base-dir", type=Path, help="local base model directory (defaults to manifest/catalog)")
    parser.add_argument("--snapshot", type=Path, help="trusted conversion snapshot containing FP32 masters")
    parser.add_argument("--device", default="cpu", help="cpu or an available torch device such as cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--prompts", type=Path, default=Path("configs/evaluation/prompts-256.jsonl"))
    parser.add_argument("--prompt-limit", type=int, default=1, help="number of fixed diagnostic prompts; default 1")
    parser.add_argument(
        "--probe-rows", type=int, default=16, help="deterministic random activation rows per projection"
    )
    parser.add_argument("--basis-chunk", type=int, default=64, help="maximum streamed identity rows for dense action")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="0,12,23", help="comma-separated FFN layers for replacement ablations")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--held-out-text", type=Path, help="explicit user-approved held-out corpus; NLL is omitted otherwise"
    )
    parser.add_argument("--heldout-seq-len", type=int, default=512)
    parser.add_argument("--heldout-num-windows", type=int, default=8)
    parser.add_argument("--output", type=Path, help="progressive JSONL path; must not already exist")
    parser.add_argument("--parity-only", action="store_true", help="skip generation and replacement ablations")
    parser.add_argument("--smoke", action="store_true", help="tiny known-correct CPU-only cascade parity smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.smoke and args.ckpt is None:
        raise SystemExit("--ckpt is required unless --smoke is selected")
    if args.smoke and args.device != "cpu":
        raise SystemExit("--smoke is CPU-only; omit --device")
    if args.prompt_limit < 1 or args.probe_rows < 1 or args.basis_chunk < 1 or args.max_new_tokens < 1:
        raise SystemExit("prompt/probe/chunk/token counts must be >= 1")
    if args.held_out_text and not args.held_out_text.is_file():
        raise SystemExit(f"held-out text file does not exist: {args.held_out_text}")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {args.device} is unavailable")
    path = run_collapse_diagnostics(args)
    print(f"published diagnostic report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
