#!/usr/bin/env python
"""Run the 256 curated raw prompts on a local base or rotor-converted checkpoint.

No chat template is applied: each prompt is tokenized as authored and greedily continued. Results are
progressively flushed to JSONL, while runtime telemetry and an ExperimentRun preserve run evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rotor.evaluation.prompts import PROMPTS_PATH, run_prompt_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, help="local base model directory; no downloads are attempted")
    parser.add_argument("--model", help="model key or HF id in configs/models.yaml (resolved under configured storage)")
    parser.add_argument("--ckpt", type=Path, help="local converted rotor checkpoint directory")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH, help="JSONL prompt corpus (default: curated 256)")
    parser.add_argument("--limit", type=int, help="run only the first N prompts (for real-model smoke runs)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="greedy continuation length (default: 64)")
    parser.add_argument("--seed", type=int, default=0, help="recorded RNG seed (default: 0)")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0 (default: cpu)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument(
        "--rotor-backend",
        choices=("reference", "triton"),
        default="reference",
        help="converted rotor-layer inference backend (default: reference)",
    )
    parser.add_argument("--compare-to", type=Path, help="base-model result JSONL to validate and compare per prompt")
    parser.add_argument("--output", type=Path, help="progressive results JSONL path; must not already exist")
    parser.add_argument("--telemetry-path", type=Path, help="RuntimeTelemetry JSONL path; must not already exist")
    parser.add_argument("--name", help="optional model label in the report")
    parser.add_argument(
        "--run-note",
        help="timing/environment note stored in report (excluded from comparison settings)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.ckpt and not args.base_dir and not args.model:
        parser.error("provide --base-dir or --model for a base run, or --ckpt for a converted run")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.ckpt and not args.ckpt.is_dir():
        parser.error(f"converted checkpoint does not exist or is not a directory: {args.ckpt}")
    if args.base_dir and not args.base_dir.is_dir():
        parser.error(f"base model directory does not exist: {args.base_dir}")
    if args.compare_to and not args.ckpt:
        parser.error("--compare-to is for a converted run; run the base first without this option")
    if not args.ckpt and args.rotor_backend != "reference":
        parser.error("--rotor-backend triton requires --ckpt")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_prompt_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
