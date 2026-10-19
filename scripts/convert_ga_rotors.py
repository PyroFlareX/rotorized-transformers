#!/usr/bin/env python
"""Convert explicitly selected local 2D tensors to algebraically derived GA rotors; no training."""

from __future__ import annotations

import argparse
from pathlib import Path

from rotor.converters.algebraic import convert_ga_tensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="configured model key; uses existing local checkpoint, never downloads")
    source.add_argument("--base-dir", type=Path, help="local safetensors directory under configured NAS root")
    parser.add_argument("--tensor", action="append", required=True, help="exact source tensor name; repeat as needed")
    parser.add_argument("--name", required=True, help="new output directory name under configured derived root")
    parser.add_argument("--device", default="cpu", help="decomposition device; ROCm uses cuda:0")
    parser.add_argument("--storage-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--rank", type=int, help="explicit SVD truncation rank; default keeps all components")
    parser.add_argument(
        "--max-relative-error", type=float,
        help="required norm error gate for weights and action; default 1e-10 (FP64), 1e-5 (FP32)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-rows", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64, help="basis rows per reconstruction verification chunk")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination, report = convert_ga_tensors(
        tensor_names=args.tensor,
        name=args.name,
        model=args.model,
        base_dir=args.base_dir,
        device=args.device,
        storage_dtype=args.storage_dtype,
        rank=args.rank,
        max_relative_error=args.max_relative_error,
        seed=args.seed,
        probe_rows=args.probe_rows,
        chunk_size=args.chunk_size,
    )
    print(f"GA rotor checkpoint: {destination}")
    print(f"Verified experiment report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
