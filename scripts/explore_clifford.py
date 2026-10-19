#!/usr/bin/env python
"""Run a no-training Clifford projection pilot on one local model tensor."""

from __future__ import annotations

import argparse

from rotor.converters.clifford_runner import (
    DEFAULT_MODEL,
    DEFAULT_TENSOR,
    create_experiment_run,
    run_clifford_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model key or Hugging Face id in configs/models.yaml")
    parser.add_argument("--tensor", default=DEFAULT_TENSOR, help="exact safetensors tensor name")
    parser.add_argument("--probe-rows", type=int, default=64, help="number of shared seeded FP64 Gaussian probes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = create_experiment_run(args.model, args.tensor, args.probe_rows)
    if args.probe_rows < 1:
        with run:
            run.config["configuration_error"] = "--probe-rows must be >= 1"
            raise ValueError("--probe-rows must be >= 1")
    with run:
        run.log(f"Loading local tensor {args.tensor!r} from model {args.model!r}; no download or fallback")
        run_clifford_experiment(args.model, args.tensor, args.probe_rows, run=run)
    print(f"run directory {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
