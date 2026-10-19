#!/usr/bin/env python
"""Jointly fit one FFN's gate, up, and down rotor projections on approved local text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rotor.converters.joint_ffn_runner import load_joint_config, run_joint_ffn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="joint FFN YAML config")
    parser.add_argument("--device", help="fit device override (ROCm PyTorch uses cuda); defaults to config")
    parser.add_argument("--steps-scale", type=float, help="multiply every optimizer phase's step count")
    parser.add_argument("--resume", type=Path, help="resume from a joint-snapshot directory")
    parser.add_argument(
        "--snapshot-interval-seconds", type=float, default=1800.0, help="snapshot interval (default: 1800)"
    )
    parser.add_argument("--snapshot-keep", type=int, default=3, help="newest snapshots to retain (default: 3)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_joint_config(args.config)
    destination = run_joint_ffn(
        config,
        config_path=args.config,
        device=args.device,
        steps_scale=args.steps_scale,
        resume_from=args.resume,
        snapshot_interval_seconds=args.snapshot_interval_seconds,
        snapshot_keep=args.snapshot_keep,
    )
    print(json.dumps({"output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
