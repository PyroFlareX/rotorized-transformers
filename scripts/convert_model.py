"""Convert FFN projections of a base model into rotor-cascade form from an experiment YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rotor.converters import convert
from rotor.converters.pipeline import ConversionConfig


def build_config_dict(args: argparse.Namespace) -> dict:
    d = yaml.safe_load(args.config.read_text())
    if args.name:
        d["name"] = args.name
    if args.layers:
        d["layers"] = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    if args.steps_scale is not None:
        if args.steps_scale <= 0:
            raise SystemExit("--steps-scale must be > 0")
        for ph in d.get("fit", {}).get("schedule", {}).get("phases", []):
            ph["steps"] = max(1, int(round(int(ph["steps"]) * args.steps_scale)))
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="configs/experiments/X.yaml")
    ap.add_argument("--base-dir", type=Path, help="override base checkpoint directory")
    ap.add_argument("--out", type=Path, help="override output directory (default: derived_root/<name>)")
    ap.add_argument("--name", help="override derived output name")
    ap.add_argument("--layers", help="comma-separated layer indices, e.g. 0,1")
    ap.add_argument("--steps-scale", type=float, help="multiply every phase's step count")
    ap.add_argument("--dry-run", action="store_true", help="print the allocation/byte plan JSON; fit nothing")
    args = ap.parse_args(argv)

    cfg = ConversionConfig.from_dict(build_config_dict(args))
    res = convert(cfg, base_dir=args.base_dir, out_dir=args.out, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(f"wrote converted checkpoint: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
