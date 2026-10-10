"""Convert FFN projections of a base model into rotor-cascade form from an experiment YAML."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import yaml

from rotor.converters import convert
from rotor.converters.budget import COMPRESSION_TARGETS, parse_target
from rotor.converters.pipeline import ConversionConfig
from rotor.telemetry import RuntimeTelemetry


def build_config_dict(args: argparse.Namespace) -> dict:
    d = yaml.safe_load(args.config.read_text())
    if args.name:
        d["name"] = args.name
    if args.layers:
        d["layers"] = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    if args.target:
        parse_target(args.target)  # fail early with the list of presets
        budget = {k: v for k, v in (d.get("budget") or {}).items() if k != "byte_ratio"}
        budget["target"] = args.target
        d["budget"] = budget
    if args.device:
        d.setdefault("fit", {})["device"] = args.device
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
    ap.add_argument("--device", help="fit device override, e.g. cuda (ROCm builds also report cuda); default: config")
    ap.add_argument(
        "--target",
        help=f"compression target override: preset {sorted(COMPRESSION_TARGETS)} or a byte-ratio float in (0, 1]",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the allocation/byte plan JSON; fit nothing")
    ap.add_argument(
        "--no-report", action="store_true", help="do not publish an ExperimentRun under reports/experiments"
    )
    ap.add_argument("--telemetry", type=Path, help="new JSONL path for 1 Hz runtime resource samples")
    ap.add_argument("--snapshot-interval-seconds", type=float, default=1800, help="seconds between resumable snapshots")
    ap.add_argument("--snapshot-keep", type=int, default=3, help="number of newest snapshots to retain")
    ap.add_argument("--resume", type=Path, help="resume from a conversion snapshot directory")
    args = ap.parse_args(argv)

    cfg = ConversionConfig.from_dict(build_config_dict(args))
    info: dict = {}
    telemetry = RuntimeTelemetry(args.telemetry) if args.telemetry is not None else nullcontext()
    with telemetry as sampler:
        if sampler is not None:
            sampler.set_phase("conversion-plan" if args.dry_run else "conversion-fit")
        res = convert(
            cfg,
            base_dir=args.base_dir,
            out_dir=args.out,
            dry_run=args.dry_run,
            report=not args.no_report,
            run_info=info,
            snapshot_interval_seconds=args.snapshot_interval_seconds,
            snapshot_keep=args.snapshot_keep,
            resume_from=args.resume,
        )
    if args.dry_run:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(f"wrote converted checkpoint: {res}")
        if info.get("run_dir") is not None:
            print(f"experiment run directory: {info['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
