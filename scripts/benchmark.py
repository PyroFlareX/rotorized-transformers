#!/usr/bin/env python
"""Microbenchmark dense nn.Linear vs RotorLinear vs LowRankLinear at equal stored bytes."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from rotor.evaluation.performance import linear_microbench
from rotor.evaluation.records import write_run_record
from rotor.formats import bytes_per_param
from rotor.math import LowRankLinear, RotorCascade, RotorLinear, Topology, rank_for_budget

NOTE = (
    "NOTE: the PyTorch reference RotorLinear is NOT an optimized kernel (sequential gather/rotate stages); its "
    "timings say nothing about what a fused kernel could achieve. 'estimated_flops' are formula estimates, "
    "not measurements; 'median_ms' etc. are measured on this device only."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_f", type=int, help="input features")
    p.add_argument("--out", dest="out_f", type=int, help="output features")
    p.add_argument("--model", help="key in configs/models.yaml; shapes from headers (needs --family)")
    p.add_argument("--family", default="ffn.up", help="ffn.gate | ffn.up | ffn.down")
    p.add_argument("--byte-ratio", type=float, default=0.25)
    p.add_argument("--batch-sizes", default="1,8,64")
    p.add_argument("--topology", default="butterfly", choices=["brick", "butterfly", "seeded"])
    p.add_argument("--stages-per-rotor", default="auto")
    p.add_argument("--residual-rank", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    return p


def _shape_from_model(key: str, family: str) -> tuple[int, int]:
    from rotor.models.discovery import local_tensors, remote_json, remote_tensors
    from rotor.models.qwen3_5 import discover_projections
    from rotor.storage import load_models, load_storage, model_entry

    _, entry = model_entry(key, load_models())
    base = load_storage().base_path(entry["local_path"])
    if base.is_dir():
        tensors = local_tensors(base)
        import json

        config = json.loads((base / "config.json").read_text())
    else:
        tensors = remote_tensors(entry["hf_id"], entry["revision"])
        config = remote_json(entry["hf_id"], "config.json", entry["revision"])
    for p in discover_projections(tensors, config):
        if p.family == family:
            return p.in_features, p.out_features
    raise SystemExit(f"family {family!r} not found for {key}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model:
        in_f, out_f = _shape_from_model(args.model, args.family)
    elif args.in_f and args.out_f:
        in_f, out_f = args.in_f, args.out_f
    else:
        raise SystemExit("give --in and --out, or --model KEY [--family F]")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    spr = args.stages_per_rotor if args.stages_per_rotor == "auto" else int(args.stages_per_rotor)

    from rotor.converters.budget import spec_for_budget

    # Equal stored bytes: budget is byte_ratio of the dense matrix; all params counted at the same width.
    bpp = bytes_per_param(args.dtype)
    budget_params = int(args.byte_ratio * in_f * out_f)
    spec = spec_for_budget(in_f, out_f, budget_params, Topology(args.topology, args.seed), spr, args.residual_rank)
    rank = rank_for_budget(in_f, out_f, budget_params)
    if rank < 1:
        raise SystemExit("byte ratio too small for rank-1 low-rank baseline")

    g = torch.Generator().manual_seed(args.seed)
    W = torch.randn(out_f, in_f, generator=g) / in_f**0.5
    dense = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        dense.weight.copy_(W)
    cascade = RotorCascade(spec, dtype=torch.float32)
    cascade.init_from_dense(W, angle_std=0.01, generator=g)
    impls = {
        "dense": dense.to(device=device, dtype=dtype),
        "rotor": RotorLinear(cascade).to(device=device, dtype=dtype),
        "lowrank": LowRankLinear.from_dense(W, rank).to(device=device, dtype=dtype),
    }
    stored = {
        "dense": in_f * out_f * bpp,
        "rotor": spec.param_count() * bpp,
        "lowrank": rank * (in_f + out_f) * bpp,
    }
    rows = linear_microbench(in_f, out_f, batch_sizes, impls, device, dtype)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    print(f"shape in={in_f} out={out_f} device={dev_name} dtype={args.dtype} byte_ratio={args.byte_ratio}")
    print("stored bytes (exact, from parameter counts):")
    for k, v in stored.items():
        print(f"  {k:<8}{v:>12} B  ({v / stored['dense']:.3f} of dense)")
    print(f"rotor spec: rotors={spec.num_rotors} stages/rotor={spec.stages_per_rotor} topology={args.topology}")
    print("\nMEASURED wall time (ms):")
    print(f"{'impl':<10}{'batch':>7}{'median':>10}{'p10':>10}{'p90':>10}")
    for r in rows:
        print(f"{r['impl']:<10}{r['batch_size']:>7}{r['median_ms']:>10.4f}{r['p10_ms']:>10.4f}{r['p90_ms']:>10.4f}")
    print("\nESTIMATED flops (formula, not measured):")
    for r in rows:
        print(f"{r['impl']:<10}{r['batch_size']:>7}{r.get('estimated_flops')!s:>16}")
    print("\n" + NOTE)

    rec = write_run_record(
        "benchmark",
        f"{args.model or f'{in_f}x{out_f}'}-{args.device}",
        {
            "config": {k: v for k, v in vars(args).items()},
            "shape": {"in_features": in_f, "out_features": out_f},
            "device_name": dev_name,
            "stored_bytes": stored,
            "rotor_spec": spec.to_dict(),
            "lowrank_rank": rank,
            "measured_rows": rows,
            "notes": [NOTE],
        },
    )
    print(f"run record {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
