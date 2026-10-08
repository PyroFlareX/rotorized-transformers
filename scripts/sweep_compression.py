"""Weight-space equal-stored-bytes sweep (SVD vs rotor vs random controls) on selected FFN tensors.

Reads individual tensors via safetensors from the base dir; never loads the full model or writes checkpoints.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from rotor.evaluation.baselines import equal_budget_baselines
from rotor.evaluation.records import write_run_record
from rotor.math import Topology
from rotor.models.discovery import local_tensors
from rotor.models.qwen3_5 import discover_projections, select_projections
from rotor.storage import load_models, load_storage, model_entry
from rotor.training.layerwise import FitSchedule

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


def _clean(o):
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() > 1 else o.item()
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="base experiment YAML")
    ap.add_argument("--base-dir", type=Path)
    ap.add_argument("--byte-ratios", default="0.1,0.25,0.5")
    ap.add_argument("--topologies", default="butterfly,brick,seeded")
    ap.add_argument("--residual-ranks", default="0")
    ap.add_argument("--layers", help="comma-separated layer indices (default: config's layers)")
    ap.add_argument("--max-tensors", type=int, help="limit number of selected tensors")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="print the planned combos; fit nothing")
    ap.add_argument("--name", help="run record name (default: config name + '-sweep')")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    ratios = [float(x) for x in args.byte_ratios.split(",")]
    topos = [x.strip() for x in args.topologies.split(",")]
    ranks = [int(x) for x in args.residual_ranks.split(",")]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else cfg.get("layers")
    storage_dtype = cfg.get("budget", {}).get("storage_dtype", "bfloat16")
    cascade = cfg.get("cascade", {})
    seed = int(cfg.get("fit", {}).get("seed", 0))
    topo_seed = int(cascade.get("topology", {}).get("seed", 0))
    sched_d = json.loads(json.dumps(cfg.get("fit", {}).get("schedule", {})))
    for ph in sched_d.get("phases", []):
        ph["steps"] = max(1, int(round(int(ph["steps"]) * args.steps_scale)))
    schedule = FitSchedule.from_dict(sched_d)

    if args.base_dir:
        base = args.base_dir
    else:
        key, entry = model_entry(cfg["model"], load_models())
        base = load_storage().base_path(entry["local_path"])
        if not base.exists():
            print(
                f"error: base {base} missing. Needed download (not performed): {entry['hf_id']} @ "
                f"{entry['revision']} -> {base}"
            )
            return 2
    base = Path(base)
    tensors = local_tensors(base)
    config = json.loads((base / "config.json").read_text())
    projs = select_projections(
        discover_projections(tensors, config), cfg.get("families", ["ffn.gate", "ffn.up", "ffn.down"]), layers
    )
    if args.max_tensors:
        projs = projs[: args.max_tensors]
    shard_of = {t.name: t.shard for t in tensors}
    combos = list(itertools.product(projs, ratios, topos, ranks))
    plan = [{"tensor": p.tensor, "byte_ratio": r, "topology": t, "residual_rank": k} for p, r, t, k in combos]
    print(
        f"{len(projs)} tensors x {len(ratios)} ratios x {len(topos)} topologies x {len(ranks)} ranks "
        f"= {len(combos)} combos"
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    rows: list[dict] = []
    cache: dict[str, torch.Tensor] = {}
    for p, ratio, topo, rank in combos:
        if p.tensor not in cache:
            cache.clear()
            with safe_open(str(base / shard_of[p.tensor]), framework="pt") as f:
                cache[p.tensor] = f.get_tensor(p.tensor).to(torch.float32)
        W = cache[p.tensor]
        budget = int(ratio * p.out_features * p.in_features * DTYPE_BYTES.get(p.dtype, 2))
        spec_kwargs = {
            "topology": Topology(topo, topo_seed),
            "stages_per_rotor": cascade.get("stages_per_rotor", "auto"),
            "residual_rank": rank,
            "adapter": cascade.get("adapter", "pad_truncate"),
        }
        res = equal_budget_baselines(W, budget, storage_dtype, spec_kwargs, schedule, X=None, seed=seed)
        rows.append(
            {
                "tensor": p.tensor,
                "layer": p.layer,
                "family": p.family,
                "shape": [p.out_features, p.in_features],
                "byte_ratio": ratio,
                "byte_budget": budget,
                "topology": topo,
                "residual_rank": rank,
                "result": _clean(res),
            }
        )
        print(
            f"{p.tensor.split('layers.')[-1]:<28} r={ratio:<5} {topo:<10} k={rank}  "
            f"{json.dumps(_clean(res), default=str)[:160]}"
        )

    payload = {
        "config": cfg,
        "args": {k: _clean(v) for k, v in vars(args).items()},
        "storage_dtype": storage_dtype,
        "objective": "weight-space only (no activations, no LM eval)",
        "rows": rows,
    }
    name = args.name or f"{cfg.get('name', args.config.stem)}-sweep"
    print(f"wrote run record {write_run_record('sweep', name, payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
