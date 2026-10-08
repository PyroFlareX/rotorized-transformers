#!/usr/bin/env python
"""Per-matrix equal-stored-byte comparison: truncated SVD vs SVD->Givens vs rotor fits, plus random controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from rotor.evaluation.baselines import equal_budget_baselines
from rotor.evaluation.records import write_run_record
from rotor.formats import bytes_per_param
from rotor.storage import load_models, load_storage, model_entry
from rotor.training.layerwise import FitSchedule

REMINDER = (
    "REMINDER: gains from truncated-SVD-derived methods are NOT rotor-specific. Only the rotor_* rows trained "
    "from scratch at the same stored bytes, compared against the svd row and the random controls, bear on the "
    "rotor hypothesis. est_flops are formula ESTIMATES, not measurements."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-dir", type=Path, help="local base checkpoint dir")
    src.add_argument("--model", help="key in configs/models.yaml (resolved under storage.base_root)")
    src.add_argument("--synthetic", metavar="IN,OUT", help="seeded Gaussian matrix with decaying spectrum (smoke)")
    p.add_argument("--tensors", default=r"layers\.0\.mlp\.(gate|up|down)_proj\.weight$", help="regex on tensor names")
    p.add_argument("--max-tensors", type=int, default=3)
    p.add_argument("--byte-ratio", type=float, default=0.25)
    p.add_argument("--storage-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--topology", default="butterfly", choices=["brick", "butterfly", "seeded"])
    p.add_argument("--topology-seed", type=int, default=0)
    p.add_argument("--stages-per-rotor", default="auto")
    p.add_argument("--residual-rank", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr-angles", type=float, default=1e-2)
    p.add_argument("--lr-scales", type=float, default=1e-2)
    p.add_argument("--lr-residual", type=float, default=1e-3)
    p.add_argument("--with-activations", action="store_true")
    p.add_argument("--text-file", type=Path, help="calibration text (default: random tokens)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-windows", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    return p


def _rows(node, label=""):
    """Flatten the baselines result into (label, row) pairs; rows have 'params' or 'skipped'."""
    if isinstance(node, dict):
        if "params" in node or "skipped" in node:
            yield label, node
        else:
            for k, v in node.items():
                yield from _rows(v, f"{label}/{k}" if label else str(k))


def _get(d: dict | None, key: str):
    return None if not isinstance(d, dict) else d.get(key)


def _fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def print_table(title: str, result: dict) -> None:
    print(f"\n== {title} (byte budget {result.get('budget')})")
    print(f"{'method':<34}{'params':>10}{'bytes':>10}{'rel_fro':>10}{'act_rel_fro':>13}{'est_flops':>12}")
    for section in ("methods", "controls"):
        if section == "controls":
            print("-- controls (random matrices with same spectrum / norm; NOT approximations of W) --")
        for label, r in _rows(result.get(section, {})):
            if "skipped" in r:
                print(f"{label:<34}  skipped: {r['skipped']}")
                continue
            flops = r.get("estimated_flops_per_token")
            print(
                f"{label:<34}{r.get('params', '-'):>10}{r.get('stored_bytes', '-'):>10}"
                f"{_fmt(_get(r.get('weight_errors'), 'rel_fro')):>10}"
                f"{_fmt(_get(r.get('activation_errors'), 'rel_fro')):>13}"
                f"{('-' if flops is None else flops):>12}"
            )
    for n in result.get("notes", []):
        print(f"note: {n}")


def _synthetic(shape: str, seed: int) -> torch.Tensor:
    n_in, n_out = (int(v) for v in shape.split(","))
    g = torch.Generator().manual_seed(seed)
    k = min(n_in, n_out)
    U = torch.linalg.qr(torch.randn(n_out, k, generator=g))[0]
    V = torch.linalg.qr(torch.randn(n_in, k, generator=g))[0]
    S = torch.arange(1, k + 1, dtype=torch.float32).pow(-0.7)
    return (U * S) @ V.T


def _load_base_matrices(args, storage) -> tuple[Path, list[tuple[str, str, torch.Tensor]]]:
    from safetensors import safe_open

    if args.model:
        _, entry = model_entry(args.model, load_models())
        base = storage.base_path(entry["local_path"])
    else:
        base = args.base_dir
    if not Path(base).is_dir():
        raise SystemExit(f"base dir {base} does not exist; this tool never downloads weights")
    from rotor.models.discovery import local_tensors

    pat = re.compile(args.tensors)
    infos = [t for t in local_tensors(base) if len(t.shape) == 2 and pat.search(t.name)][: args.max_tensors]
    if not infos:
        raise SystemExit(f"no 2-D tensors match {args.tensors!r} in {base}")
    out = []
    for t in infos:
        with safe_open(str(Path(base) / t.shard), "pt") as f:
            W = f.get_tensor(t.name).float()
        out.append((t.name, t.name, W))
    return Path(base), out


def _module_path(tensor: str) -> str:
    return re.sub(r"^model\.language_model\.", "model.", tensor).removesuffix(".weight")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = load_storage()
    spr = args.stages_per_rotor if args.stages_per_rotor == "auto" else int(args.stages_per_rotor)
    base: Path | None = None
    if args.synthetic:
        items = [(f"synthetic[{args.synthetic}]", None, _synthetic(args.synthetic, args.seed))]
        if args.with_activations:
            print("note: --with-activations ignored for --synthetic (no model); using Gaussian inputs")
    else:
        base, items = _load_base_matrices(args, storage)

    inputs: dict[str, torch.Tensor] = {}
    if args.with_activations and not args.synthetic:
        from rotor.converters.calibration import collect_inputs
        from rotor.data import random_windows, tokenize_text_file, windows_from_ids
        from rotor.models.loading import load_base
        from rotor.models.qwen3_5 import load_tokenizer

        model = load_base(base, dtype=torch.float32, device="cpu")
        if args.text_file:
            ids = tokenize_text_file(load_tokenizer(base), args.text_file)
            windows = windows_from_ids(ids, args.seq_len, args.num_windows, args.seed)
        else:
            print("note: calibration inputs from random tokens (smoke only)")
            windows = random_windows(model.config.vocab_size, args.seq_len, args.num_windows, args.seed)
        mods = [_module_path(t) for t, _, _ in items]
        inputs = collect_inputs(model, mods, windows, args.max_rows, seed=args.seed)
        del model

    schedule = FitSchedule.from_dict(
        {
            "phases": [
                {"train": ["scales"], "steps": max(1, args.steps // 10)},
                {"train": ["angles", "scales", "residual"], "steps": max(1, args.steps - args.steps // 10)},
            ],
            "lrs": {"angles": args.lr_angles, "scales": args.lr_scales, "residual": args.lr_residual},
        }
    )
    spec_kwargs = {
        "topology": __import__("rotor.math", fromlist=["Topology"]).Topology(args.topology, args.topology_seed),
        "stages_per_rotor": spr,
        "residual_rank": args.residual_rank,
    }
    dense_bpp = bytes_per_param("bfloat16")  # dense bytes are counted in the checkpoint dtype (BF16)
    results = {}
    for name, tensor, W in items:
        X = None
        if args.with_activations:
            if tensor is not None:
                X = inputs[_module_path(tensor)]
            else:
                g = torch.Generator().manual_seed(args.seed + 1)
                X = torch.randn(512, W.shape[1], generator=g)
        # equal-byte budget: fraction of the dense matrix stored in the chosen storage dtype
        budget = int(args.byte_ratio * W.numel() * dense_bpp)
        res = equal_budget_baselines(W, budget, args.storage_dtype, spec_kwargs, schedule, X=X, seed=args.seed)
        results[name] = res
        print_table(f"{name} shape={tuple(W.shape)} byte_ratio={args.byte_ratio}", res)
    print("\n" + REMINDER)

    rec = write_run_record(
        "baselines",
        args.name or (args.model or ("synthetic" if args.synthetic else Path(base).name)),
        {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "schedule": {"phases": schedule.phases, "lrs": schedule.lrs},
            "results": json.loads(json.dumps(results, default=str)),
            "notes": [REMINDER],
        },
    )
    print(f"run record {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
