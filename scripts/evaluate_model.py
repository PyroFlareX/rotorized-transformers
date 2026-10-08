"""Evaluate next-token NLL of a base or converted model on identical token windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.evaluation.lm import evaluate_lm
from rotor.evaluation.longcontext import nll_by_position
from rotor.evaluation.records import write_run_record
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
from rotor.storage import load_storage, model_entry

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path)
    ap.add_argument("--model", help="key or HF id from configs/models.yaml")
    ap.add_argument("--ckpt", type=Path, help="derived checkpoint directory (rotor format)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", type=Path)
    src.add_argument("--random-tokens", action="store_true", help="synthetic tokens; smoke test only")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--long-context-buckets", help="comma-separated bucket edges, e.g. 0,512,2048")
    ap.add_argument("--compare-base", action="store_true", help="also evaluate the base model on the same windows")
    ap.add_argument("--name", help="run record name")
    args = ap.parse_args(argv)

    if not args.base_dir and not args.model and not args.ckpt:
        ap.error("need --base-dir, --model, or --ckpt (whose manifest names its base)")
    if args.base_dir:
        base_dir = args.base_dir
    elif args.model:
        base_dir = load_storage().base_path(model_entry(args.model)[1]["local_path"])
    else:
        base_dir = None  # load_converted resolves from the manifest
    if args.compare_base and not args.ckpt:
        ap.error("--compare-base requires --ckpt")
    dtype = DTYPES[args.dtype]

    if args.random_tokens:
        print(
            "WARNING: --random-tokens uses synthetic uniform tokens. Its NLL is NOT a quality benchmark; "
            "use it only for smoke tests."
        )

    if args.ckpt:
        model, manifest = load_converted(args.ckpt, dtype=dtype, device=args.device, base_dir=base_dir)
        base_dir = base_dir or Path(manifest.base.path)
    else:
        model = load_base(base_dir, dtype=dtype, device=args.device)
    model.eval()

    if args.random_tokens:
        windows = random_windows(model.config.vocab_size, args.seq_len, args.num_windows, args.seed)
    else:
        ids = tokenize_text_file(load_tokenizer(base_dir), args.text_file)
        windows = windows_from_ids(ids, args.seq_len, args.num_windows, args.seed)
    fp = windows_fingerprint(windows)
    print(f"windows: {tuple(windows.shape)} fingerprint {fp}")

    buckets = [int(x) for x in args.long_context_buckets.split(",")] if args.long_context_buckets else None
    result: dict = {"model": evaluate_lm(model, windows, batch_size=args.batch_size, device=args.device)}
    print(
        f"mean_nll {result['model']['mean_nll']:.6f}  ppl {result['model']['perplexity']:.4f}  "
        f"tokens {result['model']['tokens']}"
    )
    if buckets:
        result["nll_by_position"] = nll_by_position(model, windows, buckets, args.device)
        for r in result["nll_by_position"]:
            print(f"  [{r['start']},{r['end']}) nll {r['mean_nll']:.6f} tokens {r['tokens']}")

    if args.compare_base:
        del model
        base = load_base(base_dir, dtype=dtype, device=args.device)
        base.eval()
        result["base"] = evaluate_lm(base, windows, batch_size=args.batch_size, device=args.device)
        result["delta_nll"] = result["model"]["mean_nll"] - result["base"]["mean_nll"]
        print(
            f"base mean_nll {result['base']['mean_nll']:.6f}; delta NLL (converted - base) {result['delta_nll']:+.6f}"
        )

    payload = {
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "base_dir": str(base_dir),
        "synthetic_tokens": bool(args.random_tokens),
        "windows_fingerprint": fp,
        "windows_shape": list(windows.shape),
        "results": result,
    }
    if args.random_tokens:
        payload["warning"] = "synthetic random tokens; not a quality benchmark"
    name = args.name or (args.ckpt.name if args.ckpt else (args.model or Path(str(base_dir)).name)).replace("/", "_")
    print(f"wrote run record {write_run_record('evaluate', name, payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
