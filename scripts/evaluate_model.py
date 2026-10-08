#!/usr/bin/env python
"""Evaluate a base or converted (rotor-scaler) model on identical inputs and publish a report run directory.

Metrics (all measured; see configs/evaluation/README.md for the YAML schema accepted by --config):

* next-token NLL / perplexity on fixed token windows (+ optional per-position buckets),
* greedy generation with KV cache: time-to-first-token, prefill and decode tokens/s, peak memory, token ids,
* multiple-choice accuracy from a user-approved JSONL file (``--mc-jsonl``),
* SYNTHETIC passkey retrieval (``--passkey N --passkey-context T``).

With ``--compare-base`` every metric is computed for the converted model and for the base model on exactly the
same windows / prompt / items, and the report contains a side-by-side table with deltas. Longer-context work
(passkey, position buckets, or any context above ``--max-context``) is refused unless ``--short-context-verified``
asserts that the short-context path already matches its reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

from rotor.data import random_windows, tokenize_text_file, windows_fingerprint, windows_from_ids
from rotor.evaluation.generation import compare_generations, greedy_decode, measure_generation
from rotor.evaluation.lm import evaluate_lm
from rotor.evaluation.longcontext import nll_by_position
from rotor.evaluation.retrieval import SYNTHETIC_LABEL, SmokeTokenizer, passkey_accuracy, passkey_items
from rotor.evaluation.tasks import load_jsonl_items, multiple_choice_accuracy
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
from rotor.reporting import ExperimentRun, markdown_table
from rotor.storage import load_storage, model_entry, sha256_file

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PATH_KEYS = ("base_dir", "ckpt", "text_file", "mc_jsonl")
SPEED_NOTE = (
    "Speed numbers are measured on this host with the PyTorch reference implementation (converted layers run the "
    "sequential reference RotorLinear, not an optimized kernel; no dense reconstruction is timed as an "
    "optimized path). They are not a statement about what a fused kernel could achieve."
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML with any of the options below (CLI flags override it)")
    ap.add_argument("--base-dir", type=Path)
    ap.add_argument("--model", help="key or HF id from configs/models.yaml")
    ap.add_argument("--ckpt", type=Path, help="derived checkpoint directory (rotor format)")
    ap.add_argument("--text-file", type=Path, help="user-approved evaluation corpus (plain text)")
    ap.add_argument("--random-tokens", action="store_true", help="synthetic tokens; smoke test only")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--long-context-buckets", help="comma-separated bucket edges, e.g. 0,512,2048")
    ap.add_argument("--generation-prompt-tokens", type=int, default=0, help="prompt = first N tokens of window 0")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="greedy tokens to generate (with the above)")
    ap.add_argument("--mc-jsonl", type=Path, help="user-approved multiple-choice JSONL (context/choices/answer)")
    ap.add_argument("--passkey", type=int, default=0, help="number of SYNTHETIC passkey-retrieval items")
    ap.add_argument("--passkey-context", type=int, default=0, help="tokens per passkey prompt")
    ap.add_argument(
        "--max-context",
        type=int,
        default=2048,
        help="largest context (tokens) allowed without --short-context-verified (default 2048)",
    )
    ap.add_argument(
        "--short-context-verified",
        action="store_true",
        help="assert the short-context path matched its reference; required for passkey, buckets, long contexts",
    )
    ap.add_argument("--compare-base", action="store_true", help="evaluate the base model on identical inputs too")
    ap.add_argument("--name", help="label recorded in the run config")
    return ap


def apply_config(ap: argparse.ArgumentParser, argv: list[str]) -> None:
    """Load ``--config`` (if given) into the parser's defaults, validating keys; CLI flags still override."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _rest = pre.parse_known_args(argv)
    if known.config is None:
        return
    cfg = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        ap.error(f"{known.config}: top level must be a mapping")
    dests = {a.dest for a in ap._actions if a.dest not in ("help", "config")}
    cfg = {str(k).replace("-", "_"): v for k, v in cfg.items()}
    unknown = sorted(set(cfg) - dests)
    if unknown:
        ap.error(f"{known.config}: unknown keys {unknown}; valid keys: {sorted(dests)}")
    for key in PATH_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])
    buckets = cfg.get("long_context_buckets")
    if isinstance(buckets, list):
        cfg["long_context_buckets"] = ",".join(str(int(b)) for b in buckets)
    ap.set_defaults(**{k: v for k, v in cfg.items() if v is not None})


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    ap = build_parser()
    apply_config(ap, argv)
    args = ap.parse_args(argv)
    if not args.base_dir and not args.model and not args.ckpt:
        ap.error("need --base-dir, --model, or --ckpt (whose manifest names its base)")
    if bool(args.text_file) == bool(args.random_tokens):
        ap.error(
            "need exactly one data source: --text-file (a corpus the user has approved) or --random-tokens "
            "(synthetic smoke test)"
        )
    if args.compare_base and not args.ckpt:
        ap.error("--compare-base requires --ckpt")
    if args.seq_len < 2 or args.num_windows < 1 or args.max_new_tokens < 1:
        ap.error("--seq-len must be >= 2; --num-windows and --max-new-tokens must be >= 1")
    if args.generation_prompt_tokens < 0 or args.generation_prompt_tokens > args.seq_len:
        ap.error("--generation-prompt-tokens must be in [0, seq-len] (the prompt is the start of the first window)")
    if (args.passkey > 0) != (args.passkey_context > 0):
        ap.error("--passkey N and --passkey-context T must be given together")
    # Spec: longer-context benchmarks only after the short-context path matches its reference.
    contexts = {"seq_len": args.seq_len, "passkey_context": args.passkey_context}
    if args.generation_prompt_tokens:
        contexts["generation_prompt+new"] = args.generation_prompt_tokens + args.max_new_tokens
    gated = []
    if args.passkey:
        gated.append("passkey retrieval")
    if args.long_context_buckets:
        gated.append("position buckets")
    gated += [f"{k}={v} > --max-context {args.max_context}" for k, v in contexts.items() if v > args.max_context]
    if gated and not args.short_context_verified:
        ap.error(
            f"refusing {', '.join(gated)} without --short-context-verified: longer-context evaluation is allowed "
            "only after the short-context path has matched its reference (run perplexity/generation first, "
            "confirm, then pass the flag)"
        )
    return args, ap


def tokenizer_info(tok, tok_dir: Path | None, synthetic: bool) -> dict:
    tj = Path(tok_dir) / "tokenizer.json" if tok_dir else None
    return {
        "name": getattr(tok, "name_or_path", None) or getattr(tok, "name", None) or type(tok).__name__,
        "class": type(tok).__name__,
        "tokenizer_json_sha256": sha256_file(tj) if tj and tj.is_file() else None,
        "synthetic_smoke_tokenizer": synthetic,
    }


def make_tokenizer(base_dir: Path, vocab_size: int, allow_smoke: bool):
    """Real tokenizer from the model dir; the hashed smoke tokenizer only for --random-tokens runs without one."""
    has_files = any((base_dir / n).is_file() for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json"))
    if has_files:
        return load_tokenizer(base_dir), False
    if allow_smoke:
        return SmokeTokenizer(vocab_size), True
    raise FileNotFoundError(f"no tokenizer files in {base_dir}; a tokenizer is required for this evaluation")


def evaluate_one(model, inputs: dict, args, log) -> dict:
    """Every requested metric for one model on the shared ``inputs``."""
    out: dict = {}
    lm = evaluate_lm(model, inputs["windows"], batch_size=args.batch_size, device=args.device)
    out["lm"] = lm
    log(f"  mean_nll {lm['mean_nll']:.6f}  ppl {lm['perplexity']:.4f}  tokens {lm['tokens']}")
    if inputs["buckets"]:
        out["nll_by_position"] = nll_by_position(model, inputs["windows"], inputs["buckets"], args.device)
        for r in out["nll_by_position"]:
            log(f"  [{r['start']},{r['end']}) nll {r['mean_nll']:.6f} tokens {r['tokens']}")
    if inputs["prompt"] is not None:
        greedy_decode(model, inputs["prompt"], min(2, args.max_new_tokens), args.device)  # untimed warm-up
        gen = measure_generation(model, inputs["prompt"], args.max_new_tokens, args.device)
        out["generation"] = gen
        log(
            f"  generation: ttft {gen['ttft_seconds'] * 1e3:.1f} ms, prefill {gen['prefill_tokens_per_second']} tok/s, "
            f"decode {gen['decode_tokens_per_second']} tok/s"
        )
    if inputs["mc_items"] is not None:
        mc = multiple_choice_accuracy(model, inputs["tokenizer"], inputs["mc_items"], args.device)
        out["multiple_choice"] = {
            "accuracy": mc["accuracy"],
            "accuracy_norm": mc["accuracy_norm"],
            "n": mc["n"],
            "pred": [r["pred"] for r in mc["per_item"]],
            "pred_norm": [r["pred_norm"] for r in mc["per_item"]],
            "answers": [r["answer"] for r in mc["per_item"]],
        }
        log(f"  multiple choice: acc {mc['accuracy']:.4f}, acc_norm {mc['accuracy_norm']:.4f} (n={mc['n']})")
    if inputs["passkey_items"] is not None:
        pk = passkey_accuracy(model, inputs["passkey_items"], args.device)
        out["passkey"] = {
            "accuracy": pk["accuracy"],
            "n": pk["n"],
            "context_tokens": pk["context_tokens"],
            "by_depth": {f"{r['depth']:.3f}": r["correct"] for r in pk["by_depth"]},
            "predicted_ids_by_depth": {f"{r['depth']:.3f}": r["predicted_ids"] for r in pk["by_depth"]},
            "synthetic": True,
            "label": SYNTHETIC_LABEL,
        }
        log(f"  passkey (synthetic): acc {pk['accuracy']:.4f} (n={pk['n']}, ctx={pk['context_tokens']})")
    return out


def _scalars(res: dict) -> dict[str, float | None]:
    """Comparable scalar metrics of one evaluate_one result, keyed by display name."""
    s: dict[str, float | None] = {"mean_nll": res["lm"]["mean_nll"], "perplexity": res["lm"]["perplexity"]}
    for r in res.get("nll_by_position", []):
        s[f"nll[{r['start']},{r['end']})"] = r["mean_nll"]
    if "generation" in res:
        g = res["generation"]
        s["ttft_seconds"] = g["ttft_seconds"]
        s["prefill_tokens_per_second"] = g["prefill_tokens_per_second"]
        s["decode_tokens_per_second"] = g["decode_tokens_per_second"]
        s["gen_peak_rss_bytes"] = g["resources"]["peak_rss_bytes"]
        s["gen_peak_device_bytes"] = g["resources"]["peak_device_bytes"]
    if "multiple_choice" in res:
        s["mc_accuracy"] = res["multiple_choice"]["accuracy"]
        s["mc_accuracy_norm"] = res["multiple_choice"]["accuracy_norm"]
    if "passkey" in res:
        s["passkey_accuracy_synthetic"] = res["passkey"]["accuracy"]
    return s


def side_by_side(base: dict, conv: dict) -> list[dict]:
    b, c = _scalars(base), _scalars(conv)
    rows = []
    for key in b:
        bv, cv = b[key], c.get(key)
        row = {"metric": key, "base": bv, "converted": cv, "delta (conv - base)": None, "ratio (conv / base)": None}
        if isinstance(bv, int | float) and isinstance(cv, int | float):
            row["delta (conv - base)"] = cv - bv
            row["ratio (conv / base)"] = cv / bv if bv else None
        rows.append(row)
    return rows


def comparison_metrics(base: dict, conv: dict) -> dict:
    rows = side_by_side(base, conv)
    cmp: dict = {"table": {r["metric"]: {k: v for k, v in r.items() if k != "metric"} for r in rows}}
    cmp["delta_nll"] = conv["lm"]["mean_nll"] - base["lm"]["mean_nll"]
    if "generation" in base:
        cmp["generation_match"] = compare_generations(
            base["generation"]["generated_ids"], conv["generation"]["generated_ids"]
        )
    return cmp


def target_from_manifest(manifest) -> str:
    budget = (manifest.conversion_config or {}).get("budget") or {}
    if budget.get("target"):
        return str(budget["target"])
    ratio = budget.get("byte_ratio")
    if ratio is not None:
        return f"b{round(float(ratio) * 100):03d}"
    return "na"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, _ap = parse_args(argv)
    dtype = DTYPES[args.dtype]

    if args.base_dir:
        base_dir = args.base_dir
    elif args.model:
        base_dir = load_storage().base_path(model_entry(args.model)[1]["local_path"])
    else:
        base_dir = None  # load_converted resolves from the manifest

    manifest = None
    if args.ckpt:
        from rotor.formats import read_manifest

        manifest = read_manifest(args.ckpt)
        base_dir = base_dir or Path(manifest.base.path)
    method = f"rotor:{args.ckpt.name}" if args.ckpt else "base"
    target = target_from_manifest(manifest) if manifest else "na"
    model_label = args.model or Path(str(base_dir)).name
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    with ExperimentRun("evaluate", model=model_label, method=method, target=target, config=config) as run:
        log = run.log
        if args.random_tokens:
            log(
                "WARNING: --random-tokens uses synthetic uniform tokens. Its NLL is NOT a quality benchmark; "
                "use it only for smoke tests."
            )
        if args.ckpt:
            model, _ = load_converted(args.ckpt, dtype=dtype, device=args.device, base_dir=base_dir)
        else:
            model = load_base(base_dir, dtype=dtype, device=args.device)
        model.eval()
        vocab = int(model.config.vocab_size)

        tokenizer, tok_synthetic = (None, False)
        need_tok = not args.random_tokens or args.mc_jsonl or args.passkey
        if need_tok:
            tokenizer, tok_synthetic = make_tokenizer(base_dir, vocab, allow_smoke=args.random_tokens)
        if args.random_tokens:
            windows = random_windows(vocab, args.seq_len, args.num_windows, args.seed)
        else:
            windows = windows_from_ids(
                tokenize_text_file(tokenizer, args.text_file), args.seq_len, args.num_windows, args.seed
            )
        fp = windows_fingerprint(windows)
        log(f"windows: {tuple(windows.shape)} fingerprint {fp}")

        mc_items = load_jsonl_items(args.mc_jsonl) if args.mc_jsonl else None
        pk_items = passkey_items(tokenizer, args.passkey, args.passkey_context, args.seed) if args.passkey else None
        inputs = {
            "windows": windows,
            "buckets": [int(x) for x in args.long_context_buckets.split(",")] if args.long_context_buckets else None,
            "prompt": windows[:1, : args.generation_prompt_tokens] if args.generation_prompt_tokens else None,
            "mc_items": mc_items,
            "passkey_items": pk_items,
            "tokenizer": tokenizer,
        }
        input_info = {
            "synthetic_tokens": bool(args.random_tokens),
            "windows_fingerprint": fp,
            "windows_shape": list(windows.shape),
            "text_file": str(args.text_file) if args.text_file else None,
            "text_file_sha256": sha256_file(args.text_file) if args.text_file else None,
            "tokenizer": tokenizer_info(tokenizer, base_dir, tok_synthetic) if tokenizer is not None else None,
            "generation_prompt_tokens": args.generation_prompt_tokens,
            "max_new_tokens": args.max_new_tokens if args.generation_prompt_tokens else None,
            "mc_jsonl": str(args.mc_jsonl) if args.mc_jsonl else None,
            "mc_jsonl_sha256": sha256_file(args.mc_jsonl) if args.mc_jsonl else None,
            "passkey_items": args.passkey,
            "passkey_context_tokens": args.passkey_context,
            "passkey_synthetic": bool(args.passkey),
            "short_context_verified": bool(args.short_context_verified),
        }
        run.metrics["inputs"] = input_info
        results: dict[str, dict] = {}

        label = "converted" if args.ckpt else "base"
        log(f"evaluating {label} ({method})")
        results[label] = evaluate_one(model, inputs, args, log)

        if args.compare_base:
            del model
            gc.collect()
            log("evaluating base on identical inputs")
            base = load_base(base_dir, dtype=dtype, device=args.device)
            base.eval()
            results["base"] = evaluate_one(base, inputs, args, log)
            del base
        run.metrics["results"] = results
        if args.compare_base:
            run.metrics["comparison"] = comparison_metrics(results["base"], results["converted"])
            rows = side_by_side(results["base"], results["converted"])
            run.add_section("Side-by-side (identical inputs)", markdown_table(rows))
            gm = run.metrics["comparison"].get("generation_match")
            if gm:
                run.add_section(
                    "Generation match (greedy, same prompt)",
                    f"exact_match={gm['exact_match']}; common prefix {gm['match_prefix_len']} of "
                    f"{gm['len_a']} (base) / {gm['len_b']} (converted) tokens.",
                )
            run.summary = (
                f"Converted `{args.ckpt.name}` vs base on identical inputs: delta NLL "
                f"{run.metrics['comparison']['delta_nll']:+.6f} (converted - base)."
            )
        else:
            run.summary = f"{label} model: mean NLL {results[label]['lm']['mean_nll']:.6f}."
        notes = [SPEED_NOTE] if "generation" in results[label] else []
        if args.random_tokens:
            notes.append("Inputs are synthetic random tokens: NLL/perplexity and generation say nothing about quality.")
        if args.passkey:
            notes.append(f"Passkey retrieval is {SYNTHETIC_LABEL}.")
        if args.short_context_verified:
            notes.append(
                "`--short-context-verified` was asserted by the operator; this tool does not verify it itself."
            )
        run.interpretation = " ".join(notes) or "No interpretation beyond the measured values above."
        log(json.dumps({"summary": run.summary}))
    print(f"published run {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
