"""Inspect a base checkpoint (local or header-only remote): tensor groups, layer types, projections."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rotor.models.discovery import local_tensors, remote_json, remote_tensors
from rotor.models.qwen3_5 import classify_tensor, discover_projections, layer_types
from rotor.reporting import ExperimentRun, markdown_table
from rotor.storage import load_storage, model_entry, sha256_file

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _nbytes(t) -> int:
    return _numel(t.shape) * DTYPE_BYTES.get(t.dtype, 0)


def build_report(tensors, config: dict) -> dict:
    groups: dict[str, dict] = defaultdict(lambda: {"tensors": 0, "params": 0, "bytes": 0})
    for t in tensors:
        g = groups[classify_tensor(t.name)]
        g["tensors"] += 1
        g["params"] += _numel(t.shape)
        g["bytes"] += _nbytes(t)
    types = layer_types(config)
    type_counts: dict[str, int] = defaultdict(int)
    for lt in types:
        type_counts[lt] += 1
    projs = discover_projections(tensors, config)
    fam: dict[tuple, dict] = {}
    for p in projs:
        key = (p.family, p.out_features, p.in_features, p.dtype)
        e = fam.setdefault(
            key,
            {"family": p.family, "shape": [p.out_features, p.in_features], "dtype": p.dtype, "count": 0, "bytes": 0},
        )
        e["count"] += 1
        e["bytes"] += p.out_features * p.in_features * DTYPE_BYTES.get(p.dtype, 0)
    names = {t.name for t in tensors}
    tied = bool(config.get("tie_word_embeddings", config.get("text_config", {}).get("tie_word_embeddings")))
    return {
        "groups": dict(groups),
        "total_bytes": sum(g["bytes"] for g in groups.values()),
        "total_params": sum(g["params"] for g in groups.values()),
        "num_layers": len(types),
        "layer_type_counts": dict(type_counts),
        "layer_types": types,
        "projections": sorted(fam.values(), key=lambda e: (e["family"], e["shape"])),
        "num_projections": len(projs),
        "tied_word_embeddings_config": tied,
        "lm_head_tensor_present": any(n.startswith("lm_head") for n in names),
        "architectures": config.get("architectures"),
    }


def print_report(rep: dict) -> None:
    print("\n== tensor groups ==")
    print(f"{'group':<14}{'tensors':>9}{'params':>16}{'bytes':>16}")
    for k, g in sorted(rep["groups"].items()):
        print(f"{k:<14}{g['tensors']:>9}{g['params']:>16,}{g['bytes']:>16,}")
    print(f"{'TOTAL':<14}{'':>9}{rep['total_params']:>16,}{rep['total_bytes']:>16,}")
    print(f"\n== layers: {rep['num_layers']} ==  {rep['layer_type_counts']}")
    print("\n== projections ==")
    print(f"{'family':<10}{'shape (out x in)':<20}{'dtype':<7}{'count':>6}{'bytes':>16}")
    for e in rep["projections"]:
        print(f"{e['family']:<10}{str(tuple(e['shape'])):<20}{e['dtype']:<7}{e['count']:>6}{e['bytes']:>16,}")
    print(
        f"\ntied word embeddings (config): {rep['tied_word_embeddings_config']}; "
        f"lm_head tensor present: {rep['lm_head_tensor_present']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="key or HF id from configs/models.yaml")
    ap.add_argument("--base-dir", type=Path, help="explicit local checkpoint directory")
    ap.add_argument("--remote", action="store_true", help="read config + safetensors headers from the Hub (no weights)")
    ap.add_argument("--checksums", action="store_true", help="sha256 local shards")
    ap.add_argument("--json", type=Path, dest="json_out", help="write full report JSON here")
    ap.add_argument("--record", action="store_true", help="write a run record (kind 'inspect')")
    args = ap.parse_args(argv)
    if not args.model and not args.base_dir:
        ap.error("one of --model or --base-dir is required")
    if args.remote and not args.model:
        ap.error("--remote requires --model")

    entry = None
    key = args.model or str(args.base_dir)
    extra: dict = {}
    if args.model:
        key, entry = model_entry(args.model)
        storage = load_storage()
        cfg_path = storage.base_path(entry["local_path"])
        exists = cfg_path.exists()
        extra.update(
            model_key=key,
            hf_id=entry["hf_id"],
            revision=entry["revision"],
            configured_base_path=str(cfg_path),
            configured_base_exists=exists,
        )
        print(f"configured base path: {cfg_path} (exists: {exists})")
        if not exists:
            print(
                f"  not downloaded. Required download (NOT performed by this tool): repo {entry['hf_id']} "
                f"@ revision {entry['revision']} -> {cfg_path}"
            )

    if args.remote:
        repo, rev = entry["hf_id"], entry["revision"]
        config = remote_json(repo, "config.json", rev)
        tensors = remote_tensors(repo, rev)
        extra["source"] = f"remote:{repo}@{rev} (headers only)"
    else:
        if args.base_dir:
            base = args.base_dir
        else:
            base = load_storage().base_path(entry["local_path"])
        if not Path(base).exists():
            print(f"error: {base} does not exist; use --remote for a header-only inspection.", file=sys.stderr)
            return 2
        config = json.loads((Path(base) / "config.json").read_text())
        tensors = local_tensors(Path(base))
        extra["source"] = f"local:{base}"
        if args.checksums:
            extra["shard_sha256"] = {s.name: sha256_file(s) for s in sorted(Path(base).glob("*.safetensors"))}
            for n, h in extra["shard_sha256"].items():
                print(f"sha256 {n}: {h}")

    rep = build_report(tensors, config)
    rep.update(extra)
    print(f"source: {rep['source']}")
    print_report(rep)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.json_out}")
    if args.record:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with ExperimentRun("inspect", model=str(key), method="inspect", target="na", config=config) as run:
            run.log(f"inspecting {rep['source']}")
            run.metrics.update({k: v for k, v in rep.items() if k not in ("projections", "groups")})
            run.metrics["groups"] = rep["groups"]
            run.metrics["projections"] = rep["projections"]
            run.add_section(
                "Tensor groups (header-derived, exact)",
                markdown_table([{"group": g, **v} for g, v in sorted(rep["groups"].items())]),
            )
            run.add_section(
                "FFN/attention projections (header-derived, exact)",
                markdown_table(
                    [{**e, "shape": "x".join(str(s) for s in e["shape"])} for e in rep["projections"]],
                    ["family", "shape", "dtype", "count", "bytes"],
                ),
            )
            run.summary = (
                f"{rep['total_params']:,} parameters / {rep['total_bytes']:,} bytes across {rep['num_layers']} layers; "
                f"source {rep['source']}."
            )
        print(f"wrote run record {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
