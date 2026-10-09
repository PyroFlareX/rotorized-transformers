"""Curated, raw-prompt greedy generation runs with progressive matched-output records."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch

from rotor.evaluation.generation import compare_generations, measure_generation
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
from rotor.reporting import ExperimentRun
from rotor.storage import REPO_ROOT, load_storage, model_entry, sha256_file

PROMPTS_PATH = REPO_ROOT / "configs" / "evaluation" / "prompts-256.jsonl"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict[str, str]]:
    """Read and validate the curated JSONL prompts; each row has unique id, domain, and raw prompt text."""
    path = Path(path)
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or set(row) != {"id", "domain", "prompt"}:
            raise ValueError(f"{path}:{line_number}: expected exactly id, domain, and prompt fields")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("id", "domain", "prompt")):
            raise ValueError(f"{path}:{line_number}: id, domain, and prompt must be nonempty strings")
        if row["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate prompt id {row['id']!r}")
        if row["prompt"] in seen_prompts:
            raise ValueError(f"{path}:{line_number}: duplicate prompt text for {row['id']!r}")
        seen_ids.add(row["id"])
        seen_prompts.add(row["prompt"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prompts found")
    if path.resolve() == PROMPTS_PATH.resolve() and len(rows) != 256:
        raise ValueError(f"curated prompt corpus must contain exactly 256 rows; got {len(rows)}")
    return rows


def _token_hash(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_provenance(base_dir: Path) -> dict:
    found = {name: sha256_file(base_dir / name) for name in TOKENIZER_FILES if (base_dir / name).is_file()}
    if not found:
        raise FileNotFoundError(
            f"no local tokenizer files found in {base_dir}; refusing remote tokenizer lookup"
        )
    encoded = json.dumps(found, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": str(base_dir),
        "files_sha256": found,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _read_baseline(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"baseline {path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"baseline {path}:{line_number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"baseline results file is empty: {path}")
    return rows


def _validate_baseline(path: Path, expected: list[dict], settings: dict) -> list[dict]:
    baseline = _read_baseline(path)
    if len(baseline) != len(expected):
        raise ValueError(f"baseline has {len(baseline)} rows, but this run has {len(expected)} selected prompts")
    for index, (actual, current) in enumerate(zip(baseline, expected, strict=True), 1):
        prompt_id = current["prompt_id"]
        if actual.get("prompt_id") != prompt_id:
            raise ValueError(f"baseline row {index} prompt id mismatch: {actual.get('prompt_id')!r} != {prompt_id!r}")
        if actual.get("settings") != settings:
            raise ValueError(f"baseline prompt {prompt_id}: prompt dataset or generation settings do not match")
        if actual.get("domain") != current["domain"]:
            raise ValueError(f"baseline prompt {prompt_id}: domain does not match")
        for key in ("input_text", "input_ids", "input_token_count", "input_tokens_sha256"):
            if actual.get(key) != current[key]:
                raise ValueError(f"baseline prompt {prompt_id}: {key} does not match")
        if not isinstance(actual.get("output_ids"), list):
            raise ValueError(f"baseline prompt {prompt_id}: missing output_ids")
        if actual.get("output_token_count") != len(actual["output_ids"]):
            raise ValueError(f"baseline prompt {prompt_id}: output token count does not match output IDs")
        if actual.get("output_tokens_sha256") != _token_hash(actual["output_ids"]):
            raise ValueError(f"baseline prompt {prompt_id}: output token SHA-256 does not match output IDs")
    return baseline


def _resolve_model_paths(args) -> tuple[Path, Path | None, dict | None, str, str, str | None, str | None]:
    """Resolve local base/converted paths and catalog provenance without downloading anything."""
    manifest = None
    model_key = args.model or "local-model"
    hf_id = revision = None
    configured_base = None
    if args.model:
        model_key, entry = model_entry(args.model)
        configured_base = load_storage().base_path(entry["local_path"])
        hf_id, revision = entry.get("hf_id"), entry.get("revision")
    if args.base_dir:
        base_dir = args.base_dir.expanduser().resolve()
    elif configured_base is not None:
        base_dir = configured_base
    else:
        base_dir = None
    ckpt = args.ckpt.expanduser().resolve() if args.ckpt else None
    if ckpt is not None:
        from rotor.formats import read_manifest

        manifest = read_manifest(ckpt)
        if base_dir is None:
            base_dir = Path(manifest.base.path).expanduser().resolve()
        hf_id = hf_id or manifest.base.hf_id
        revision = revision or manifest.base.revision
        model_key = args.model or (hf_id.rsplit("/", 1)[-1] if hf_id else base_dir.name)
    if base_dir is None:
        raise ValueError("need --base-dir, --model, or --ckpt")
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"local base model directory does not exist: {base_dir}; no model download is attempted"
        )
    config_path = base_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing local model config: {config_path}; no model download is attempted")
    return base_dir, ckpt, manifest, model_key, str(config_path), hf_id, revision


def run_prompt_evaluation(args) -> Path:
    """Run each selected raw prompt once and publish JSONL rows plus ExperimentRun evidence."""
    prompts_path = args.prompts.expanduser().resolve()
    prompts = load_prompts(prompts_path)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        prompts = prompts[: args.limit]
    dataset_sha256 = sha256_file(prompts_path)
    base_dir, ckpt, manifest, model_key, config_path, hf_id, revision = _resolve_model_paths(args)
    config_path = Path(config_path)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config_sha256 = sha256_file(config_path)
    tokenizer_info = _tokenizer_provenance(base_dir)
    dtype = DTYPES[args.dtype]
    settings = {
        "prompt_dataset_sha256": dataset_sha256,
        "base_model": {
            "hf_id": hf_id,
            "revision": revision,
            "config_sha256": model_config_sha256,
            "model_type": model_config.get("model_type"),
        },
        "tokenizer_sha256": tokenizer_info["sha256"],
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"strategy": "greedy", "do_sample": False, "chat_template": False},
    }
    prepared = [
        {"prompt_id": prompt["id"], "domain": prompt["domain"], "input_text": prompt["prompt"]}
        for prompt in prompts
    ]

    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_label = args.name or model_key
    method = f"rotor:{ckpt.name}" if ckpt else "base"
    target = "na"
    if manifest is not None:
        budget = (manifest.conversion_config or {}).get("budget") or {}
        ratio = budget.get("byte_ratio")
        target = str(budget.get("target") or (f"b{round(float(ratio) * 100):03d}" if ratio else "na"))
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"base_dir_resolved": str(base_dir), "checkpoint_resolved": str(ckpt) if ckpt else None})
    config.update({"settings": settings, "prompt_dataset_sha256": dataset_sha256})
    with ExperimentRun(
        "prompt-generation",
        model=model_label,
        method=method,
        target=target,
        config=config,
        title=f"Curated prompt generation: {model_label} / {method}",
    ) as run:
        output_path = args.output.expanduser().resolve() if args.output else (
            run.out_root.parent / "prompt-evaluations" / f"{run.run_id}.jsonl"
        )
        telemetry_path = args.telemetry_path.expanduser().resolve() if args.telemetry_path else output_path.with_suffix(
            ".telemetry.jsonl"
        )
        if output_path == telemetry_path:
            raise ValueError("results JSONL and telemetry JSONL paths must be distinct")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() or telemetry_path.exists():
            raise FileExistsError(
                f"result or telemetry path already exists: {output_path if output_path.exists() else telemetry_path}"
            )
        run.add_log_reference(str(output_path))
        run.add_log_reference(str(telemetry_path))
        run.metrics["provenance"] = {
            "prompt_dataset": str(prompts_path),
            "prompt_dataset_sha256": dataset_sha256,
            "prompt_count": len(load_prompts(prompts_path)),
            "evaluated_prompt_count": len(prompts),
            "tokenizer": tokenizer_info,
            "model_config_sha256": model_config_sha256,
            "base_model_hf_id": hf_id,
            "base_model_revision": revision,
            "base_model_path": str(base_dir),
            "checkpoint_path": str(ckpt) if ckpt else None,
            "seed": seed,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "run_note": args.run_note,
        }
        run.metrics["settings"] = settings
        run.log(f"dataset: {len(prompts)} prompts; sha256 {dataset_sha256}")
        if args.device == "cpu":
            run.log("CPU slow fallback selected; timings are diagnostic, not GPU performance measurements.")
        if args.run_note:
            run.log(f"Run note: {args.run_note}")

        from rotor.telemetry import RuntimeTelemetry

        comparisons = []
        total_measured_seconds = 0.0
        with output_path.open("x", encoding="utf-8") as results_file:
            with RuntimeTelemetry(telemetry_path) as telemetry:
                telemetry.set_phase("tokenizer_loading")
                tokenizer = load_tokenizer(base_dir)
                telemetry.set_phase("tokenization")
                for item, prompt in zip(prepared, prompts, strict=True):
                    ids = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
                    if not ids:
                        raise ValueError(f"tokenizer produced no input tokens for {prompt['id']}")
                    item["input_ids"] = [int(token_id) for token_id in ids]
                    item["input_tokens_sha256"] = _token_hash(item["input_ids"])
                    item["input_token_count"] = len(ids)
                if args.compare_to:
                    baseline = _validate_baseline(Path(args.compare_to).expanduser().resolve(), prepared, settings)
                else:
                    baseline = None
                telemetry.set_phase("model_loading")
                model = (
                    load_converted(ckpt, dtype=dtype, device=args.device, base_dir=base_dir)[0]
                    if ckpt
                    else load_base(base_dir, dtype=dtype, device=args.device)
                )
                model.eval()
                for index, item in enumerate(prepared):
                    telemetry.set_phase("generation", request_id=item["prompt_id"])
                    prompt_ids = torch.tensor([item["input_ids"]], dtype=torch.long)
                    measured = measure_generation(model, prompt_ids, args.max_new_tokens, args.device)
                    output_ids = measured.pop("generated_ids")
                    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                    comparison = compare_generations(baseline[index]["output_ids"], output_ids) if baseline else None
                    record = {
                        "prompt_id": item["prompt_id"],
                        "domain": item["domain"],
                        "settings": settings,
                        "input_text": item["input_text"],
                        "input_ids": item["input_ids"],
                        "input_token_count": item["input_token_count"],
                        "input_tokens_sha256": item["input_tokens_sha256"],
                        "output_text": output_text,
                        "output_ids": output_ids,
                        "output_token_count": len(output_ids),
                        "output_tokens_sha256": _token_hash(output_ids),
                        "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                        "total_tokens": item["input_token_count"] + len(output_ids),
                        "timing_condition": "cold_first_generation" if index == 0 else "warm_model_resident",
                        "measure_generation": measured,
                        "comparison": comparison,
                    }
                    results_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    results_file.flush()
                    os.fsync(results_file.fileno())
                    total_measured_seconds += float(measured["total_seconds"])
                    if comparison is not None:
                        comparisons.append(comparison)
                    run.log(
                        f"[{index + 1}/{len(prepared)}] {item['prompt_id']}: {len(output_ids)} tokens, "
                        f"{measured['total_seconds']:.3f}s ({record['timing_condition']})"
                    )
            telemetry_summary = telemetry.as_dict()
        run.metrics["results"] = {
            "jsonl_path": str(output_path),
            "jsonl_rows": len(prepared),
            "telemetry_jsonl_path": str(telemetry_path),
            "runtime_telemetry": telemetry_summary,
            "generation_total_seconds": total_measured_seconds,
            "timing_note": "No generation warm-up: first measured prompt is cold; later prompts use a resident model.",
            "comparison": {
                "baseline_jsonl": str(Path(args.compare_to).expanduser().resolve()) if args.compare_to else None,
                "exact_matches": sum(int(row["exact_match"]) for row in comparisons),
                "prompts": len(comparisons),
            },
        }
        run.summary = (
            f"Generated {len(prepared)} greedy continuations from the curated prompt corpus; "
            f"progressive records: `{output_path}`."
        )
        notes = ["Raw prompts are tokenized without a chat template or conversational wrapper."]
        notes.append(
            "The first prompt is a cold generation measurement; "
            "subsequent prompts are warm with the model resident."
        )
        if args.device == "cpu":
            notes.append(
                "CPU is the explicitly selected slow reference fallback; "
                "timing is not GPU performance."
            )
        if args.run_note:
            notes.append(args.run_note)
        if args.compare_to:
            notes.append(
                "Per-prompt output prefix lengths and exact equality compare "
                "against the validated base JSONL."
            )
        run.interpretation = " ".join(notes)
        run.add_section(
            "Progressive result files",
            f"- JSONL prompt results: `{output_path}` (one flushed record per completed prompt)\n"
            f"- Runtime telemetry: `{telemetry_path}`\n- Prompt corpus SHA-256: `{dataset_sha256}`",
        )
    print(f"published run {run.path}; results {output_path}; telemetry {telemetry_path}")
    return run.path
