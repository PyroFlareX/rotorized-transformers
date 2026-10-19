#!/usr/bin/env python
"""Benchmark resident base and rotor models on identical curated prompt tokens."""
from __future__ import annotations

import argparse
import gc
import statistics
from pathlib import Path

import torch

from rotor.data import windows_fingerprint
from rotor.evaluation.generation import greedy_decode, measure_generation
from rotor.evaluation.prompts import load_prompts
from rotor.models.loading import load_base, load_converted
from rotor.models.qwen3_5 import load_tokenizer
from rotor.reporting import ExperimentRun, markdown_table
from rotor.telemetry import RuntimeTelemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "triton"], default="triton")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--telemetry", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 2:
        parser.error("repeats must be positive and max-new-tokens >= 2")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("GPU benchmark requires a usable CUDA/ROCm device; CPU fallback refused")
    from rotor.formats import read_manifest

    manifest = read_manifest(args.ckpt)
    base = Path(manifest.base.path)
    tokenizer = load_tokenizer(base)
    prompts = load_prompts()[:3]
    tokens = [torch.tensor([tokenizer.encode(p["prompt"], add_special_tokens=False)]) for p in prompts]
    config = vars(args).copy()
    config.update(dtype="float16", structured_dtype="float32", warmup_tokens=4,
                  prompt_ids=[p["id"] for p in prompts],
                  input_fingerprints=[windows_fingerprint(t) for t in tokens])
    with ExperimentRun("inference-benchmark", model="qwen3.5-0.8b", method=f"rotor-{args.backend}",
                       target="b025", config=config) as run, RuntimeTelemetry(args.telemetry) as telemetry:
        results = {}
        rows = []
        for label in ("base", "converted"):
            telemetry.set_phase(f"{label}-loading")
            model = (load_base(base, dtype=torch.float16, device=args.device) if label == "base" else
                     load_converted(args.ckpt, dtype=torch.float16, device=args.device,
                                    rotor_backend=args.backend)[0])
            model.eval()
            raw = []
            for prompt, ids in zip(prompts, tokens, strict=True):
                telemetry.set_phase(f"{label}-warmup", request_id=prompt["id"])
                greedy_decode(model, ids, 4, args.device)
                for repeat in range(args.repeats):
                    telemetry.set_phase(f"{label}-benchmark", request_id=prompt["id"])
                    measured = measure_generation(model, ids, args.max_new_tokens, args.device)
                    raw.append({"prompt_id": prompt["id"], "repeat": repeat, **measured})
                    run.log(f"{label} {prompt['id']} repeat {repeat}: "
                            f"TTFT={measured['ttft_seconds']:.4f}s decode="
                            f"{measured['decode_tokens_per_second']:.3f} tok/s")
            results[label] = raw
            rows.append({"model": label, "backend": "torch-dense" if label == "base" else args.backend,
                         "TTFT_ms_median": statistics.median(r["ttft_seconds"] for r in raw) * 1000,
                         "prefill_tok_s_median": statistics.median(r["prefill_tokens_per_second"] for r in raw),
                         "decode_tok_s_median": statistics.median(r["decode_tokens_per_second"] for r in raw),
                         "peak_device_bytes": max(r["resources"]["peak_device_bytes"] or 0 for r in raw)})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        run.metrics.update(results=results, summary=rows)
        run.add_section("Warmed inference", markdown_table(rows))
        run.summary = (
            "Resident GPU inference, three identical curated prompts with explicit warmup and repeated trials."
        )
        run.interpretation = ("Short-prompt batch-one measurements; no optimized DeltaNet kernels. "
                              "Reference and fused rotor backends execute stored rotations, not dense expansion. "
                              "Output agreement is not a language-quality score. Peak RSS is process high-water mark.")
    print(f"benchmark report: {run.path}")


if __name__ == "__main__":
    main()
