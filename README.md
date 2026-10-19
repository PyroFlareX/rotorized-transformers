# Rotor Transformer

Research code testing whether pretrained Transformer linear weights admit smaller structured Givens-rotor / diagonal-scaling representations without large language-model quality loss.

## Results so far

Model: Qwen3.5-0.8B-Base. Host: Ryzen 5 7600X, RX 7900 XTX (ROCm). Every number below comes from a run directory under
`reports/experiments/`; links point to the source report. Errors are relative L2 (Frobenius for weights) against the
original dense tensor or model.

| Attempt | Training | Stored size vs dense | Accuracy | Speed | Verdict |
|---|---|---|---|---|---|
| 1. Learned butterfly rotor cascade, all 24 FFNs | Yes (gradient fit per projection) | 25% of BF16 bytes | Layer-0 weight error 0.73–0.74; 0/256 prompts matched base, 186/256 outputs blank or end-of-text | Decode 5.8 tok/s fused Triton, 0.28 tok/s reference, vs 37.8–38.2 dense | Failed: generation collapsed |
| 2. Algebraic SVD → GA rotor angles, one full projection | No | 3,672,064 scalars vs 3,670,016 (+0.06%) | Weight error 8.2e-15, output error 8.3e-15 (FP64) | Not measured | Exact, but not compression |
| 3. Clifford block projection, q=4 / q=8 | No | 917,504 / 458,752 coefficients (25% / 12.5%) | Weight error 0.866 / 0.936; retains 24.96% / 12.48% of weight energy | Not measured | Failed: matches random-weight control |

**1. Learned rotor cascade.** Each FFN projection became `D_0 R_1 D_1 … R_k D_k` (37 butterfly rotor blocks), fitted
to the dense weight by gradient descent. Diagnostics isolated the cause to the fitted functions: one replaced layer-0
FFN had output cosine 0.34 and 27% of reference RMS, FP32 masters reproduced it, and BF16 storage rounding added only
1.9–2.4%. Truncated SVD at the same 25% budget was worse (weight error 0.76–0.80).
[decision report](reports/experiments/qwen3.5-0.8b__output-collapse-experiment-decision__b025__20261011T005909Z/report.md) ·
[fused speed](reports/experiments/qwen3.5-0.8b__rotor-triton__b025__20261011T002857Z/report.md) ·
[reference speed](reports/experiments/qwen3.5-0.8b__rotor-reference__b025__20261011T003148Z/report.md) ·
[prompt outputs](reports/prompt-evaluations/fullcompare-fused-20261010-table.md)

**2. Algebraic GA rotors.** Full-rank SVD, with each orthonormal factor reduced to plane-rotor angles. It reconstructs
the 3584×1024 layer-0 up projection to machine precision without training. At full rank the angles plus singular values
sum to exactly `m·n`, so this is a change of coordinates; only the sign vectors push it above dense.
[derivation](docs/ALGEBRAIC_GA_CONVERSION.md) ·
[run](reports/experiments/qwen3.5-0.8b__algebraic-ga-rotors__unrestricted__20261011T040143Z/report.md)

**3. Clifford block projection.** q×q weight blocks are projected in closed form onto left multiplication by a
Cl(2,0) or Cl(3,0) multivector, storing q coefficients per block instead of q². Projecting a random matrix keeps 1/q
of its energy; the pretrained weight kept the same fraction (norm-matched Gaussian controls: 24.95–25.05% and
12.48–12.53%), and feature regrouping did not help. Same-budget SVD retained more (36.66% / 20.29%).
[run](reports/experiments/qwen3.5-0.8b__clifford-block-projection__model.language_model.layers.0.mlp.up_proj.weight.probes-64__20261011T184547Z/report.md)

Next candidate: sums of left and right geometric products, gated on beating the random-weight control.

## Model storage

All base and derived checkpoints live under `/mnt/NAS/MediaHome/Local-Models/` (base: `base/`, new derived outputs: `temp/`) and never in this repository. Paths resolve through `configs/storage.yaml` and `rotor.storage.load_storage()`. Model downloads are never automatic.
Derived model storage uses the configured `temp/` output root, e.g. `/mnt/NAS/MediaHome/Local-Models/temp/<name>/`. Model checkpoints are never written into this repository.

## Algebraic GA conversion (no training)

`scripts/convert_ga_rotors.py` converts explicitly selected local 2D tensors into ordered GA plane-rotation
angles (one scalar per rotation), signs, and singular-value scalings. Full rank is the default;
there is no optimizer or byte-budget constraint.
The direct operator does not retain dense factors. Whole-tensor reconstruction, direct action, and serialized
reload are checked before atomic publication. See [derivation and usage](docs/ALGEBRAIC_GA_CONVERSION.md).
This separate `ga-rotor-svd` format is not accepted by the existing learned `rotor-scaler` model loader.

## Evaluation: curated prompt continuation

`configs/evaluation/prompts-256.jsonl` is a curated corpus of exactly 256 distinct, manually authored natural-language prompts across 16 domains. Each is a raw continuation prefix: no chat template or assistant-turn wrapper is added. The `scripts/evaluate_prompts.py` runner uses the tokenizer in the local base checkpoint, records exact prompt IDs/text/counts and SHA-256 fingerprints, and generates with deterministic greedy decoding (`do_sample=false`) using seed 0 and default `max_new_tokens=64`.

A base run for the 0.8B model:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/evaluate_prompts.py \
  --model qwen3.5-0.8b --device cpu --dtype bfloat16 --max-new-tokens 64 \
  --output reports/prompt-evaluations/qwen35-08b-base-256.jsonl \
  --run-note "Shared host: CPU conversion running concurrently; timing is diagnostic, not isolated throughput."
```

This uses the already downloaded local checkpoint only; no files are fetched. The output is a progressive JSONL file outside checkpoint storage. One row is flushed and synced after each prompt, including input and output texts, IDs, token counts, token hashes, total tokens, `measure_generation` timing and resource fields, and a cold-first/warm-resident timing label. `RuntimeTelemetry` samples model loading and inference to the adjacent `.telemetry.jsonl` with phase/request IDs. CPU is explicitly labeled as the slow reference fallback; concurrency notes are preserved in the run report. `ExperimentRun` publishes provenance, environment, metrics, and logs under `reports/experiments/`.

For a real tiny-checkpoint smoke, use a tiny local model directory with a tokenizer: `.venv/bin/python scripts/evaluate_prompts.py --base-dir /path/to/tiny/qwen --limit 1 --output /tmp/tiny-prompts.jsonl`. `--limit 1` is also supported with the NAS model for smoke validation.

After conversion to `/mnt/NAS/MediaHome/Local-Models/temp/<name>`, run converted output against the saved base file:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/evaluate_prompts.py \
  --model qwen3.5-0.8b --ckpt /mnt/NAS/MediaHome/Local-Models/temp/<name> \
  --device cpu --dtype bfloat16 --max-new-tokens 64 \
  --compare-to reports/prompt-evaluations/qwen35-08b-base-256.jsonl \
  --output reports/prompt-evaluations/qwen35-08b-converted-256.jsonl \
  --run-note "Shared host: CPU workload concurrent; timing is diagnostic, not isolated throughput."
```

Before inference, `--compare-to` validates dataset hash, selected prompts, base model config/revision, tokenizer fingerprint, seed, dtype, device, decoding settings, input text/token IDs/count/hash, plus baseline output integrity. A mismatch is rejected. Each matched output row records longest-common-prefix length and exact equality. Output and telemetry paths are exclusive and cannot overwrite prior results. Model/tokenizer loading is local-only.

## Other scripts

- `scripts/evaluate_model.py`: loss, generation, multiple-choice, and passkey evaluation.
- `scripts/convert_model.py`: projection-local checkpoint conversion from experiment YAML.
- `scripts/fit_joint_ffn.py`: joint gate/up/down FFN output fitting and export (`configs/experiments/qwen3.5-0.8b-joint-ffn.yaml`).
- `scripts/convert_ga_rotors.py`: algebraic GA tensor conversion with measured numerical gates, no training.
- `scripts/explore_clifford.py`: no-training q=4/q=8 geometric-product block projection pilot, with measured
  compactness, weight/action error, grouping controls, and numerical operator parity.
- `scripts/diagnose_collapse.py`: bounded rotor parity, serialization, activation-trace, and fixed-prompt ablation diagnostics.
- `scripts/benchmark_inference.py`: resident base versus rotor generation benchmark on identical prompt tokens.
- `scripts/benchmark.py`: linear-layer microbenchmark.
- `.venv/bin/python -m pytest`: CPU test suite.
