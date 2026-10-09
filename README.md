# Rotor Transformer

Research code testing whether pretrained Transformer linear weights admit smaller structured Givens-rotor / diagonal-scaling representations without large language-model quality loss.

## Model storage

All base and derived checkpoints live under `/mnt/NAS/MediaHome/Local-Models/` (base: `base/`, new derived outputs: `temp/`) and never in this repository. Paths resolve through `configs/storage.yaml` and `rotor.storage.load_storage()`. Model downloads are never automatic.
Derived model storage uses the configured `temp/` output root, e.g. `/mnt/NAS/MediaHome/Local-Models/temp/<name>/`. Model checkpoints are never written into this repository.

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
- `scripts/convert_model.py`: local checkpoint conversion from experiment YAML.
- `scripts/benchmark.py`: linear-layer microbenchmark.
- `.venv/bin/python -m pytest`: CPU test suite (after integration by the main agent).
