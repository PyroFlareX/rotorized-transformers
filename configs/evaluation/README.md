# Evaluation configs

YAML files for `scripts/evaluate_model.py --config <file>`. Keys are the CLI option names (`--seq-len` ->
`seq_len`; dashes are also accepted). Explicit CLI flags override the YAML. Unknown keys are rejected.
`null` values are ignored (the CLI default applies). Relative paths are resolved from the current directory.

| key | type | default | meaning |
|---|---|---|---|
| `base_dir` / `model` / `ckpt` | path / str / path | none | base checkpoint dir, key in `configs/models.yaml`, derived rotor checkpoint (at least one) |
| `text_file` | path or null | null | user-approved plain-text corpus; **exactly one** of `text_file` / `random_tokens` is required |
| `random_tokens` | bool | false | synthetic uniform tokens, smoke tests only (uses a hashed toy tokenizer if the dir has none) |
| `seq_len`, `num_windows`, `seed` | int | 512, 8, 0 | non-overlapping windows, seeded subset; identical for base and converted |
| `device`, `dtype`, `batch_size` | str, `float32`/`bfloat16`, int | cpu, float32, 1 | runtime |
| `long_context_buckets` | list or `"a,b,c"` | null | per-position NLL buckets; needs `--short-context-verified` |
| `generation_prompt_tokens`, `max_new_tokens` | int | 0, 32 | greedy KV-cache generation; the prompt is the first N tokens of window 0 (0 disables) |
| `mc_jsonl` | path or null | null | multiple-choice JSONL, one `{"context", "choices": [..], "answer": idx}` per line |
| `passkey`, `passkey_context` | int | 0, 0 | number of SYNTHETIC passkey items and tokens per prompt (set together; needs `--short-context-verified`) |
| `max_context` | int | 2048 | contexts above this also need `--short-context-verified` |
| `short_context_verified` | bool | false | operator assertion that the short-context path matched its reference |
| `compare_base` | bool | false | evaluate base on identical inputs; report side-by-side table, deltas, generation match |
| `name` | str | null | free-form label stored in the run config |

## Outputs

Each run publishes `reports/experiments/<model>__<method>__<target>__<UTC stamp>/` (see `reports/experiments/README.md`);
`method` is `base` or `rotor:<ckpt name>`, `target` comes from the checkpoint manifest's `budget` (`na` for base).
`metrics.json` records the tokenizer name and `sha256` of `tokenizer.json` (when present), the windows fingerprint,
and hashes of the corpus / JSONL files. Everything in the Measured metrics section is measured on this host.

## Metric definitions

- **NLL / perplexity**: next-token cross-entropy over every position of each window (float64 accumulation).
- **Generation**: greedy, deterministic. `ttft_seconds` = prefill + first token; `prefill_tokens_per_second` =
  prompt tokens / ttft; `decode_tokens_per_second` = (new tokens - 1) / time of the remaining steps. Peak RSS is a
  process high-water mark (upper bound); peak device bytes only on CUDA. Generated token ids of base and converted
  are compared (`match_prefix_len`, `exact_match`).
- **Multiple choice**: summed continuation log-likelihood (`accuracy`) and per-token normalised (`accuracy_norm`).
- **Passkey**: synthetic, generated in code from a fixed word list; a random 5-digit key at seeded depths; exact match
  of the greedily decoded key tokens. It is a probe, not a standard benchmark.

## Policy

Longer-context evaluation is allowed only after the short-context path matches its reference behaviour; the script
refuses passkey, position buckets, or any context above `max_context` unless `--short-context-verified` is passed.

- `smoke.yaml` -- tiny synthetic run (seconds).
- `qwen3.5-short-context.yaml` -- 512-token windows x 32, generation 128+64; corpus must be supplied by the user.
