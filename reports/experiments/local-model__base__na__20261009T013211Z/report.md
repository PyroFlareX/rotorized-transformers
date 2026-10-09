# Curated prompt generation: local-model / base

| Field | Value |
|---|---|
| Run id | `local-model__base__na__20261009T013211Z` |
| Status | **failed** |
| Created (UTC) | 2026-10-09T01:32:11+00:00 |
| Kind | prompt-generation |
| Model | local-model |
| Method | base |
| Compression target | na |

## Summary

_(none provided)_

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| provenance.prompt_dataset | /mnt/Projects/rotorized-transformers/configs/evaluation/prompts-256.jsonl |
| provenance.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| provenance.prompt_count | 256 |
| provenance.evaluated_prompt_count | 2 |
| provenance.tokenizer.name | /tmp/rotor-prompt-smoke-v82q05pq/base |
| provenance.tokenizer.files_sha256.tokenizer.json | ef1c3ff58cc98c1a751cc0fa074dff66d033b78f9b906bf13eb4b7e6563fd418 |
| provenance.tokenizer.files_sha256.tokenizer_config.json | 2e82dbc3e20a656b022331e8b616deb37897c4544b5c4489a598793cae589c91 |
| provenance.tokenizer.sha256 | adfd1d298310edca0c63b33461d0afcf46bae040b330ebbc1fa6fd24ea7984a3 |
| provenance.model_config_sha256 | ffb6951b48c1fc0de525051736835eb8e92e0316326b9b6970f3eabca4956416 |
| provenance.base_model_hf_id |  |
| provenance.base_model_revision |  |
| provenance.base_model_path | /tmp/rotor-prompt-smoke-v82q05pq/base |
| provenance.checkpoint_path |  |
| provenance.seed | 0 |
| provenance.device | cpu |
| provenance.dtype | float32 |
| provenance.max_new_tokens | 4 |
| provenance.run_note |  |
| settings.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| settings.base_model.hf_id |  |
| settings.base_model.revision |  |
| settings.base_model.config_sha256 | ffb6951b48c1fc0de525051736835eb8e92e0316326b9b6970f3eabca4956416 |
| settings.base_model.model_type | qwen3_5 |
| settings.tokenizer_sha256 | adfd1d298310edca0c63b33461d0afcf46bae040b330ebbc1fa6fd24ea7984a3 |
| settings.seed | 0 |
| settings.device | cpu |
| settings.dtype | float32 |
| settings.max_new_tokens | 4 |
| settings.decoding.strategy | greedy |
| settings.decoding.do_sample | False |
| settings.decoding.chat_template | False |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.79 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | ada677fe96ec412a1c1af45312c00bf875e03c60 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.8 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

```
Traceback (most recent call last):
  File "/mnt/Projects/rotorized-transformers/src/rotor/evaluation/prompts.py", line 264, in run_prompt_evaluation
    raise ValueError(f"tokenizer produced no input tokens for {prompt['id']}")
ValueError: tokenizer produced no input tokens for science-001

```

## Interpretation

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /tmp/rotor-prompt-smoke-v82q05pq/base-results.jsonl
- /tmp/rotor-prompt-smoke-v82q05pq/base-results.telemetry.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
