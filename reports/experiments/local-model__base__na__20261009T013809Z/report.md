# Curated prompt generation: local-model / base

| Field | Value |
|---|---|
| Run id | `local-model__base__na__20261009T013809Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T01:38:09+00:00 |
| Kind | prompt-generation |
| Model | local-model |
| Method | base |
| Compression target | na |

## Summary

Generated 2 greedy continuations from the curated prompt corpus; progressive records: `/tmp/rotor-prompt-smoke-2y25zy0p/base-results.jsonl`.

## Progressive result files

- JSONL prompt results: `/tmp/rotor-prompt-smoke-2y25zy0p/base-results.jsonl` (one flushed record per completed prompt)
- Runtime telemetry: `/tmp/rotor-prompt-smoke-2y25zy0p/base-results.telemetry.jsonl`
- Prompt corpus SHA-256: `73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641`

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| provenance.prompt_dataset | /mnt/Projects/rotorized-transformers/configs/evaluation/prompts-256.jsonl |
| provenance.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| provenance.prompt_count | 256 |
| provenance.evaluated_prompt_count | 2 |
| provenance.tokenizer.name | /tmp/rotor-prompt-smoke-2y25zy0p/base |
| provenance.tokenizer.files_sha256.tokenizer.json | fc1288cf9a8df1800a3081f77f3382f50e196a0d4b08966d9b1033e005e0dc80 |
| provenance.tokenizer.files_sha256.tokenizer_config.json | 2e82dbc3e20a656b022331e8b616deb37897c4544b5c4489a598793cae589c91 |
| provenance.tokenizer.sha256 | c12c3a3ddb926c50e354fa743b6566496e4cceaecb0945007c813d8c53ff33ce |
| provenance.model_config_sha256 | ffb6951b48c1fc0de525051736835eb8e92e0316326b9b6970f3eabca4956416 |
| provenance.base_model_hf_id |  |
| provenance.base_model_revision |  |
| provenance.base_model_path | /tmp/rotor-prompt-smoke-2y25zy0p/base |
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
| settings.tokenizer_sha256 | c12c3a3ddb926c50e354fa743b6566496e4cceaecb0945007c813d8c53ff33ce |
| settings.seed | 0 |
| settings.device | cpu |
| settings.dtype | float32 |
| settings.max_new_tokens | 4 |
| settings.decoding.strategy | greedy |
| settings.decoding.do_sample | False |
| settings.decoding.chat_template | False |
| results.jsonl_path | /tmp/rotor-prompt-smoke-2y25zy0p/base-results.jsonl |
| results.jsonl_rows | 2 |
| results.telemetry_jsonl_path | /tmp/rotor-prompt-smoke-2y25zy0p/base-results.telemetry.jsonl |
| results.runtime_telemetry.schema_version | 1 |
| results.runtime_telemetry.measured | True |
| results.runtime_telemetry.samples_path | /tmp/rotor-prompt-smoke-2y25zy0p/base-results.telemetry.jsonl |
| results.runtime_telemetry.sampling_interval_seconds | 1 |
| results.runtime_telemetry.sample_count | 2 |
| results.runtime_telemetry.duration_seconds | 0.0595999 |
| results.runtime_telemetry.cpu_logical_count | 12 |
| results.runtime_telemetry.process_cpu_time_seconds | 0.190228 |
| results.runtime_telemetry.process_cpu_utilization_percent | 319.175 |
| results.runtime_telemetry.current_process_rss_bytes | 374390784 |
| results.runtime_telemetry.peak_sampled_process_rss_bytes | 374390784 |
| results.runtime_telemetry.peak_sampled_cpu_utilization_percent | 314.664 |
| results.runtime_telemetry.devices | [{'backend': 'amd_sysfs', 'card': 'card1', 'pci_address': '0000:03:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 16, 'vram_used_bytes': 3143032832, 'vram_total_bytes': 25753026560, 'gtt_used_bytes': 320331776, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 44.0, 'junction': 50.0, 'mem': 50.0}, 'power_w': 35.0}, {'backend': 'amd_sysfs', 'card': 'card2', 'pci_address': '0000:0e:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 0, 'vram_used_bytes': 26116096, 'vram_total_bytes': 536870912, 'gtt_used_bytes': 16257024, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 40.0}, 'power_w': 12.03}] |
| results.runtime_telemetry.peak_sampled_device_memory | [{'backend': 'amd_sysfs', 'device_id': '0000:03:00.0', 'name': None, 'vram_used_bytes': 3143032832, 'gtt_used_bytes': 320331776}, {'backend': 'amd_sysfs', 'device_id': '0000:0e:00.0', 'name': None, 'vram_used_bytes': 26116096, 'gtt_used_bytes': 16257024}] |
| results.runtime_telemetry.host_available_ram_bytes | 52590141440 |
| results.runtime_telemetry.host_available_swap_bytes | 4517539840 |
| results.runtime_telemetry.host_total_swap_bytes | 8589930496 |
| results.runtime_telemetry.process_read_bytes | 135168 |
| results.runtime_telemetry.process_write_bytes | 1216512 |
| results.runtime_telemetry.phase | generation |
| results.runtime_telemetry.request_id | science-002 |
| results.runtime_telemetry.sampling_error |  |
| results.generation_total_seconds | 0.0198985 |
| results.timing_note | No generation warm-up: first measured prompt is cold; later prompts use a resident model. |
| results.comparison.baseline_jsonl |  |
| results.comparison.exact_matches | 0 |
| results.comparison.prompts | 0 |

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

None.

## Interpretation

Raw prompts are tokenized without a chat template or conversational wrapper. The first prompt is a cold generation measurement; subsequent prompts are warm with the model resident. CPU is the explicitly selected slow reference fallback; timing is not GPU performance.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /tmp/rotor-prompt-smoke-2y25zy0p/base-results.jsonl
- /tmp/rotor-prompt-smoke-2y25zy0p/base-results.telemetry.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
