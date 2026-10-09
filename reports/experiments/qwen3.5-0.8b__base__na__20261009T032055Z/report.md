# Curated prompt generation: qwen3.5-0.8b / base

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__base__na__20261009T032055Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T03:20:55+00:00 |
| Kind | prompt-generation |
| Model | qwen3.5-0.8b |
| Method | base |
| Compression target | na |

## Summary

Generated 256 greedy continuations from the curated prompt corpus; progressive records: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.jsonl`.

## Progressive result files

- JSONL prompt results: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.jsonl` (one flushed record per completed prompt)
- Runtime telemetry: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.telemetry.jsonl`
- Prompt corpus SHA-256: `73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641`

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| provenance.prompt_dataset | /mnt/Projects/rotorized-transformers/configs/evaluation/prompts-256.jsonl |
| provenance.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| provenance.prompt_count | 256 |
| provenance.evaluated_prompt_count | 256 |
| provenance.tokenizer.name | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| provenance.tokenizer.files_sha256.tokenizer.json | fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927 |
| provenance.tokenizer.files_sha256.tokenizer_config.json | e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de |
| provenance.tokenizer.files_sha256.vocab.json | ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003 |
| provenance.tokenizer.files_sha256.merges.txt | a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d |
| provenance.tokenizer.sha256 | 2e62378632909af479a51106c74337cb6a11f23cbc832e71eb422931ba85a118 |
| provenance.model_config_sha256 | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |
| provenance.base_model_hf_id | Qwen/Qwen3.5-0.8B-Base |
| provenance.base_model_revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| provenance.base_model_path | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| provenance.checkpoint_path |  |
| provenance.seed | 0 |
| provenance.device | cpu |
| provenance.dtype | bfloat16 |
| provenance.max_new_tokens | 64 |
| provenance.run_note | Shared host: CPU conversion is running concurrently; timing is diagnostic, not isolated throughput. |
| settings.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| settings.base_model.hf_id | Qwen/Qwen3.5-0.8B-Base |
| settings.base_model.revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| settings.base_model.config_sha256 | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |
| settings.base_model.model_type | qwen3_5 |
| settings.tokenizer_sha256 | 2e62378632909af479a51106c74337cb6a11f23cbc832e71eb422931ba85a118 |
| settings.seed | 0 |
| settings.device | cpu |
| settings.dtype | bfloat16 |
| settings.max_new_tokens | 64 |
| settings.decoding.strategy | greedy |
| settings.decoding.do_sample | False |
| settings.decoding.chat_template | False |
| results.jsonl_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.jsonl |
| results.jsonl_rows | 256 |
| results.telemetry_jsonl_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.telemetry.jsonl |
| results.runtime_telemetry.schema_version | 1 |
| results.runtime_telemetry.measured | True |
| results.runtime_telemetry.samples_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.telemetry.jsonl |
| results.runtime_telemetry.sampling_interval_seconds | 1 |
| results.runtime_telemetry.sample_count | 3396 |
| results.runtime_telemetry.duration_seconds | 3460.62 |
| results.runtime_telemetry.cpu_logical_count | 12 |
| results.runtime_telemetry.process_cpu_time_seconds | 10877.5 |
| results.runtime_telemetry.process_cpu_utilization_percent | 314.323 |
| results.runtime_telemetry.current_process_rss_bytes | 2139967488 |
| results.runtime_telemetry.peak_sampled_process_rss_bytes | 2139971584 |
| results.runtime_telemetry.peak_sampled_cpu_utilization_percent | 385.677 |
| results.runtime_telemetry.devices | [{'backend': 'amd_sysfs', 'card': 'card1', 'pci_address': '0000:03:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 6, 'vram_used_bytes': 1090371584, 'vram_total_bytes': 25753026560, 'gtt_used_bytes': 268652544, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 40.0, 'junction': 47.0, 'mem': 47.0}, 'power_w': 32.0}, {'backend': 'amd_sysfs', 'card': 'card2', 'pci_address': '0000:0e:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 0, 'vram_used_bytes': 26116096, 'vram_total_bytes': 536870912, 'gtt_used_bytes': 16257024, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 40.0}, 'power_w': 59.107}] |
| results.runtime_telemetry.peak_sampled_device_memory | [{'backend': 'amd_sysfs', 'device_id': '0000:03:00.0', 'name': None, 'vram_used_bytes': 3315453952, 'gtt_used_bytes': 975417344}, {'backend': 'amd_sysfs', 'device_id': '0000:0e:00.0', 'name': None, 'vram_used_bytes': 26116096, 'gtt_used_bytes': 16257024}] |
| results.runtime_telemetry.host_available_ram_bytes | 44387889152 |
| results.runtime_telemetry.host_available_swap_bytes | 4396154880 |
| results.runtime_telemetry.host_total_swap_bytes | 8589930496 |
| results.runtime_telemetry.process_read_bytes | 42590208 |
| results.runtime_telemetry.process_write_bytes | 6365184 |
| results.runtime_telemetry.phase | generation |
| results.runtime_telemetry.request_id | reasoning-016 |
| results.runtime_telemetry.sampling_error |  |
| results.generation_total_seconds | 3241.07 |
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
| swap used | 5.65 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | ada677fe96ec412a1c1af45312c00bf875e03c60 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (5.6 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

None.

## Interpretation

Raw prompts are tokenized without a chat template or conversational wrapper. The first prompt is a cold generation measurement; subsequent prompts are warm with the model resident. CPU is the explicitly selected slow reference fallback; timing is not GPU performance. Shared host: CPU conversion is running concurrently; timing is diagnostic, not isolated throughput.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.jsonl
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256-r2.telemetry.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
