# Curated prompt generation: qwen3.5-0.8b / rotor:qwen3.5-0.8b-full-ffn-b025-gpu-r3

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-qwen3.5-0.8b-full-ffn-b025-gpu-r3__b025__20261010T234116Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:41:16+00:00 |
| Kind | prompt-generation |
| Model | qwen3.5-0.8b |
| Method | rotor:qwen3.5-0.8b-full-ffn-b025-gpu-r3 |
| Compression target | b025 |

## Summary

Generated 256 greedy continuations from the curated prompt corpus; progressive records: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.jsonl`.

## Progressive result files

- JSONL prompt results: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.jsonl` (one flushed record per completed prompt)
- Runtime telemetry: `/mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.telemetry.jsonl`
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
| provenance.checkpoint_path | /mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3 |
| provenance.rotor_backend | triton |
| provenance.effective_runtime_backend | triton |
| provenance.seed | 0 |
| provenance.device | cuda:0 |
| provenance.dtype | float16 |
| provenance.max_new_tokens | 64 |
| provenance.run_note | Fused rotor correctness/output run concurrent with reference evaluation; timings are shared-GPU diagnostics, not isolated performance. |
| settings.prompt_dataset_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| settings.base_model.hf_id | Qwen/Qwen3.5-0.8B-Base |
| settings.base_model.revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| settings.base_model.config_sha256 | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |
| settings.base_model.model_type | qwen3_5 |
| settings.tokenizer_sha256 | 2e62378632909af479a51106c74337cb6a11f23cbc832e71eb422931ba85a118 |
| settings.seed | 0 |
| settings.device | cuda:0 |
| settings.dtype | float16 |
| settings.max_new_tokens | 64 |
| settings.decoding.strategy | greedy |
| settings.decoding.do_sample | False |
| settings.decoding.chat_template | False |
| results.rotor_backend | triton |
| results.effective_runtime_backend | triton |
| results.jsonl_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.jsonl |
| results.jsonl_rows | 256 |
| results.telemetry_jsonl_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.telemetry.jsonl |
| results.runtime_telemetry.schema_version | 1 |
| results.runtime_telemetry.measured | True |
| results.runtime_telemetry.samples_path | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.telemetry.jsonl |
| results.runtime_telemetry.sampling_interval_seconds | 1 |
| results.runtime_telemetry.sample_count | 2807 |
| results.runtime_telemetry.duration_seconds | 2832.63 |
| results.runtime_telemetry.cpu_logical_count | 12 |
| results.runtime_telemetry.process_cpu_time_seconds | 4658.65 |
| results.runtime_telemetry.process_cpu_utilization_percent | 164.464 |
| results.runtime_telemetry.current_process_rss_bytes | 2906816512 |
| results.runtime_telemetry.peak_sampled_process_rss_bytes | 3997818880 |
| results.runtime_telemetry.peak_sampled_cpu_utilization_percent | 201.199 |
| results.runtime_telemetry.devices | [{'backend': 'amd_sysfs', 'card': 'card1', 'pci_address': '0000:03:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 97, 'vram_used_bytes': 13069676544, 'vram_total_bytes': 25753026560, 'gtt_used_bytes': 338022400, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 59.0, 'junction': 70.0, 'mem': 66.0}, 'power_w': 143.0}, {'backend': 'amd_sysfs', 'card': 'card2', 'pci_address': '0000:0e:00.0', 'name': None, 'driver': 'amdgpu', 'gpu_busy_percent': 0, 'vram_used_bytes': 26660864, 'vram_total_bytes': 536870912, 'gtt_used_bytes': 16273408, 'gtt_total_bytes': 33266819072, 'temperature_c': {'edge': 46.0}, 'power_w': 15.057}] |
| results.runtime_telemetry.peak_sampled_device_memory | [{'backend': 'amd_sysfs', 'device_id': '0000:03:00.0', 'name': None, 'vram_used_bytes': 18157740032, 'gtt_used_bytes': 1067696128}, {'backend': 'amd_sysfs', 'device_id': '0000:0e:00.0', 'name': None, 'vram_used_bytes': 26898432, 'gtt_used_bytes': 16281600}] |
| results.runtime_telemetry.host_available_ram_bytes | 37227433984 |
| results.runtime_telemetry.host_available_swap_bytes | 5278167040 |
| results.runtime_telemetry.host_total_swap_bytes | 8589930496 |
| results.runtime_telemetry.process_read_bytes | 122880 |
| results.runtime_telemetry.process_write_bytes | 273166336 |
| results.runtime_telemetry.phase | generation |
| results.runtime_telemetry.request_id | reasoning-016 |
| results.runtime_telemetry.sampling_error |  |
| results.generation_total_seconds | 2780.4 |
| results.timing_note | No generation warm-up: first measured prompt is cold; later prompts use a resident model. |
| results.comparison.baseline_jsonl | /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-fullcompare-20261010.jsonl |
| results.comparison.exact_matches | 0 |
| results.comparison.prompts | 256 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.08 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.10.0+rocm7.0 cuda=True |
| git | d8da3c574cc0d838d4d66ac82f6c336d3d0d7dd1 dirty=True |

Validity warnings for this host:

- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.1 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

None.

## Interpretation

Raw prompts are tokenized without a chat template or conversational wrapper. The first prompt is a cold generation measurement; subsequent prompts are warm with the model resident. Fused rotor correctness/output run concurrent with reference evaluation; timings are shared-GPU diagnostics, not isolated performance. Per-prompt output prefix lengths and exact equality compare against the validated base JSONL.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.jsonl
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-fused-fullcompare-20261010.telemetry.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
