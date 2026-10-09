# Curated prompt generation: qwen3.5-0.8b / base

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__base__na__20261009T024407Z` |
| Status | **failed** |
| Created (UTC) | 2026-10-09T02:44:07+00:00 |
| Kind | prompt-generation |
| Model | qwen3.5-0.8b |
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

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 4.37 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | ada677fe96ec412a1c1af45312c00bf875e03c60 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (4.4 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

```
Traceback (most recent call last):
  File "/mnt/Projects/rotorized-transformers/src/rotor/evaluation/prompts.py", line 264, in run_prompt_evaluation
    tokenizer = load_tokenizer(base_dir)
                ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/Projects/rotorized-transformers/src/rotor/models/qwen3_5.py", line 157, in load_tokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/Projects/rotorized-transformers/.venv/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 817, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/Projects/rotorized-transformers/.venv/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 652, in get_tokenizer_config
    with open(resolved_config_file, encoding="utf-8") as reader:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 112] Host is down: '/mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base/tokenizer_config.json'

```

## Interpretation

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256.jsonl
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen35-08b-base-256.telemetry.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
