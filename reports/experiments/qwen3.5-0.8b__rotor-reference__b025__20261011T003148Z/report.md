# inference-benchmark: qwen3.5-0.8b / rotor-reference / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-reference__b025__20261011T003148Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:31:48+00:00 |
| Kind | inference-benchmark |
| Model | qwen3.5-0.8b |
| Method | rotor-reference |
| Compression target | b025 |

## Summary

Resident GPU inference, three identical curated prompts with explicit warmup and repeated trials.

## Warmed inference

| model | backend | TTFT_ms_median | prefill_tok_s_median | decode_tok_s_median | peak_device_bytes |
|---|---|---|---|---|---|
| base | torch-dense | 51.585 | 639.721 | 38.2431 | 1594103296 |
| converted | reference | 3393.74 | 8.84036 | 0.275318 | 1398810112 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| results.base | [{'prompt_id': 'science-001', 'repeat': 0, 'prompt_tokens': 33, 'generated_tokens': 64, 'ttft_seconds': 0.05158496298827231, 'prefill_tokens_per_second': 639.7213080777523, 'decode_seconds': 1.6473581420141272, 'decode_tokens_per_second': 38.24305012568404, 'total_seconds': 1.6989431050023995, 'resources': {'wall_seconds': 1.6992839229933452, 'cpu_seconds': 1.3196273209999996, 'peak_rss_bytes': 4037144576, 'peak_device_bytes': 1594103296, 'measured': True}, 'generated_ids': [279, 9041, 70614, 8364, 5964, 11, 748, 424, 31130, 8364, 10281, 11, 3154, 424, 2579, 72555, 13, 1061, 4145, 21254, 7062, 279, 6228, 364, 264, 18539, 9915, 883, 1204, 8364, 33060, 1472, 2086, 30999, 13, 271, 5666, 11, 279, 10756, 37631, 279, 7059, 314, 27462, 94494, 13, 2838, 14330, 421, 264, 3558, 725, 5548, 310, 8059, 8364, 13402, 383, 1141, 5138, 5759, 13, 89520], 'measured': True}, {'prompt_id': 'science-002', 'repeat': 0, 'prompt_tokens': 26, 'generated_tokens': 64, 'ttft_seconds': 0.0743881490197964, 'prefill_tokens_per_second': 349.51803940007704, 'decode_seconds': 1.6396262740017846, 'decode_tokens_per_second': 38.42339013404431, 'total_seconds': 1.714014423021581, 'resources': {'wall_seconds': 1.714245095005026, 'cpu_seconds': 1.4333666310000002, 'peak_rss_bytes': 4037144576, 'peak_device_bytes': 1593869824, 'measured': True}, 'generated_ids': [271, 16, 13, 2972, 64859, 5130, 4649, 314, 279, 35896, 332, 1317, 279, 35896, 725, 11051, 369, 15625, 1083, 8364, 321, 4950, 11, 864, 513, 524, 9476, 430, 21461, 4649, 13, 198, 17, 13, 2972, 64859, 5130, 4649, 314, 279, 33828, 332, 1317, 279, 33828, 725, 11051, 369, 22773, 1083, 8364, 321, 4950, 11, 1495, 524, 33129, 430, 21461, 4649, 13, 198, 18, 13], 'measured': True}, {'prompt_id': 'science-003', 'repeat': 0, 'prompt_tokens': 29, 'generated_tokens': 64, 'ttft_seconds': 0.03661800699774176, 'prefill_tokens_per_second': 791.9600867897709, 'decode_seconds': 1.928415214002598, 'decode_tokens_per_second': 32.669312885806306, 'total_seconds': 1.9650332210003398, 'resources': {'wall_seconds': 1.9652616090024821, 'cpu_seconds': 1.425225609, 'peak_rss_bytes': 4037144576, 'peak_device_bytes': 1593969152, 'measured': True}, 'generated_ids': [271, 32, 13, 279, 3594, 198, 33, 13, 279, 9561, 198, 34, 13, 279, 8720, 198, 35, 13, 279, 1898, 271, 248068, 198, 1596, 513, 2574, 25, 328, 4350, 264, 8720, 314, 9561, 2919, 369, 2047, 383, 264, 7948, 1898, 11, 6756, 86412, 4833, 383, 279, 4732, 13, 561, 2919, 303, 1782, 6756, 86412, 3571, 494, 25, 357, 13, 279, 3594, 417, 13, 279], 'measured': True}] |
| results.converted | [{'prompt_id': 'science-001', 'repeat': 0, 'prompt_tokens': 33, 'generated_tokens': 64, 'ttft_seconds': 3.393739868013654, 'prefill_tokens_per_second': 9.72378593628474, 'decode_seconds': 221.87275701199542, 'decode_tokens_per_second': 0.2839465324559605, 'total_seconds': 225.26649688000907, 'resources': {'wall_seconds': 225.26985125200008, 'cpu_seconds': 169.346605487, 'peak_rss_bytes': 5782560768, 'peak_device_bytes': 1398810112, 'measured': True}, 'generated_ids': [220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220], 'measured': True}, {'prompt_id': 'science-002', 'repeat': 0, 'prompt_tokens': 26, 'generated_tokens': 64, 'ttft_seconds': 2.9410582190030254, 'prefill_tokens_per_second': 8.840355431254812, 'decode_seconds': 228.82661420598743, 'decode_tokens_per_second': 0.275317625174876, 'total_seconds': 231.76767242499045, 'resources': {'wall_seconds': 231.77233012398938, 'cpu_seconds': 170.70980141599998, 'peak_rss_bytes': 5782560768, 'peak_device_bytes': 1398576640, 'measured': True}, 'generated_ids': [271, 271, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044], 'measured': True}, {'prompt_id': 'science-003', 'repeat': 0, 'prompt_tokens': 29, 'generated_tokens': 64, 'ttft_seconds': 3.788038005994167, 'prefill_tokens_per_second': 7.655678204418907, 'decode_seconds': 236.5488402310002, 'decode_tokens_per_second': 0.26632977755662535, 'total_seconds': 240.33687823699438, 'resources': {'wall_seconds': 240.33737337999628, 'cpu_seconds': 174.87690613999996, 'peak_rss_bytes': 5782560768, 'peak_device_bytes': 1398675968, 'measured': True}, 'generated_ids': [271, 271, 248044, 248044, 248044, 271, 271, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044, 271, 248044, 248044, 248044, 248044, 271, 248044, 248044, 271, 248044, 248044, 248044, 248044, 271, 248044, 271, 248044, 248044, 271, 248044, 271, 271, 271, 271, 271, 271, 248044], 'measured': True}] |
| summary | [{'model': 'base', 'backend': 'torch-dense', 'TTFT_ms_median': 51.58496298827231, 'prefill_tok_s_median': 639.7213080777523, 'decode_tok_s_median': 38.24305012568404, 'peak_device_bytes': 1594103296}, {'model': 'converted', 'backend': 'reference', 'TTFT_ms_median': 3393.739868013654, 'prefill_tok_s_median': 8.840355431254812, 'decode_tok_s_median': 0.275317625174876, 'peak_device_bytes': 1398810112}] |

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

Short-prompt batch-one measurements; no optimized DeltaNet kernels. Reference and fused rotor backends execute stored rotations, not dense expansion. Output agreement is not a language-quality score. Peak RSS is process high-water mark.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
