# inference-benchmark: qwen3.5-0.8b / rotor-triton / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-triton__b025__20261011T002857Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:28:57+00:00 |
| Kind | inference-benchmark |
| Model | qwen3.5-0.8b |
| Method | rotor-triton |
| Compression target | b025 |

## Summary

Resident GPU inference, three identical curated prompts with explicit warmup and repeated trials.

## Warmed inference

| model | backend | TTFT_ms_median | prefill_tok_s_median | decode_tok_s_median | peak_device_bytes |
|---|---|---|---|---|---|
| base | torch-dense | 43.9136 | 699.951 | 37.8318 | 1594103296 |
| converted | triton | 277.142 | 110.378 | 5.79535 | 1781458432 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| results.base | [9 items] |
| results.converted | [9 items] |
| summary | [{'model': 'base', 'backend': 'torch-dense', 'TTFT_ms_median': 43.913572997553274, 'prefill_tok_s_median': 699.9510105527993, 'decode_tok_s_median': 37.83178563827999, 'peak_device_bytes': 1594103296}, {'model': 'converted', 'backend': 'triton', 'TTFT_ms_median': 277.14201901108027, 'prefill_tok_s_median': 110.37798683728765, 'decode_tok_s_median': 5.795347543262021, 'peak_device_bytes': 1781458432}] |

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
