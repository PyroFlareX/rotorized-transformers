# diagnosis: qwen3.5-0.8b / collapse-input-inventory / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__collapse-input-inventory__b025__20261010T232359Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:23:59+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | collapse-input-inventory |
| Compression target | b025 |

## Summary

Execution prerequisite inventory; existing progressive output snapshot is qualitative only, not held-out language quality.

## Prerequisites

- No approved real-text corpus configured or recorded; calibration/training/held-out splits and held-out quality gates unavailable.
- Existing matched GPU evaluation owns GPU; preserve without competing.

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| qualitative_snapshot.base_rows | 256 |
| qualitative_snapshot.converted_rows_at_read | 29 |
| qualitative_snapshot.all_whitespace_rows | 7 |
| qualitative_snapshot.empty_rows | 0 |
| qualitative_snapshot.matched_input_fingerprints | True |
| qualitative_snapshot.output_token_frequency | [[248044, 725], [220, 582], [271, 455], [198, 56], [11, 33], [13, 5]] |
| snapshot.path | /mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3.snapshots/snapshot-1791623497413929547-49eb563c4c464f4699da11f2e945cf3e |
| snapshot.completed | 69 |
| snapshot.completed_modules | [69 items] |
| snapshot.active | model.layers.23.mlp.down_proj |
| checkpoint_integrity.projections | 72 |
| checkpoint_integrity.loaded_scalar_count | 64862208 |
| checkpoint_integrity.nonfinite_values | 0 |
| runtime.cpu_torch | 2.14.1+cpu |
| runtime.rocm_torch | 2.10.0+rocm7.0 |
| runtime.hip | 7.0.51831 |
| runtime.gpu_available | True |
| runtime.protected_gpu_evaluation_pid | 2015520 |
| blocked_prerequisites | ['No approved real-text corpus configured or recorded; calibration/training/held-out splits and held-out quality gates unavailable.', 'Existing matched GPU evaluation owns GPU; preserve without competing.'] |
| whole_model_fp32_intervention_allowed | False |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.09 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | d8da3c574cc0d838d4d66ac82f6c336d3d0d7dd1 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.1 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

None.

## Interpretation

No full-model scaling authorization: required held-out quality gate has not been measured. Latest snapshot covers only 69 completed projections, so a whole-model FP32 master intervention is invalid.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
