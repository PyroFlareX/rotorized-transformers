# diagnosis: qwen3.5-0.8b / storage-rounding-fp32-runtime / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__storage-rounding-fp32-runtime__b025__20261011T005418Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:54:18+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | storage-rounding-fp32-runtime |
| Compression target | b025 |

## Summary

Storage-only rounding intervention on all three layer-0 masters with identical captured teacher inputs and FP32 runtime; FP32 masters double payload bytes.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| model.layers.0.mlp.gate_proj.float16.elements | 118272 |
| model.layers.0.mlp.gate_proj.float16.finite_reference | 118272 |
| model.layers.0.mlp.gate_proj.float16.finite_actual | 118272 |
| model.layers.0.mlp.gate_proj.float16.finite_pairs | 118272 |
| model.layers.0.mlp.gate_proj.float16.max_abs_error | 0.0122458 |
| model.layers.0.mlp.gate_proj.float16.mean_abs_error | 0.000533339 |
| model.layers.0.mlp.gate_proj.float16.max_relative_error_nonzero_reference | 138.228 |
| model.layers.0.mlp.gate_proj.float16.relative_l2_error | 0.00246296 |
| model.layers.0.mlp.gate_proj.float16.cosine_similarity | 0.999997 |
| model.layers.0.mlp.gate_proj.float16.reference_rms | 0.342535 |
| model.layers.0.mlp.gate_proj.float16.actual_rms | 0.342519 |
| model.layers.0.mlp.gate_proj.float16.row_rms_reference.min | 0.249207 |
| model.layers.0.mlp.gate_proj.float16.row_rms_reference.p50 | 0.347245 |
| model.layers.0.mlp.gate_proj.float16.row_rms_reference.p90 | 0.3892 |
| model.layers.0.mlp.gate_proj.float16.row_rms_reference.p99 | 0.415034 |
| model.layers.0.mlp.gate_proj.float16.row_rms_reference.max | 0.422169 |
| model.layers.0.mlp.gate_proj.float16.row_rms_actual.min | 0.249255 |
| model.layers.0.mlp.gate_proj.float16.row_rms_actual.p50 | 0.347231 |
| model.layers.0.mlp.gate_proj.float16.row_rms_actual.p90 | 0.389155 |
| model.layers.0.mlp.gate_proj.float16.row_rms_actual.p99 | 0.414987 |
| model.layers.0.mlp.gate_proj.float16.row_rms_actual.max | 0.422091 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_reference.min | 0.000875762 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_reference.p50 | 0.267745 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_reference.p90 | 0.498664 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_reference.p99 | 0.87918 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_reference.max | 1.7852 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_actual.min | 0.000876514 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_actual.p50 | 0.267703 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_actual.p90 | 0.498605 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_actual.p99 | 0.881018 |
| model.layers.0.mlp.gate_proj.float16.channel_rms_actual.max | 1.78662 |
| model.layers.0.mlp.gate_proj.float16.nonfinite_reference | 0 |
| model.layers.0.mlp.gate_proj.float16.nonfinite_actual | 0 |
| model.layers.0.mlp.gate_proj.bfloat16.elements | 118272 |
| model.layers.0.mlp.gate_proj.bfloat16.finite_reference | 118272 |
| model.layers.0.mlp.gate_proj.bfloat16.finite_actual | 118272 |
| model.layers.0.mlp.gate_proj.bfloat16.finite_pairs | 118272 |
| model.layers.0.mlp.gate_proj.bfloat16.max_abs_error | 0.0914215 |
| model.layers.0.mlp.gate_proj.bfloat16.mean_abs_error | 0.00425881 |
| model.layers.0.mlp.gate_proj.bfloat16.max_relative_error_nonzero_reference | 447.85 |
| model.layers.0.mlp.gate_proj.bfloat16.relative_l2_error | 0.0192471 |
| model.layers.0.mlp.gate_proj.bfloat16.cosine_similarity | 0.999815 |
| model.layers.0.mlp.gate_proj.bfloat16.reference_rms | 0.342535 |
| model.layers.0.mlp.gate_proj.bfloat16.actual_rms | 0.34259 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_reference.min | 0.249207 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_reference.p50 | 0.347245 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_reference.p90 | 0.3892 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_reference.p99 | 0.415034 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_reference.max | 0.422169 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_actual.min | 0.249519 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_actual.p50 | 0.347198 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_actual.p90 | 0.389533 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_actual.p99 | 0.414939 |
| model.layers.0.mlp.gate_proj.bfloat16.row_rms_actual.max | 0.421992 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_reference.min | 0.000875762 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_reference.p50 | 0.267745 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_reference.p90 | 0.498664 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_reference.p99 | 0.87918 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_reference.max | 1.7852 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_actual.min | 0.000875581 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_actual.p50 | 0.267276 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_actual.p90 | 0.499117 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_actual.p99 | 0.873009 |
| model.layers.0.mlp.gate_proj.bfloat16.channel_rms_actual.max | 1.7905 |
| model.layers.0.mlp.gate_proj.bfloat16.nonfinite_reference | 0 |
| model.layers.0.mlp.gate_proj.bfloat16.nonfinite_actual | 0 |
| model.layers.0.mlp.gate_proj.original_serialized.elements | 118272 |
| model.layers.0.mlp.gate_proj.original_serialized.finite_reference | 118272 |
| model.layers.0.mlp.gate_proj.original_serialized.finite_actual | 118272 |
| model.layers.0.mlp.gate_proj.original_serialized.finite_pairs | 118272 |
| model.layers.0.mlp.gate_proj.original_serialized.max_abs_error | 0.0914215 |
| model.layers.0.mlp.gate_proj.original_serialized.mean_abs_error | 0.00425881 |
| model.layers.0.mlp.gate_proj.original_serialized.max_relative_error_nonzero_reference | 447.85 |
| model.layers.0.mlp.gate_proj.original_serialized.relative_l2_error | 0.0192471 |
| model.layers.0.mlp.gate_proj.original_serialized.cosine_similarity | 0.999815 |
| model.layers.0.mlp.gate_proj.original_serialized.reference_rms | 0.342535 |
| model.layers.0.mlp.gate_proj.original_serialized.actual_rms | 0.34259 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_reference.min | 0.249207 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_reference.p50 | 0.347245 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_reference.p90 | 0.3892 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_reference.p99 | 0.415034 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_reference.max | 0.422169 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_actual.min | 0.249519 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_actual.p50 | 0.347198 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_actual.p90 | 0.389533 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_actual.p99 | 0.414939 |
| model.layers.0.mlp.gate_proj.original_serialized.row_rms_actual.max | 0.421992 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_reference.min | 0.000875762 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_reference.p50 | 0.267745 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_reference.p90 | 0.498664 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_reference.p99 | 0.87918 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_reference.max | 1.7852 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_actual.min | 0.000875581 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_actual.p50 | 0.267276 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_actual.p90 | 0.499117 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_actual.p99 | 0.873009 |
| model.layers.0.mlp.gate_proj.original_serialized.channel_rms_actual.max | 1.7905 |
| model.layers.0.mlp.gate_proj.original_serialized.nonfinite_reference | 0 |
| model.layers.0.mlp.gate_proj.original_serialized.nonfinite_actual | 0 |
| model.layers.0.mlp.up_proj.float16.elements | 118272 |
| model.layers.0.mlp.up_proj.float16.finite_reference | 118272 |
| model.layers.0.mlp.up_proj.float16.finite_actual | 118272 |
| model.layers.0.mlp.up_proj.float16.finite_pairs | 118272 |
| model.layers.0.mlp.up_proj.float16.max_abs_error | 0.00786757 |
| model.layers.0.mlp.up_proj.float16.mean_abs_error | 0.000297566 |
| model.layers.0.mlp.up_proj.float16.max_relative_error_nonzero_reference | 544.114 |
| model.layers.0.mlp.up_proj.float16.relative_l2_error | 0.00305001 |
| model.layers.0.mlp.up_proj.float16.cosine_similarity | 0.999995 |
| model.layers.0.mlp.up_proj.float16.reference_rms | 0.13086 |
| model.layers.0.mlp.up_proj.float16.actual_rms | 0.130861 |
| model.layers.0.mlp.up_proj.float16.row_rms_reference.min | 0.0738706 |
| model.layers.0.mlp.up_proj.float16.row_rms_reference.p50 | 0.138269 |
| model.layers.0.mlp.up_proj.float16.row_rms_reference.p90 | 0.147026 |
| model.layers.0.mlp.up_proj.float16.row_rms_reference.p99 | 0.155295 |
| model.layers.0.mlp.up_proj.float16.row_rms_reference.max | 0.157477 |
| model.layers.0.mlp.up_proj.float16.row_rms_actual.min | 0.0738793 |
| model.layers.0.mlp.up_proj.float16.row_rms_actual.p50 | 0.138281 |
| model.layers.0.mlp.up_proj.float16.row_rms_actual.p90 | 0.14703 |
| model.layers.0.mlp.up_proj.float16.row_rms_actual.p99 | 0.155292 |
| model.layers.0.mlp.up_proj.float16.row_rms_actual.max | 0.157475 |
| model.layers.0.mlp.up_proj.float16.channel_rms_reference.min | 8.13966e-06 |
| model.layers.0.mlp.up_proj.float16.channel_rms_reference.p50 | 0.127613 |
| model.layers.0.mlp.up_proj.float16.channel_rms_reference.p90 | 0.168057 |
| model.layers.0.mlp.up_proj.float16.channel_rms_reference.p99 | 0.209255 |
| model.layers.0.mlp.up_proj.float16.channel_rms_reference.max | 0.653874 |
| model.layers.0.mlp.up_proj.float16.channel_rms_actual.min | 8.12895e-06 |
| model.layers.0.mlp.up_proj.float16.channel_rms_actual.p50 | 0.127612 |
| model.layers.0.mlp.up_proj.float16.channel_rms_actual.p90 | 0.168133 |
| model.layers.0.mlp.up_proj.float16.channel_rms_actual.p99 | 0.209363 |
| model.layers.0.mlp.up_proj.float16.channel_rms_actual.max | 0.653353 |
| model.layers.0.mlp.up_proj.float16.nonfinite_reference | 0 |
| model.layers.0.mlp.up_proj.float16.nonfinite_actual | 0 |
| model.layers.0.mlp.up_proj.bfloat16.elements | 118272 |
| model.layers.0.mlp.up_proj.bfloat16.finite_reference | 118272 |
| model.layers.0.mlp.up_proj.bfloat16.finite_actual | 118272 |
| model.layers.0.mlp.up_proj.bfloat16.finite_pairs | 118272 |
| model.layers.0.mlp.up_proj.bfloat16.max_abs_error | 0.0450319 |
| model.layers.0.mlp.up_proj.bfloat16.mean_abs_error | 0.0023737 |
| model.layers.0.mlp.up_proj.bfloat16.max_relative_error_nonzero_reference | 1021.27 |
| model.layers.0.mlp.up_proj.bfloat16.relative_l2_error | 0.0243196 |
| model.layers.0.mlp.up_proj.bfloat16.cosine_similarity | 0.999704 |
| model.layers.0.mlp.up_proj.bfloat16.reference_rms | 0.13086 |
| model.layers.0.mlp.up_proj.bfloat16.actual_rms | 0.130826 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_reference.min | 0.0738706 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_reference.p50 | 0.138269 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_reference.p90 | 0.147026 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_reference.p99 | 0.155295 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_reference.max | 0.157477 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_actual.min | 0.0737895 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_actual.p50 | 0.138189 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_actual.p90 | 0.147056 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_actual.p99 | 0.155083 |
| model.layers.0.mlp.up_proj.bfloat16.row_rms_actual.max | 0.157143 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_reference.min | 8.13966e-06 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_reference.p50 | 0.127613 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_reference.p90 | 0.168057 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_reference.p99 | 0.209255 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_reference.max | 0.653874 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_actual.min | 8.14821e-06 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_actual.p50 | 0.127706 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_actual.p90 | 0.16818 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_actual.p99 | 0.209359 |
| model.layers.0.mlp.up_proj.bfloat16.channel_rms_actual.max | 0.647681 |
| model.layers.0.mlp.up_proj.bfloat16.nonfinite_reference | 0 |
| model.layers.0.mlp.up_proj.bfloat16.nonfinite_actual | 0 |
| model.layers.0.mlp.up_proj.original_serialized.elements | 118272 |
| model.layers.0.mlp.up_proj.original_serialized.finite_reference | 118272 |
| model.layers.0.mlp.up_proj.original_serialized.finite_actual | 118272 |
| model.layers.0.mlp.up_proj.original_serialized.finite_pairs | 118272 |
| model.layers.0.mlp.up_proj.original_serialized.max_abs_error | 0.0450319 |
| model.layers.0.mlp.up_proj.original_serialized.mean_abs_error | 0.0023737 |
| model.layers.0.mlp.up_proj.original_serialized.max_relative_error_nonzero_reference | 1021.27 |
| model.layers.0.mlp.up_proj.original_serialized.relative_l2_error | 0.0243196 |
| model.layers.0.mlp.up_proj.original_serialized.cosine_similarity | 0.999704 |
| model.layers.0.mlp.up_proj.original_serialized.reference_rms | 0.13086 |
| model.layers.0.mlp.up_proj.original_serialized.actual_rms | 0.130826 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_reference.min | 0.0738706 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_reference.p50 | 0.138269 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_reference.p90 | 0.147026 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_reference.p99 | 0.155295 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_reference.max | 0.157477 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_actual.min | 0.0737895 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_actual.p50 | 0.138189 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_actual.p90 | 0.147056 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_actual.p99 | 0.155083 |
| model.layers.0.mlp.up_proj.original_serialized.row_rms_actual.max | 0.157143 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_reference.min | 8.13966e-06 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_reference.p50 | 0.127613 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_reference.p90 | 0.168057 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_reference.p99 | 0.209255 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_reference.max | 0.653874 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_actual.min | 8.14821e-06 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_actual.p50 | 0.127706 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_actual.p90 | 0.16818 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_actual.p99 | 0.209359 |
| model.layers.0.mlp.up_proj.original_serialized.channel_rms_actual.max | 0.647681 |
| model.layers.0.mlp.up_proj.original_serialized.nonfinite_reference | 0 |
| model.layers.0.mlp.up_proj.original_serialized.nonfinite_actual | 0 |
| model.layers.0.mlp.down_proj.float16.elements | 33792 |
| model.layers.0.mlp.down_proj.float16.finite_reference | 33792 |

_97 more keys in metrics.json_

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

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
