# diagnosis: qwen3.5-0.8b / selected-depth-ffn-interventions / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__selected-depth-ffn-interventions__b025__20261011T005803Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:58:03+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | selected-depth-ffn-interventions |
| Compression target | b025 |

## Summary

Single complete FFN interventions at layers12 and23 on identical fixed diagnostic prompt; no held-out quality gate.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| 12.logits.teacher_to_student_kl_last_token | 0.0408942 |
| 12.logits.teacher_to_student_kl_all_positions_mean | 0.0638281 |
| 12.logits.top_token_agreement | 0.818182 |
| 12.logits.teacher_top_token | 279 |
| 12.logits.student_top_token | 279 |
| 12.logits.finite_teacher_logits | 8194560 |
| 12.logits.finite_student_logits | 8194560 |
| 12.ffn.elements | 33792 |
| 12.ffn.finite_reference | 33792 |
| 12.ffn.finite_actual | 33792 |
| 12.ffn.finite_pairs | 33792 |
| 12.ffn.max_abs_error | 0.158492 |
| 12.ffn.mean_abs_error | 0.0190971 |
| 12.ffn.max_relative_error_nonzero_reference | 14722.8 |
| 12.ffn.relative_l2_error | 0.950322 |
| 12.ffn.cosine_similarity | 0.33311 |
| 12.ffn.reference_rms | 0.0259998 |
| 12.ffn.actual_rms | 0.0117452 |
| 12.ffn.row_rms_reference.min | 0.0203799 |
| 12.ffn.row_rms_reference.p50 | 0.0259746 |
| 12.ffn.row_rms_reference.p90 | 0.0298011 |
| 12.ffn.row_rms_reference.p99 | 0.0313657 |
| 12.ffn.row_rms_reference.max | 0.0313761 |
| 12.ffn.row_rms_actual.min | 0.00930009 |
| 12.ffn.row_rms_actual.p50 | 0.0117947 |
| 12.ffn.row_rms_actual.p90 | 0.0129596 |
| 12.ffn.row_rms_actual.p99 | 0.0140783 |
| 12.ffn.row_rms_actual.max | 0.0142509 |
| 12.ffn.channel_rms_reference.min | 0.00596038 |
| 12.ffn.channel_rms_reference.p50 | 0.0256019 |
| 12.ffn.channel_rms_reference.p90 | 0.0330401 |
| 12.ffn.channel_rms_reference.p99 | 0.0394589 |
| 12.ffn.channel_rms_reference.max | 0.0857112 |
| 12.ffn.channel_rms_actual.min | 6.5227e-06 |
| 12.ffn.channel_rms_actual.p50 | 0.0113919 |
| 12.ffn.channel_rms_actual.p90 | 0.0159239 |
| 12.ffn.channel_rms_actual.p99 | 0.0206965 |
| 12.ffn.channel_rms_actual.max | 0.0431808 |
| 12.ffn.nonfinite_reference | 0 |
| 12.ffn.nonfinite_actual | 0 |
| 12.block.elements | 33792 |
| 12.block.finite_reference | 33792 |
| 12.block.finite_actual | 33792 |
| 12.block.finite_pairs | 33792 |
| 12.block.max_abs_error | 0.158492 |
| 12.block.mean_abs_error | 0.0190971 |
| 12.block.max_relative_error_nonzero_reference | 19056.5 |
| 12.block.relative_l2_error | 0.31052 |
| 12.block.cosine_similarity | 0.950679 |
| 12.block.reference_rms | 0.0795702 |
| 12.block.actual_rms | 0.0768087 |
| 12.block.row_rms_reference.min | 0.0673949 |
| 12.block.row_rms_reference.p50 | 0.0789034 |
| 12.block.row_rms_reference.p90 | 0.0872996 |
| 12.block.row_rms_reference.p99 | 0.0944116 |
| 12.block.row_rms_reference.max | 0.0967455 |
| 12.block.row_rms_actual.min | 0.0646192 |
| 12.block.row_rms_actual.p50 | 0.0759191 |
| 12.block.row_rms_actual.p90 | 0.0847568 |
| 12.block.row_rms_actual.p99 | 0.100967 |
| 12.block.row_rms_actual.max | 0.107203 |
| 12.block.channel_rms_reference.min | 0.0231049 |
| 12.block.channel_rms_reference.p50 | 0.054614 |
| 12.block.channel_rms_reference.p90 | 0.0983383 |
| 12.block.channel_rms_reference.p99 | 0.240107 |
| 12.block.channel_rms_reference.max | 0.542834 |
| 12.block.channel_rms_actual.min | 0.0227866 |
| 12.block.channel_rms_actual.p50 | 0.0512532 |
| 12.block.channel_rms_actual.p90 | 0.0983762 |
| 12.block.channel_rms_actual.p99 | 0.229983 |
| 12.block.channel_rms_actual.max | 0.562305 |
| 12.block.nonfinite_reference | 0 |
| 12.block.nonfinite_actual | 0 |
| 23.logits.teacher_to_student_kl_last_token | 0.456858 |
| 23.logits.teacher_to_student_kl_all_positions_mean | 0.539852 |
| 23.logits.top_token_agreement | 0.666667 |
| 23.logits.teacher_top_token | 279 |
| 23.logits.student_top_token | 279 |
| 23.logits.finite_teacher_logits | 8194560 |
| 23.logits.finite_student_logits | 8194560 |
| 23.ffn.elements | 33792 |
| 23.ffn.finite_reference | 33792 |
| 23.ffn.finite_actual | 33792 |
| 23.ffn.finite_pairs | 33792 |
| 23.ffn.max_abs_error | 6.3914 |
| 23.ffn.mean_abs_error | 0.115441 |
| 23.ffn.max_relative_error_nonzero_reference | 15039 |
| 23.ffn.relative_l2_error | 0.91559 |
| 23.ffn.cosine_similarity | 0.403219 |
| 23.ffn.reference_rms | 0.213827 |
| 23.ffn.actual_rms | 0.0925978 |
| 23.ffn.row_rms_reference.min | 0.146477 |
| 23.ffn.row_rms_reference.p50 | 0.19648 |
| 23.ffn.row_rms_reference.p90 | 0.272908 |
| 23.ffn.row_rms_reference.p99 | 0.32929 |
| 23.ffn.row_rms_reference.max | 0.33381 |
| 23.ffn.row_rms_actual.min | 0.0742109 |
| 23.ffn.row_rms_actual.p50 | 0.0901175 |
| 23.ffn.row_rms_actual.p90 | 0.105802 |
| 23.ffn.row_rms_actual.p99 | 0.127809 |
| 23.ffn.row_rms_actual.max | 0.135017 |
| 23.ffn.channel_rms_reference.min | 0.056188 |
| 23.ffn.channel_rms_reference.p50 | 0.121091 |
| 23.ffn.channel_rms_reference.p90 | 0.251433 |
| 23.ffn.channel_rms_reference.p99 | 0.657901 |
| 23.ffn.channel_rms_reference.max | 2.63256 |
| 23.ffn.channel_rms_actual.min | 0.00753516 |
| 23.ffn.channel_rms_actual.p50 | 0.0477988 |
| 23.ffn.channel_rms_actual.p90 | 0.103537 |
| 23.ffn.channel_rms_actual.p99 | 0.285805 |
| 23.ffn.channel_rms_actual.max | 1.24317 |
| 23.ffn.nonfinite_reference | 0 |
| 23.ffn.nonfinite_actual | 0 |
| 23.block.elements | 33792 |
| 23.block.finite_reference | 33792 |
| 23.block.finite_actual | 33792 |
| 23.block.finite_pairs | 33792 |
| 23.block.max_abs_error | 6.3914 |
| 23.block.mean_abs_error | 0.115441 |
| 23.block.max_relative_error_nonzero_reference | 25337.2 |
| 23.block.relative_l2_error | 0.554862 |
| 23.block.cosine_similarity | 0.840179 |
| 23.block.reference_rms | 0.352841 |
| 23.block.actual_rms | 0.337859 |
| 23.block.row_rms_reference.min | 0.300036 |
| 23.block.row_rms_reference.p50 | 0.339119 |
| 23.block.row_rms_reference.p90 | 0.404604 |
| 23.block.row_rms_reference.p99 | 0.436866 |
| 23.block.row_rms_reference.max | 0.450022 |
| 23.block.row_rms_actual.min | 0.303337 |
| 23.block.row_rms_actual.p50 | 0.33272 |
| 23.block.row_rms_actual.p90 | 0.369417 |
| 23.block.row_rms_actual.p99 | 0.384838 |
| 23.block.row_rms_actual.max | 0.388885 |
| 23.block.channel_rms_reference.min | 0.122659 |
| 23.block.channel_rms_reference.p50 | 0.279198 |
| 23.block.channel_rms_reference.p90 | 0.430653 |
| 23.block.channel_rms_reference.p99 | 1.03097 |
| 23.block.channel_rms_reference.max | 2.03631 |
| 23.block.channel_rms_actual.min | 0.116615 |
| 23.block.channel_rms_actual.p50 | 0.248422 |
| 23.block.channel_rms_actual.p90 | 0.391259 |
| 23.block.channel_rms_actual.p99 | 0.798042 |
| 23.block.channel_rms_actual.max | 4.44421 |
| 23.block.nonfinite_reference | 0 |
| 23.block.nonfinite_actual | 0 |

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
