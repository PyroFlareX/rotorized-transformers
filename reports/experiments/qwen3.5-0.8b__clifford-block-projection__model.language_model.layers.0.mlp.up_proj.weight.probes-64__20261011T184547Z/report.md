# Clifford projection pilot: qwen3.5-0.8b / model.language_model.layers.0.mlp.up_proj.weight

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__clifford-block-projection__model.language_model.layers.0.mlp.up_proj.weight.probes-64__20261011T184547Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T18:45:47+00:00 |
| Kind | exploration |
| Model | qwen3.5-0.8b |
| Method | clifford-block-projection |
| Compression target | model.language_model.layers.0.mlp.up_proj.weight.probes-64 |

## Summary

Measured no-training Clifford projections for `model.language_model.layers.0.mlp.up_proj.weight` (3584x1024) from `qwen3.5-0.8b`; 8 projection arms, one shared FP64 probe set, one Gaussian control, and bounded small-block baseline subsets.

## Clifford projection arms

| arm | q | grouping | seed | coefficients | FP64 coeff bytes | relative Frobenius error | retained energy | probe action relative error |
|---|---|---|---|---|---|---|---|---|
| d2/contiguous | 4 | contiguous |  | 917504 | 7340032 | 0.86625 | 0.249612 | 0.865462 |
| d2/seeded_permutation-0 | 4 | seeded_permutation | 0 | 917504 | 7340032 | 0.866251 | 0.249609 | 0.864932 |
| d2/seeded_permutation-1 | 4 | seeded_permutation | 1 | 917504 | 7340032 | 0.866289 | 0.249543 | 0.86789 |
| d2/seeded_permutation-2 | 4 | seeded_permutation | 2 | 917504 | 7340032 | 0.866272 | 0.249572 | 0.866009 |
| d3/contiguous | 8 | contiguous |  | 458752 | 3670016 | 0.935543 | 0.124759 | 0.936111 |
| d3/seeded_permutation-0 | 8 | seeded_permutation | 0 | 458752 | 3670016 | 0.935573 | 0.124703 | 0.935646 |
| d3/seeded_permutation-1 | 8 | seeded_permutation | 1 | 458752 | 3670016 | 0.935624 | 0.124607 | 0.935359 |
| d3/seeded_permutation-2 | 8 | seeded_permutation | 2 | 458752 | 3670016 | 0.93545 | 0.124934 | 0.934659 |

## Equal coefficient-count SVD and random control

| q | SVD rank | SVD relative Frobenius error | SVD probe action relative error | Gaussian control arms |
|---|---|---|---|---|
| 4 | 199 | 0.795848 | 0.795561 | 4 |
| 8 | 99 | 0.892804 | 0.892286 | 4 |

## Bounded equal-budget small-block comparator subset

The complete helper output is in `metrics.json` under `small_block_equal_budget_baselines`; it retains dense, SVD, SVD-Givens, magnitude-sparse, butterfly, rotor-weight, rotor-residual, and both random-control rows, including every infeasible/skipped row. These q-by-q FP32-budget comparisons are not equal-FP64-byte comparisons and are not a broad trained-baseline tournament.

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| scope.experiment | no-training blockwise Clifford left-geometric-product projection pilot |
| scope.checkpoint_export | False |
| scope.weights_downloaded | False |
| scope.full_model_quality_or_performance_claim | False |
| scope.other_trained_baseline_comparison | broad tournament not run; only the separately identified four-step q-by-q small-block subset is included |
| scope.winner_claim | none |
| scope.note | No model-quality or performance claim is made. The bounded small-block equal-budget subset is reported separately for context only; no broad trained-baseline tournament was run. |
| source.model_key | qwen3.5-0.8b |
| source.hf_id | Qwen/Qwen3.5-0.8B-Base |
| source.revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| source.base_path | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| source.tensor | model.language_model.layers.0.mlp.up_proj.weight |
| source.shape | [3584, 1024] |
| source.checkpoint_dtype | BF16 |
| source.shard | model.safetensors-00001-of-00001.safetensors |
| source.shard_sha256 | c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c |
| source.tensor_values_fp64_sha256 | b2d819d404f02442565871dace21687b2752e1181e84806e27ff71387bd4f95a |
| probes.rows | 64 |
| probes.input_features | 1024 |
| probes.dtype | float64 |
| probes.distribution | iid standard Gaussian |
| probes.seed | 42 |
| probes.sha256 | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| probes.shared_by_all_projection_arms_svd_rows_and_controls | True |
| gaussian_control.seed | 43 |
| gaussian_control.distribution | iid standard Gaussian scaled to the source tensor's Frobenius norm |
| gaussian_control.shape | [3584, 1024] |
| gaussian_control.dtype | float64 |
| gaussian_control.sha256 | d7b6aa87bb62794a1ec9110d2aed2f7941cf84aab79a659a2f3c8423d88278e1 |
| gaussian_control.frobenius_norm | 16.4287 |
| gaussian_control.probe_action_fingerprint | c6be0a64e52ef2b55d2a9600469efd533e79b438215f93a09dceee0ecda6d254 |
| clifford_arms.d2/contiguous.dimensions | 2 |
| clifford_arms.d2/contiguous.q | 4 |
| clifford_arms.d2/contiguous.grouping | contiguous |
| clifford_arms.d2/contiguous.seed |  |
| clifford_arms.d2/contiguous.shape | [3584, 1024] |
| clifford_arms.d2/contiguous.coefficient_shape | [896, 256, 4] |
| clifford_arms.d2/contiguous.coefficient_scalar_count | 917504 |
| clifford_arms.d2/contiguous.coefficient_dtype | torch.float64 |
| clifford_arms.d2/contiguous.coefficient_bytes_fp64 | 7340032 |
| clifford_arms.d2/contiguous.permutation_index_bytes | 36864 |
| clifford_arms.d2/contiguous.coefficients_plus_permutation_bytes | 7376896 |
| clifford_arms.d2/contiguous.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d2/contiguous.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d2/contiguous.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d2/contiguous.coefficient_bytes_over_source_bf16 | 1 |
| clifford_arms.d2/contiguous.coefficient_bytes_over_equal_fp64_dense | 0.25 |
| clifford_arms.d2/contiguous.weight_approximation.absolute_frobenius_error | 14.2314 |
| clifford_arms.d2/contiguous.weight_approximation.relative_frobenius_error | 0.86625 |
| clifford_arms.d2/contiguous.retained_energy_fraction | 0.249612 |
| clifford_arms.d2/contiguous.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d2/contiguous.probe_action_vs_original.absolute_frobenius_error | 114.256 |
| clifford_arms.d2/contiguous.probe_action_vs_original.relative_frobenius_error | 0.865462 |
| clifford_arms.d2/contiguous.operator_vs_reconstructed_probe_action_max_abs_error | 1.55431e-15 |
| clifford_arms.d2/contiguous.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d2/contiguous.operator_reconstruction_assertion_passed | True |
| clifford_arms.d2/seeded_permutation-0.dimensions | 2 |
| clifford_arms.d2/seeded_permutation-0.q | 4 |
| clifford_arms.d2/seeded_permutation-0.grouping | seeded_permutation |
| clifford_arms.d2/seeded_permutation-0.seed | 0 |
| clifford_arms.d2/seeded_permutation-0.shape | [3584, 1024] |
| clifford_arms.d2/seeded_permutation-0.coefficient_shape | [896, 256, 4] |
| clifford_arms.d2/seeded_permutation-0.coefficient_scalar_count | 917504 |
| clifford_arms.d2/seeded_permutation-0.coefficient_dtype | torch.float64 |
| clifford_arms.d2/seeded_permutation-0.coefficient_bytes_fp64 | 7340032 |
| clifford_arms.d2/seeded_permutation-0.permutation_index_bytes | 36864 |
| clifford_arms.d2/seeded_permutation-0.coefficients_plus_permutation_bytes | 7376896 |
| clifford_arms.d2/seeded_permutation-0.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d2/seeded_permutation-0.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d2/seeded_permutation-0.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d2/seeded_permutation-0.coefficient_bytes_over_source_bf16 | 1 |
| clifford_arms.d2/seeded_permutation-0.coefficient_bytes_over_equal_fp64_dense | 0.25 |
| clifford_arms.d2/seeded_permutation-0.weight_approximation.absolute_frobenius_error | 14.2314 |
| clifford_arms.d2/seeded_permutation-0.weight_approximation.relative_frobenius_error | 0.866251 |
| clifford_arms.d2/seeded_permutation-0.retained_energy_fraction | 0.249609 |
| clifford_arms.d2/seeded_permutation-0.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d2/seeded_permutation-0.probe_action_vs_original.absolute_frobenius_error | 114.186 |
| clifford_arms.d2/seeded_permutation-0.probe_action_vs_original.relative_frobenius_error | 0.864932 |
| clifford_arms.d2/seeded_permutation-0.operator_vs_reconstructed_probe_action_max_abs_error | 1.60982e-15 |
| clifford_arms.d2/seeded_permutation-0.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d2/seeded_permutation-0.operator_reconstruction_assertion_passed | True |
| clifford_arms.d2/seeded_permutation-1.dimensions | 2 |
| clifford_arms.d2/seeded_permutation-1.q | 4 |
| clifford_arms.d2/seeded_permutation-1.grouping | seeded_permutation |
| clifford_arms.d2/seeded_permutation-1.seed | 1 |
| clifford_arms.d2/seeded_permutation-1.shape | [3584, 1024] |
| clifford_arms.d2/seeded_permutation-1.coefficient_shape | [896, 256, 4] |
| clifford_arms.d2/seeded_permutation-1.coefficient_scalar_count | 917504 |
| clifford_arms.d2/seeded_permutation-1.coefficient_dtype | torch.float64 |
| clifford_arms.d2/seeded_permutation-1.coefficient_bytes_fp64 | 7340032 |
| clifford_arms.d2/seeded_permutation-1.permutation_index_bytes | 36864 |
| clifford_arms.d2/seeded_permutation-1.coefficients_plus_permutation_bytes | 7376896 |
| clifford_arms.d2/seeded_permutation-1.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d2/seeded_permutation-1.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d2/seeded_permutation-1.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d2/seeded_permutation-1.coefficient_bytes_over_source_bf16 | 1 |
| clifford_arms.d2/seeded_permutation-1.coefficient_bytes_over_equal_fp64_dense | 0.25 |
| clifford_arms.d2/seeded_permutation-1.weight_approximation.absolute_frobenius_error | 14.232 |
| clifford_arms.d2/seeded_permutation-1.weight_approximation.relative_frobenius_error | 0.866289 |
| clifford_arms.d2/seeded_permutation-1.retained_energy_fraction | 0.249543 |
| clifford_arms.d2/seeded_permutation-1.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d2/seeded_permutation-1.probe_action_vs_original.absolute_frobenius_error | 114.576 |
| clifford_arms.d2/seeded_permutation-1.probe_action_vs_original.relative_frobenius_error | 0.86789 |
| clifford_arms.d2/seeded_permutation-1.operator_vs_reconstructed_probe_action_max_abs_error | 1.60982e-15 |
| clifford_arms.d2/seeded_permutation-1.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d2/seeded_permutation-1.operator_reconstruction_assertion_passed | True |
| clifford_arms.d2/seeded_permutation-2.dimensions | 2 |
| clifford_arms.d2/seeded_permutation-2.q | 4 |
| clifford_arms.d2/seeded_permutation-2.grouping | seeded_permutation |
| clifford_arms.d2/seeded_permutation-2.seed | 2 |
| clifford_arms.d2/seeded_permutation-2.shape | [3584, 1024] |
| clifford_arms.d2/seeded_permutation-2.coefficient_shape | [896, 256, 4] |
| clifford_arms.d2/seeded_permutation-2.coefficient_scalar_count | 917504 |
| clifford_arms.d2/seeded_permutation-2.coefficient_dtype | torch.float64 |
| clifford_arms.d2/seeded_permutation-2.coefficient_bytes_fp64 | 7340032 |
| clifford_arms.d2/seeded_permutation-2.permutation_index_bytes | 36864 |
| clifford_arms.d2/seeded_permutation-2.coefficients_plus_permutation_bytes | 7376896 |
| clifford_arms.d2/seeded_permutation-2.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d2/seeded_permutation-2.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d2/seeded_permutation-2.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d2/seeded_permutation-2.coefficient_bytes_over_source_bf16 | 1 |
| clifford_arms.d2/seeded_permutation-2.coefficient_bytes_over_equal_fp64_dense | 0.25 |
| clifford_arms.d2/seeded_permutation-2.weight_approximation.absolute_frobenius_error | 14.2317 |
| clifford_arms.d2/seeded_permutation-2.weight_approximation.relative_frobenius_error | 0.866272 |
| clifford_arms.d2/seeded_permutation-2.retained_energy_fraction | 0.249572 |
| clifford_arms.d2/seeded_permutation-2.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d2/seeded_permutation-2.probe_action_vs_original.absolute_frobenius_error | 114.328 |
| clifford_arms.d2/seeded_permutation-2.probe_action_vs_original.relative_frobenius_error | 0.866009 |
| clifford_arms.d2/seeded_permutation-2.operator_vs_reconstructed_probe_action_max_abs_error | 1.55431e-15 |
| clifford_arms.d2/seeded_permutation-2.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d2/seeded_permutation-2.operator_reconstruction_assertion_passed | True |
| clifford_arms.d3/contiguous.dimensions | 3 |
| clifford_arms.d3/contiguous.q | 8 |
| clifford_arms.d3/contiguous.grouping | contiguous |
| clifford_arms.d3/contiguous.seed |  |
| clifford_arms.d3/contiguous.shape | [3584, 1024] |
| clifford_arms.d3/contiguous.coefficient_shape | [448, 128, 8] |
| clifford_arms.d3/contiguous.coefficient_scalar_count | 458752 |
| clifford_arms.d3/contiguous.coefficient_dtype | torch.float64 |
| clifford_arms.d3/contiguous.coefficient_bytes_fp64 | 3670016 |
| clifford_arms.d3/contiguous.permutation_index_bytes | 36864 |
| clifford_arms.d3/contiguous.coefficients_plus_permutation_bytes | 3706880 |
| clifford_arms.d3/contiguous.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d3/contiguous.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d3/contiguous.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d3/contiguous.coefficient_bytes_over_source_bf16 | 0.5 |
| clifford_arms.d3/contiguous.coefficient_bytes_over_equal_fp64_dense | 0.125 |
| clifford_arms.d3/contiguous.weight_approximation.absolute_frobenius_error | 15.3698 |
| clifford_arms.d3/contiguous.weight_approximation.relative_frobenius_error | 0.935543 |
| clifford_arms.d3/contiguous.retained_energy_fraction | 0.124759 |
| clifford_arms.d3/contiguous.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d3/contiguous.probe_action_vs_original.absolute_frobenius_error | 123.583 |
| clifford_arms.d3/contiguous.probe_action_vs_original.relative_frobenius_error | 0.936111 |
| clifford_arms.d3/contiguous.operator_vs_reconstructed_probe_action_max_abs_error | 1.27676e-15 |
| clifford_arms.d3/contiguous.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d3/contiguous.operator_reconstruction_assertion_passed | True |
| clifford_arms.d3/seeded_permutation-0.dimensions | 3 |
| clifford_arms.d3/seeded_permutation-0.q | 8 |
| clifford_arms.d3/seeded_permutation-0.grouping | seeded_permutation |
| clifford_arms.d3/seeded_permutation-0.seed | 0 |
| clifford_arms.d3/seeded_permutation-0.shape | [3584, 1024] |
| clifford_arms.d3/seeded_permutation-0.coefficient_shape | [448, 128, 8] |
| clifford_arms.d3/seeded_permutation-0.coefficient_scalar_count | 458752 |
| clifford_arms.d3/seeded_permutation-0.coefficient_dtype | torch.float64 |
| clifford_arms.d3/seeded_permutation-0.coefficient_bytes_fp64 | 3670016 |
| clifford_arms.d3/seeded_permutation-0.permutation_index_bytes | 36864 |
| clifford_arms.d3/seeded_permutation-0.coefficients_plus_permutation_bytes | 3706880 |
| clifford_arms.d3/seeded_permutation-0.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d3/seeded_permutation-0.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d3/seeded_permutation-0.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d3/seeded_permutation-0.coefficient_bytes_over_source_bf16 | 0.5 |
| clifford_arms.d3/seeded_permutation-0.coefficient_bytes_over_equal_fp64_dense | 0.125 |
| clifford_arms.d3/seeded_permutation-0.weight_approximation.absolute_frobenius_error | 15.3702 |
| clifford_arms.d3/seeded_permutation-0.weight_approximation.relative_frobenius_error | 0.935573 |
| clifford_arms.d3/seeded_permutation-0.retained_energy_fraction | 0.124703 |
| clifford_arms.d3/seeded_permutation-0.probe_fingerprint | 46c4585bae4215cee185fa94ab11e211627b9b7a40c8bdc570edbfbf15edd719 |
| clifford_arms.d3/seeded_permutation-0.probe_action_vs_original.absolute_frobenius_error | 123.521 |
| clifford_arms.d3/seeded_permutation-0.probe_action_vs_original.relative_frobenius_error | 0.935646 |
| clifford_arms.d3/seeded_permutation-0.operator_vs_reconstructed_probe_action_max_abs_error | 1.22125e-15 |
| clifford_arms.d3/seeded_permutation-0.operator_vs_reconstructed_probe_action_tolerance | 1e-10 |
| clifford_arms.d3/seeded_permutation-0.operator_reconstruction_assertion_passed | True |
| clifford_arms.d3/seeded_permutation-1.dimensions | 3 |
| clifford_arms.d3/seeded_permutation-1.q | 8 |
| clifford_arms.d3/seeded_permutation-1.grouping | seeded_permutation |
| clifford_arms.d3/seeded_permutation-1.seed | 1 |
| clifford_arms.d3/seeded_permutation-1.shape | [3584, 1024] |
| clifford_arms.d3/seeded_permutation-1.coefficient_shape | [448, 128, 8] |
| clifford_arms.d3/seeded_permutation-1.coefficient_scalar_count | 458752 |
| clifford_arms.d3/seeded_permutation-1.coefficient_dtype | torch.float64 |
| clifford_arms.d3/seeded_permutation-1.coefficient_bytes_fp64 | 3670016 |
| clifford_arms.d3/seeded_permutation-1.permutation_index_bytes | 36864 |
| clifford_arms.d3/seeded_permutation-1.coefficients_plus_permutation_bytes | 3706880 |
| clifford_arms.d3/seeded_permutation-1.source_tensor_bytes_checkpoint_dtype | 7340032 |
| clifford_arms.d3/seeded_permutation-1.source_tensor_bytes_bf16 | 7340032 |
| clifford_arms.d3/seeded_permutation-1.equal_fp64_dense_bytes | 29360128 |
| clifford_arms.d3/seeded_permutation-1.coefficient_bytes_over_source_bf16 | 0.5 |
| clifford_arms.d3/seeded_permutation-1.coefficient_bytes_over_equal_fp64_dense | 0.125 |
| clifford_arms.d3/seeded_permutation-1.weight_approximation.absolute_frobenius_error | 15.3711 |
| clifford_arms.d3/seeded_permutation-1.weight_approximation.relative_frobenius_error | 0.935624 |
| clifford_arms.d3/seeded_permutation-1.retained_energy_fraction | 0.124607 |

_328 more keys in metrics.json_

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 0.00 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | d8da3c574cc0d838d4d66ac82f6c336d3d0d7dd1 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- git working tree is dirty: results are not tied to a clean commit

## Failures

None.

## Interpretation

These are tensor reconstruction and seeded random-probe action measurements only. The small-block equal-budget rows use an FP32 parameter budget equal in scalar count to q FP64 Clifford coefficients and only four optimizer steps; they are not a broad trained tournament. No model-quality or performance claim, checkpoint export, or winner claim is made.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
