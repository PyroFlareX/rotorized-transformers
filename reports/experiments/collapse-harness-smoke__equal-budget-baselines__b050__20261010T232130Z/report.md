# baselines: collapse-harness-smoke / equal-budget-baselines / b050

| Field | Value |
|---|---|
| Run id | `collapse-harness-smoke__equal-budget-baselines__b050__20261010T232130Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:21:30+00:00 |
| Kind | baselines |
| Model | collapse-harness-smoke |
| Method | equal-budget-baselines |
| Compression target | b050 |

## Summary

Equal-stored-byte comparison at b050 (byte ratio 0.5 of dense BF16 bytes) on 1 matrix/matrices; methods: dense (reference), svd, svd_givens, magnitude_sparse, butterfly, rotor_weight, rotor_residual, rotor_activation (only with calibration inputs); controls: spectrum_matched_random, gaussian_random. Schedule: 4 steps.

## Baseline comparison

| matrix | section | method | status | params | stored_bytes | index_bytes | rel_fro | rel_spectral | act_rel_fro | estimated_flops_per_token |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic[12,16] | method | dense | reference | 192 | 384 | 0 | 0 | 0 | 0 | 384 |
| synthetic[12,16] | method | svd | ok | 84 | 168 | 0 | 0.523508 | 0.378929 | 0.521315 | 168 |
| synthetic[12,16] | method | svd_givens | ok | 75 | 151 | 0 | 0.523508 | 0.378929 | 0.521315 | 435 |
| synthetic[12,16] | method | magnitude_sparse | ok | 32 | 192 | 128 | 0.612726 | 0.466917 | 0.605223 | 64 |
| synthetic[12,16] | method | butterfly | skipped: budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |  |  |  |  |  |  |  |
| synthetic[12,16] | method | rotor_weight | ok | 60 | 120 | 0 | 1.23149 | 0.99529 | 1.22968 | 220 |
| synthetic[12,16] | method | rotor_residual | skipped: 25% of 96 params is below one rank-1 residual (28 params) |  |  |  |  |  |  |  |
| synthetic[12,16] | method | rotor_activation | ok | 60 | 120 | 0 | 1.2315 | 0.995308 | 1.22967 | 220 |
| synthetic[12,16] | control:spectrum_matched_random | svd | ok | 84 | 168 | 0 | 0.523508 | 0.378929 | 0.537556 | 168 |
| synthetic[12,16] | control:spectrum_matched_random | magnitude_sparse | ok | 32 | 192 | 128 | 0.598987 | 0.419394 | 0.599489 | 64 |
| synthetic[12,16] | control:spectrum_matched_random | butterfly | skipped: budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |  |  |  |  |  |  |  |
| synthetic[12,16] | control:spectrum_matched_random | rotor_weight | ok | 60 | 120 | 0 | 1.27019 | 1.03157 | 1.27074 | 220 |
| synthetic[12,16] | control:gaussian_random | svd | ok | 84 | 168 | 0 | 0.667421 | 0.726528 | 0.65306 | 168 |
| synthetic[12,16] | control:gaussian_random | magnitude_sparse | ok | 32 | 192 | 128 | 0.635512 | 0.63453 | 0.636504 | 64 |
| synthetic[12,16] | control:gaussian_random | butterfly | skipped: budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |  |  |  |  |  |  |  |
| synthetic[12,16] | control:gaussian_random | rotor_weight | ok | 60 | 120 | 0 | 1.40408 | 1.32606 | 1.40384 | 220 |

## Notes

- Equal STORED-BYTE budget is enforced: param_budget = byte_budget // bytes_per_param(storage_dtype) and every non-reference method's stored_bytes <= byte_budget. The `dense` row is the uncompressed REFERENCE (reference: true, stored at storage_dtype, error 0) and is exempt from the budget. Compute differs per method and is reported separately as estimated_flops_per_token; it is not equalized.
- estimated_flops_per_token are formula ESTIMATES (multiply+add per input vector), not measurements: dense 2*in*out; svd 2*r*(in+out); svd_givens 6*angles + r; magnitude_sparse 2*nnz; butterfly 6 per 2x2 block (n/2 blocks per factor); rotor CascadeSpec.flops_per_token() (6 per Givens pair, 1 per diagonal entry, 2*r*(in+out) residual).
- weight_errors/activation_errors are measured on the float32/float64 master values; rounding the stored parameters to storage_dtype is NOT simulated (storage_dtype only sets bytes per parameter).
- svd is ordinary truncated-SVD compression: any gain over dense it shows is a rank-truncation effect, not a rotor gain. A structural (rotor) gain requires the rotor rows to beat svd and svd_givens at equal bytes.
- svd_givens stored_bytes include ceil(2*rank/8) bytes of 1-bit Givens signs on top of params*bytes_per_param.
- magnitude_sparse keeps the largest-|w| entries (largest nnz that fits); stored_bytes = nnz*bytes_per_param + index_bytes where the index encoding is the smaller of int32 flat COO indices (nnz*4) and int32 CSR (nnz*4 + (rows+1)*4). It is not fit by gradient descent.
- butterfly is the classic factorization with FREE 2x2 blocks (4 params each, not rotations) on a width padded to the next power of two of max(in,out) (pad/truncate), as a product of m butterfly matrices of log2(n) factors; m is the largest that fits. Fit by Adam with the same FitSchedule (the block tensor is trained in every phase whose trainable kinds include 'scales', at the 'scales' learning rate). Skipped if one product exceeds the budget.
- rotor_weight is the full-rank rotor/scaler cascade with NO residual (residual_rank is forced to 0). rotor_residual adds a low-rank residual: spec_kwargs['residual_rank'] if > 0, otherwise the rank whose residual parameters r*(in+out) take at most ~25% of the parameter budget.
- index_bytes: 0 for svd, svd_givens, butterfly and rotor rows (deterministic topology; no stored indices).
- controls: spectrum_matched_random keeps W's singular values with Haar-random singular vectors; gaussian_random is iid Gaussian with W's Frobenius norm. svd, magnitude_sparse, butterfly and rotor_weight are re-run on each control at the same budget; control errors are measured against the control matrix itself (and X for activation_errors).
- rotor_activation is fit from scratch with the same schedule and structure as rotor_weight (no warm start from the weight fit).
- REMINDER: gains from truncated-SVD-derived methods are NOT rotor-specific. Only the rotor_* rows trained from scratch at the same stored bytes, compared against the svd row and the random controls, bear on the rotor hypothesis. est_flops are formula ESTIMATES, not measurements.

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| byte_ratio | 0.5 |
| results.synthetic[12,16].budget.byte_budget | 192 |
| results.synthetic[12,16].budget.storage_dtype | bfloat16 |
| results.synthetic[12,16].budget.bytes_per_param | 2 |
| results.synthetic[12,16].budget.param_budget | 96 |
| results.synthetic[12,16].budget.in_features | 12 |
| results.synthetic[12,16].budget.out_features | 16 |
| results.synthetic[12,16].budget.dense_params | 192 |
| results.synthetic[12,16].budget.dense_bytes | 384 |
| results.synthetic[12,16].methods.dense.reference | True |
| results.synthetic[12,16].methods.dense.params | 192 |
| results.synthetic[12,16].methods.dense.stored_bytes | 384 |
| results.synthetic[12,16].methods.dense.index_bytes | 0 |
| results.synthetic[12,16].methods.dense.weight_errors.rel_fro | 0 |
| results.synthetic[12,16].methods.dense.weight_errors.rel_spectral | 0 |
| results.synthetic[12,16].methods.dense.weight_errors.max_abs | 0 |
| results.synthetic[12,16].methods.dense.activation_errors.rel_fro | 0 |
| results.synthetic[12,16].methods.dense.activation_errors.mean_row_cosine | 1 |
| results.synthetic[12,16].methods.dense.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.dense.estimated_flops_per_token | 384 |
| results.synthetic[12,16].methods.svd.rank | 3 |
| results.synthetic[12,16].methods.svd.params | 84 |
| results.synthetic[12,16].methods.svd.stored_bytes | 168 |
| results.synthetic[12,16].methods.svd.index_bytes | 0 |
| results.synthetic[12,16].methods.svd.weight_errors.rel_fro | 0.523508 |
| results.synthetic[12,16].methods.svd.weight_errors.rel_spectral | 0.378929 |
| results.synthetic[12,16].methods.svd.weight_errors.max_abs | 0.15771 |
| results.synthetic[12,16].methods.svd.activation_errors.rel_fro | 0.521315 |
| results.synthetic[12,16].methods.svd.activation_errors.mean_row_cosine | 0.780425 |
| results.synthetic[12,16].methods.svd.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.svd.estimated_flops_per_token | 168 |
| results.synthetic[12,16].methods.svd_givens.rank | 3 |
| results.synthetic[12,16].methods.svd_givens.params | 75 |
| results.synthetic[12,16].methods.svd_givens.sign_bytes | 1 |
| results.synthetic[12,16].methods.svd_givens.stored_bytes | 151 |
| results.synthetic[12,16].methods.svd_givens.index_bytes | 0 |
| results.synthetic[12,16].methods.svd_givens.weight_errors.rel_fro | 0.523508 |
| results.synthetic[12,16].methods.svd_givens.weight_errors.rel_spectral | 0.378929 |
| results.synthetic[12,16].methods.svd_givens.weight_errors.max_abs | 0.15771 |
| results.synthetic[12,16].methods.svd_givens.activation_errors.rel_fro | 0.521315 |
| results.synthetic[12,16].methods.svd_givens.activation_errors.mean_row_cosine | 0.780425 |
| results.synthetic[12,16].methods.svd_givens.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.svd_givens.estimated_flops_per_token | 435 |
| results.synthetic[12,16].methods.magnitude_sparse.nnz | 32 |
| results.synthetic[12,16].methods.magnitude_sparse.density | 0.166667 |
| results.synthetic[12,16].methods.magnitude_sparse.encoding | coo_flat |
| results.synthetic[12,16].methods.magnitude_sparse.params | 32 |
| results.synthetic[12,16].methods.magnitude_sparse.stored_bytes | 192 |
| results.synthetic[12,16].methods.magnitude_sparse.index_bytes | 128 |
| results.synthetic[12,16].methods.magnitude_sparse.weight_errors.rel_fro | 0.612726 |
| results.synthetic[12,16].methods.magnitude_sparse.weight_errors.rel_spectral | 0.466917 |
| results.synthetic[12,16].methods.magnitude_sparse.weight_errors.max_abs | 0.138758 |
| results.synthetic[12,16].methods.magnitude_sparse.activation_errors.rel_fro | 0.605223 |
| results.synthetic[12,16].methods.magnitude_sparse.activation_errors.mean_row_cosine | 0.766771 |
| results.synthetic[12,16].methods.magnitude_sparse.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.magnitude_sparse.estimated_flops_per_token | 64 |
| results.synthetic[12,16].methods.butterfly.skipped | budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |
| results.synthetic[12,16].methods.rotor_weight.params | 60 |
| results.synthetic[12,16].methods.rotor_weight.stored_bytes | 120 |
| results.synthetic[12,16].methods.rotor_weight.index_bytes | 0 |
| results.synthetic[12,16].methods.rotor_weight.weight_errors.rel_fro | 1.23149 |
| results.synthetic[12,16].methods.rotor_weight.weight_errors.rel_spectral | 0.99529 |
| results.synthetic[12,16].methods.rotor_weight.weight_errors.max_abs | 0.473984 |
| results.synthetic[12,16].methods.rotor_weight.activation_errors.rel_fro | 1.22968 |
| results.synthetic[12,16].methods.rotor_weight.activation_errors.mean_row_cosine | 0.14581 |
| results.synthetic[12,16].methods.rotor_weight.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.rotor_weight.estimated_flops_per_token | 220 |
| results.synthetic[12,16].methods.rotor_weight.spec.in_features | 12 |
| results.synthetic[12,16].methods.rotor_weight.spec.out_features | 16 |
| results.synthetic[12,16].methods.rotor_weight.spec.num_rotors | 1 |
| results.synthetic[12,16].methods.rotor_weight.spec.stages_per_rotor | 4 |
| results.synthetic[12,16].methods.rotor_weight.spec.topology.kind | butterfly |
| results.synthetic[12,16].methods.rotor_weight.spec.topology.seed | 0 |
| results.synthetic[12,16].methods.rotor_weight.spec.adapter | pad_truncate |
| results.synthetic[12,16].methods.rotor_weight.spec.residual_rank | 0 |
| results.synthetic[12,16].methods.rotor_weight.fit.objective | weight |
| results.synthetic[12,16].methods.rotor_weight.fit.final_loss | 1.52693 |
| results.synthetic[12,16].methods.rotor_weight.fit.steps | 5 |
| results.synthetic[12,16].methods.rotor_residual.skipped | 25% of 96 params is below one rank-1 residual (28 params) |
| results.synthetic[12,16].methods.rotor_activation.params | 60 |
| results.synthetic[12,16].methods.rotor_activation.stored_bytes | 120 |
| results.synthetic[12,16].methods.rotor_activation.index_bytes | 0 |
| results.synthetic[12,16].methods.rotor_activation.weight_errors.rel_fro | 1.2315 |
| results.synthetic[12,16].methods.rotor_activation.weight_errors.rel_spectral | 0.995308 |
| results.synthetic[12,16].methods.rotor_activation.weight_errors.max_abs | 0.473984 |
| results.synthetic[12,16].methods.rotor_activation.activation_errors.rel_fro | 1.22967 |
| results.synthetic[12,16].methods.rotor_activation.activation_errors.mean_row_cosine | 0.145836 |
| results.synthetic[12,16].methods.rotor_activation.activation_errors.rows | 512 |
| results.synthetic[12,16].methods.rotor_activation.estimated_flops_per_token | 220 |
| results.synthetic[12,16].methods.rotor_activation.spec.in_features | 12 |
| results.synthetic[12,16].methods.rotor_activation.spec.out_features | 16 |
| results.synthetic[12,16].methods.rotor_activation.spec.num_rotors | 1 |
| results.synthetic[12,16].methods.rotor_activation.spec.stages_per_rotor | 4 |
| results.synthetic[12,16].methods.rotor_activation.spec.topology.kind | butterfly |
| results.synthetic[12,16].methods.rotor_activation.spec.topology.seed | 0 |
| results.synthetic[12,16].methods.rotor_activation.spec.adapter | pad_truncate |
| results.synthetic[12,16].methods.rotor_activation.spec.residual_rank | 0 |
| results.synthetic[12,16].methods.rotor_activation.fit.objective | activation |
| results.synthetic[12,16].methods.rotor_activation.fit.final_loss | 1.52231 |
| results.synthetic[12,16].methods.rotor_activation.fit.steps | 5 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.rank | 3 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.params | 84 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.stored_bytes | 168 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.index_bytes | 0 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.weight_errors.rel_fro | 0.523508 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.weight_errors.rel_spectral | 0.378929 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.weight_errors.max_abs | 0.182257 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.activation_errors.rel_fro | 0.537556 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.activation_errors.mean_row_cosine | 0.763001 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.spectrum_matched_random.svd.estimated_flops_per_token | 168 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.nnz | 32 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.density | 0.166667 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.encoding | coo_flat |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.params | 32 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.stored_bytes | 192 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.index_bytes | 128 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.weight_errors.rel_fro | 0.598987 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.weight_errors.rel_spectral | 0.419394 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.weight_errors.max_abs | 0.143486 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.activation_errors.rel_fro | 0.599489 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.activation_errors.mean_row_cosine | 0.758904 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.spectrum_matched_random.magnitude_sparse.estimated_flops_per_token | 64 |
| results.synthetic[12,16].controls.spectrum_matched_random.butterfly.skipped | budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.params | 60 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.stored_bytes | 120 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.index_bytes | 0 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.weight_errors.rel_fro | 1.27019 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.weight_errors.rel_spectral | 1.03157 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.weight_errors.max_abs | 0.558315 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.activation_errors.rel_fro | 1.27074 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.activation_errors.mean_row_cosine | 0.0936085 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.estimated_flops_per_token | 220 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.in_features | 12 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.out_features | 16 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.num_rotors | 1 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.stages_per_rotor | 4 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.topology.kind | butterfly |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.topology.seed | 0 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.adapter | pad_truncate |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.spec.residual_rank | 0 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.fit.objective | weight |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.fit.final_loss | 1.62356 |
| results.synthetic[12,16].controls.spectrum_matched_random.rotor_weight.fit.steps | 5 |
| results.synthetic[12,16].controls.gaussian_random.svd.rank | 3 |
| results.synthetic[12,16].controls.gaussian_random.svd.params | 84 |
| results.synthetic[12,16].controls.gaussian_random.svd.stored_bytes | 168 |
| results.synthetic[12,16].controls.gaussian_random.svd.index_bytes | 0 |
| results.synthetic[12,16].controls.gaussian_random.svd.weight_errors.rel_fro | 0.667421 |
| results.synthetic[12,16].controls.gaussian_random.svd.weight_errors.rel_spectral | 0.726528 |
| results.synthetic[12,16].controls.gaussian_random.svd.weight_errors.max_abs | 0.278041 |
| results.synthetic[12,16].controls.gaussian_random.svd.activation_errors.rel_fro | 0.65306 |
| results.synthetic[12,16].controls.gaussian_random.svd.activation_errors.mean_row_cosine | 0.694908 |
| results.synthetic[12,16].controls.gaussian_random.svd.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.gaussian_random.svd.estimated_flops_per_token | 168 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.nnz | 32 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.density | 0.166667 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.encoding | coo_flat |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.params | 32 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.stored_bytes | 192 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.index_bytes | 128 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.weight_errors.rel_fro | 0.635512 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.weight_errors.rel_spectral | 0.63453 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.weight_errors.max_abs | 0.14504 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.activation_errors.rel_fro | 0.636504 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.activation_errors.mean_row_cosine | 0.754196 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.gaussian_random.magnitude_sparse.estimated_flops_per_token | 64 |
| results.synthetic[12,16].controls.gaussian_random.butterfly.skipped | budget of 96 params is below one butterfly product (128 params = log2(16) factors x 2*16) |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.params | 60 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.stored_bytes | 120 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.index_bytes | 0 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.weight_errors.rel_fro | 1.40408 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.weight_errors.rel_spectral | 1.32606 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.weight_errors.max_abs | 0.630035 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.activation_errors.rel_fro | 1.40384 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.activation_errors.mean_row_cosine | -0.0887664 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.activation_errors.rows | 512 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.estimated_flops_per_token | 220 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.in_features | 12 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.out_features | 16 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.num_rotors | 1 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.stages_per_rotor | 4 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.topology.kind | butterfly |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.topology.seed | 0 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.adapter | pad_truncate |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.spec.residual_rank | 0 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.fit.objective | weight |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.fit.final_loss | 1.98478 |
| results.synthetic[12,16].controls.gaussian_random.rotor_weight.fit.steps | 5 |
| results.synthetic[12,16].notes | [11 items] |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

| key | value |
|---|---|
| estimated_flops_per_token.synthetic[12,16]/method/dense | 384 |
| estimated_flops_per_token.synthetic[12,16]/method/svd | 168 |
| estimated_flops_per_token.synthetic[12,16]/method/svd_givens | 435 |
| estimated_flops_per_token.synthetic[12,16]/method/magnitude_sparse | 64 |
| estimated_flops_per_token.synthetic[12,16]/method/rotor_weight | 220 |
| estimated_flops_per_token.synthetic[12,16]/method/rotor_activation | 220 |
| estimated_flops_per_token.synthetic[12,16]/control:spectrum_matched_random/svd | 168 |
| estimated_flops_per_token.synthetic[12,16]/control:spectrum_matched_random/magnitude_sparse | 64 |
| estimated_flops_per_token.synthetic[12,16]/control:spectrum_matched_random/rotor_weight | 220 |
| estimated_flops_per_token.synthetic[12,16]/control:gaussian_random/svd | 168 |
| estimated_flops_per_token.synthetic[12,16]/control:gaussian_random/magnitude_sparse | 64 |
| estimated_flops_per_token.synthetic[12,16]/control:gaussian_random/rotor_weight | 220 |

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

- synthetic[12,16] @ b050: the lowest MEASURED weight-space rel_fro among compressed methods was `svd_givens` (0.523508). Single matrix, single seed, single schedule.

Gains from SVD-derived methods (`svd`, `svd_givens`) are rank-truncation effects and are NOT rotor gains. A rotor gain exists only where a `rotor_*` row beats both SVD-derived rows at equal stored bytes and also beats the same method's result on the random controls. Errors here are weight-space (and calibration-activation where an `act_rel_fro` is shown) reconstruction errors, not language-model quality. `estimated_flops_per_token` are formula estimates, not measurements; compute is not equalized across methods.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
