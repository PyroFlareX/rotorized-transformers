# comparison: qwen3.5-0.8b / output-collapse-experiment-decision / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__output-collapse-experiment-decision__b025__20261011T005909Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:59:09+00:00 |
| Kind | comparison |
| Model | qwen3.5-0.8b |
| Method | output-collapse-experiment-decision |
| Compression target | b025 |

## Summary

NO-GO full-model scaling: no held-out gate measured, single-FFN severe functional distortion already present in FP32 masters

## Replacement interventions (one fixed prompt)

| arm | last_token_KL | FFN_relative_L2 | FFN_RMS_ratio | teacher_top_token | student_top_token |
|---|---|---|---|---|---|
| layer-0-ffn.gate | 0.332919 | 0.67419 | 0.693192 | 279 | 1001 |
| layer-0-ffn.up | 0.0675436 | 0.748687 | 0.520672 | 279 | 279 |
| layer-0-ffn.down | 0.0651524 | 0.705951 | 0.69824 | 279 | 279 |
| layer-0-complete-ffn | 0.287596 | 0.942703 | 0.273625 | 279 | 1001 |
| layer-0-complete-ffn-fp32-master | 0.285422 | 0.942611 | 0.273415 | 279 | 1001 |
| layer-12-complete-ffn | 0.0408942 | 0.950322 | 0.451744 | 279 | 279 |
| layer-23-complete-ffn | 0.456858 | 0.91559 | 0.433049 | 279 | 279 |

## Storage-only rounding: relative L2

| module | float16 | bfloat16 | original_serialized |
|---|---|---|---|
| model.layers.0.mlp.gate_proj | 0.00246296 | 0.0192471 | 0.0192471 |
| model.layers.0.mlp.up_proj | 0.00305001 | 0.0243196 | 0.0243196 |
| model.layers.0.mlp.down_proj | 0.00274181 | 0.022271 | 0.022271 |

## SVD spectral screen only; FP32 masters, factor payload budget

| tensor | byte_ratio | rank | factor_payload_bytes | fp32_master_rel_fro |
|---|---|---|---|---|
| model.language_model.layers.0.mlp.down_proj.weight | 0.25 | 199 | 1833984 | 0.791861 |
| model.language_model.layers.0.mlp.down_proj.weight | 0.5 | 398 | 3667968 | 0.61366 |
| model.language_model.layers.0.mlp.down_proj.weight | 0.75 | 597 | 5501952 | 0.442342 |
| model.language_model.layers.0.mlp.gate_proj.weight | 0.25 | 199 | 1833984 | 0.759556 |
| model.language_model.layers.0.mlp.gate_proj.weight | 0.5 | 398 | 3667968 | 0.582953 |
| model.language_model.layers.0.mlp.gate_proj.weight | 0.75 | 597 | 5501952 | 0.421045 |
| model.language_model.layers.0.mlp.up_proj.weight | 0.25 | 199 | 1833984 | 0.795848 |
| model.language_model.layers.0.mlp.up_proj.weight | 0.5 | 398 | 3667968 | 0.616748 |
| model.language_model.layers.0.mlp.up_proj.weight | 0.75 | 597 | 5501952 | 0.447547 |

## Representative outputs (special tokens and whitespace preserved)

### dense-base

```text
[SP]the[SP]metal[SP]conducts[SP]heat[SP]quickly,[SP]so[SP]it
```

### layer-0-ffn.gate

```text
[SP]“Heat[SP]moves[SP]from[SP]the[SP]hotter[SP]object[SP]to
```

### layer-0-ffn.up

```text
[SP]the[SP]spoon[SP]that[SP]conducts[SP]heat[SP]better[SP](the
```

### layer-0-ffn.down

```text
[SP]the[SP]metal’s[SP]higher[SP]thermal[SP]conductivity[SP]lets[SP]heat
```

### layer-0-complete-ffn

```text
[SP]“Heat[SP]moves[SP]from[SP]the[SP]hotter[SP]material[SP]to
```

### layer-0-complete-ffn-fp32-master

```text
[SP]“Heat[SP]moves[SP]from[SP]the[SP]hotter[SP]material[SP]to
```

## Evidence boundaries

- One fixed diagnostic prompt, FP32 CPU arithmetic, eight generated tokens per arm; not held-out quality.
- Old all72 runs FP16/BF16 casts changed runtime arithmetic; do not interpret those cast rows as pure serialization. Dedicated storage-only report supersedes layer0 cast interpretation.
- SVD spectrum report also contains MTP rows from broad selection; final comparison excludes MTP.
- No matched full baseline/seed/topology optimization screen or real-text activation/joint comparison completed.
- 69 completed FP32 masters; whole-model FP32 intervention disallowed.
- No recovery/expansion or long-context evaluation: quality gate not passed.

## Blocked execution

Approved disjoint calibration/recovery/held-out natural text is missing. No reduced-budget arm has passed the NLL and attenuation gate. Recovery, prefix expansion and long-context runs were not started. Full matched-method optimization/seed variability remains unmeasured.

## Artifacts

- inventory: `/mnt/Projects/rotorized-transformers/reports/experiments/qwen3.5-0.8b__collapse-input-inventory__b025__20261010T232359Z`
- operator: `reports/experiments/qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T235114Z`
- layer0: `reports/experiments/qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T235955Z`
- storage_only: `reports/experiments/qwen3.5-0.8b__storage-rounding-fp32-runtime__b025__20261011T005418Z`
- depth: `reports/experiments/qwen3.5-0.8b__selected-depth-ffn-interventions__b025__20261011T005803Z`
- svd: `reports/experiments/qwen3.5-0.8b__collapse-svd-capacity-screen__b025-b075__20261010T232447Z`
- joint_smoke: `reports/experiments/tiny-joint-ffn__joint-output-fit-smoke__synthetic__20261011T003019Z`
- baseline_smoke: `reports/experiments/collapse-harness-smoke__equal-budget-baselines__b050__20261010T232130Z`

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| decision | NO-GO full-model scaling: no held-out gate measured, single-FFN severe functional distortion already present in FP32 masters |
| causal_classification | [INFERENCE] approximation/fitting damage dominates sampled layer0 intervention; serialization contributes but is not necessary for severe attenuation. Optimization vs capacity vs projection-local objective unresolved. Execution not fully tolerance/GPU-qualified. |
| arm_table | [{'arm': 'layer-0-ffn.gate', 'last_token_KL': 0.3329188823699951, 'FFN_relative_L2': 0.6741900479687418, 'FFN_RMS_ratio': 0.6931918525057084, 'teacher_top_token': 279, 'student_top_token': 1001}, {'arm': 'layer-0-ffn.up', 'last_token_KL': 0.06754357367753983, 'FFN_relative_L2': 0.7486874170607497, 'FFN_RMS_ratio': 0.5206721911152515, 'teacher_top_token': 279, 'student_top_token': 279}, {'arm': 'layer-0-ffn.down', 'last_token_KL': 0.06515242159366608, 'FFN_relative_L2': 0.7059509810649601, 'FFN_RMS_ratio': 0.6982395318326374, 'teacher_top_token': 279, 'student_top_token': 279}, {'arm': 'layer-0-complete-ffn', 'last_token_KL': 0.28759562969207764, 'FFN_relative_L2': 0.9427028216379422, 'FFN_RMS_ratio': 0.2736253363620708, 'teacher_top_token': 279, 'student_top_token': 1001}, {'arm': 'layer-0-complete-ffn-fp32-master', 'last_token_KL': 0.28542235493659973, 'FFN_relative_L2': 0.942611416759937, 'FFN_RMS_ratio': 0.2734154057594301, 'teacher_top_token': 279, 'student_top_token': 1001}, {'arm': 'layer-12-complete-ffn', 'last_token_KL': 0.04089416563510895, 'FFN_relative_L2': 0.9503219563274294, 'FFN_RMS_ratio': 0.4517437491616044, 'teacher_top_token': 279, 'student_top_token': 279}, {'arm': 'layer-23-complete-ffn', 'last_token_KL': 0.4568582773208618, 'FFN_relative_L2': 0.9155896601296398, 'FFN_RMS_ratio': 0.43304916991110054, 'teacher_top_token': 279, 'student_top_token': 279}] |
| storage_only_layer0.model.layers.0.mlp.gate_proj.float16 | 0.00246296 |
| storage_only_layer0.model.layers.0.mlp.gate_proj.bfloat16 | 0.0192471 |
| storage_only_layer0.model.layers.0.mlp.gate_proj.original_serialized | 0.0192471 |
| storage_only_layer0.model.layers.0.mlp.up_proj.float16 | 0.00305001 |
| storage_only_layer0.model.layers.0.mlp.up_proj.bfloat16 | 0.0243196 |
| storage_only_layer0.model.layers.0.mlp.up_proj.original_serialized | 0.0243196 |
| storage_only_layer0.model.layers.0.mlp.down_proj.float16 | 0.00274181 |
| storage_only_layer0.model.layers.0.mlp.down_proj.bfloat16 | 0.022271 |
| storage_only_layer0.model.layers.0.mlp.down_proj.original_serialized | 0.022271 |
| operator.projections | 72 |
| operator.all_outputs_finite | True |
| operator.max_dense_action_abs_error | 7.88372e-05 |
| operator.max_serialized_base_weight_relative_frobenius_error | 0.744244 |
| operator.FP64_reference_tolerance.rtol | 1e-10 |
| operator.FP64_reference_tolerance.atol | 1e-10 |
| operator.FP64_reference_tolerance.source | tests/test_math.py RotorCascade forward/dense contract |
| operator.model_precision_note | FP32 action discrepancies are reported, not judged with a newly invented tolerance. |
| svd_spectrum_screen_text_only | [9 items] |
| paths.inventory | /mnt/Projects/rotorized-transformers/reports/experiments/qwen3.5-0.8b__collapse-input-inventory__b025__20261010T232359Z |
| paths.operator | reports/experiments/qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T235114Z |
| paths.layer0 | reports/experiments/qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T235955Z |
| paths.storage_only | reports/experiments/qwen3.5-0.8b__storage-rounding-fp32-runtime__b025__20261011T005418Z |
| paths.depth | reports/experiments/qwen3.5-0.8b__selected-depth-ffn-interventions__b025__20261011T005803Z |
| paths.svd | reports/experiments/qwen3.5-0.8b__collapse-svd-capacity-screen__b025-b075__20261010T232447Z |
| paths.joint_smoke | reports/experiments/tiny-joint-ffn__joint-output-fit-smoke__synthetic__20261011T003019Z |
| paths.baseline_smoke | reports/experiments/collapse-harness-smoke__equal-budget-baselines__b050__20261010T232130Z |
| provisional_gate.NLL_delta_nats | 0.05 |
| provisional_gate.measured | False |
| scope_limits | ['One fixed diagnostic prompt, FP32 CPU arithmetic, eight generated tokens per arm; not held-out quality.', 'Old all72 runs FP16/BF16 casts changed runtime arithmetic; do not interpret those cast rows as pure serialization. Dedicated storage-only report supersedes layer0 cast interpretation.', 'SVD spectrum report also contains MTP rows from broad selection; final comparison excludes MTP.', 'No matched full baseline/seed/topology optimization screen or real-text activation/joint comparison completed.', '69 completed FP32 masters; whole-model FP32 intervention disallowed.', 'No recovery/expansion or long-context evaluation: quality gate not passed.'] |
| verification.pytest_passed | 339 |
| verification.pytest_skipped | 6 |
| verification.pytest_deselected | 1 |
| verification.ruff | passed |
| verification.joint_runner_integration | passed |
| verification.independent_tiny_oracle_max_abs | 1.02696e-15 |
| verification.joint_tiny_smoke_initial_relL2 | 3.56736 |
| verification.joint_tiny_smoke_final_relL2 | 0.948312 |

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

[INFERENCE] approximation/fitting damage dominates sampled layer0 intervention; serialization contributes but is not necessary for severe attenuation. Optimization vs capacity vs projection-local objective unresolved. Execution not fully tolerance/GPU-qualified.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
