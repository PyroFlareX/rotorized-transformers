# Collapse diagnosis: tiny-cascade-smoke

| Field | Value |
|---|---|
| Run id | `tiny-cascade-smoke__reference-rotor-diagnostics__smoke__20261010T234153Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:41:53+00:00 |
| Kind | collapse-diagnosis |
| Model | tiny-cascade-smoke |
| Method | reference-rotor-diagnostics |
| Compression target | smoke |

## Summary

Tiny CPU-only, known-correct rectangular rotor dense-action parity harness passed.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| smoke.elements | 45 |
| smoke.finite_reference | 45 |
| smoke.finite_actual | 45 |
| smoke.finite_pairs | 45 |
| smoke.max_abs_error | 1.02696e-15 |
| smoke.mean_abs_error | 3.04386e-16 |
| smoke.max_relative_error_nonzero_reference | 6.84656e-15 |
| smoke.relative_l2_error | 3.37946e-16 |
| smoke.cosine_similarity | 1 |
| smoke.reference_rms | 1.22452 |
| smoke.actual_rms | 1.22452 |
| smoke.row_rms_reference.min | 0.501896 |
| smoke.row_rms_reference.p50 | 1.07382 |
| smoke.row_rms_reference.p90 | 1.75027 |
| smoke.row_rms_reference.p99 | 1.87926 |
| smoke.row_rms_reference.max | 1.89359 |
| smoke.row_rms_actual.min | 0.501896 |
| smoke.row_rms_actual.p50 | 1.07382 |
| smoke.row_rms_actual.p90 | 1.75027 |
| smoke.row_rms_actual.p99 | 1.87926 |
| smoke.row_rms_actual.max | 1.89359 |
| smoke.channel_rms_reference.min | 0.740249 |
| smoke.channel_rms_reference.p50 | 1.18476 |
| smoke.channel_rms_reference.p90 | 1.53517 |
| smoke.channel_rms_reference.p99 | 1.60641 |
| smoke.channel_rms_reference.max | 1.61432 |
| smoke.channel_rms_actual.min | 0.740249 |
| smoke.channel_rms_actual.p50 | 1.18476 |
| smoke.channel_rms_actual.p90 | 1.53517 |
| smoke.channel_rms_actual.p99 | 1.60641 |
| smoke.channel_rms_actual.max | 1.61432 |
| smoke.nonfinite_reference | 0 |
| smoke.nonfinite_actual | 0 |
| smoke.dense_basis_chunk | 3 |
| smoke.dense_max_temporary_basis_elements | 21 |
| smoke.harness | known-correct-rectangular-cascade-residual-bias |
| smoke.tolerance.rtol | 1e-10 |
| smoke.tolerance.atol | 1e-10 |
| smoke.tolerance.source | tests/test_math.py forward-vs-dense cascade contract |

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

Synthetic harness validation only; it does not measure model or language quality.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
