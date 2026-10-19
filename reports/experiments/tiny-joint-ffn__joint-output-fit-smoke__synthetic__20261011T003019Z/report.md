# diagnosis: tiny-joint-ffn / joint-output-fit-smoke / synthetic

| Field | Value |
|---|---|
| Run id | `tiny-joint-ffn__joint-output-fit-smoke__synthetic__20261011T003019Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T00:30:19+00:00 |
| Kind | diagnosis |
| Model | tiny-joint-ffn |
| Method | joint-output-fit-smoke |
| Compression target | synthetic |

## Summary

_(none provided)_

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| initial_heldout_relative_l2 | 3.56736 |
| final_heldout_relative_l2 | 0.948312 |
| fit.objective | joint_ffn_output |
| fit.final_loss | 0.49732 |
| fit.train_output_relative_error | 0.49732 |
| fit.heldout_output_relative_error | 0.899296 |
| fit.loss_history_tail | [10 items] |
| fit.steps | 40 |
| fit.per_projection_params.gate.angles | 14 |
| fit.per_projection_params.gate.diagonals | 16 |
| fit.per_projection_params.gate.residual | 0 |
| fit.per_projection_params.gate.bias | 0 |
| fit.per_projection_params.up.angles | 14 |
| fit.per_projection_params.up.diagonals | 16 |
| fit.per_projection_params.up.residual | 0 |
| fit.per_projection_params.up.bias | 0 |
| fit.per_projection_params.down.angles | 14 |
| fit.per_projection_params.down.diagonals | 16 |
| fit.per_projection_params.down.residual | 0 |
| fit.per_projection_params.down.bias | 0 |
| fit.per_projection_bytes.gate | 60 |
| fit.per_projection_bytes.up | 60 |
| fit.per_projection_bytes.down | 60 |
| fit.per_projection_dense_bytes.gate | 96 |
| fit.per_projection_dense_bytes.up | 96 |
| fit.per_projection_dense_bytes.down | 96 |
| fit.combined_bytes | 180 |
| fit.combined_dense_bytes | 288 |
| fit.combined_byte_budget |  |
| fit.combined_byte_ratio | 0.625 |
| fit.storage_dtype | bfloat16 |

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

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
