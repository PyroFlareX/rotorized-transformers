# diagnosis: qwen3.5-0.8b / algebraic-givens-direct-smoke / unrestricted

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__algebraic-givens-direct-smoke__unrestricted__20261011T023803Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T02:38:03+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | algebraic-givens-direct-smoke |
| Compression target | unrestricted |

## Summary

Direct algebraic rotation/scaling execution matches local real-weight submatrix; no training, truncation, dense reconstruction runtime, or byte limit.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| max_absolute_output_error | 5.27356e-16 |
| relative_l2_output_error | 1.57678e-15 |
| rank | 32 |
| rotation_angles | 1504 |

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
