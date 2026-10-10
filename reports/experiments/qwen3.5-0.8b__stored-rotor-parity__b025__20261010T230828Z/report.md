# diagnosis: qwen3.5-0.8b / stored-rotor-parity / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__stored-rotor-parity__b025__20261010T230828Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:08:28+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | stored-rotor-parity |
| Compression target | b025 |

## Summary

Stored tensor finiteness, loader equality, and bounded dense-action parity checks.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| loaded_parameter_mismatches | [] |
| operator_checks | [{'module': 'model.layers.0.mlp.up_proj', 'output_abs_max': 1.3824540376663208, 'finite': True, 'dense_action_max_abs_error': 8.195638656616211e-07, 'output_rms': 0.19259463250637054}, {'module': 'model.layers.0.mlp.down_proj', 'output_abs_max': 3.918875217437744, 'finite': True, 'dense_action_max_abs_error': 9.5367431640625e-07, 'output_rms': 0.3697752356529236}, {'module': 'model.layers.23.mlp.up_proj', 'output_abs_max': 1.898498296737671, 'finite': True, 'dense_action_max_abs_error': 1.7136335372924805e-07, 'output_rms': 0.24491643905639648}] |

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
| git | 458a96e35541d2de7a9303732e4e405fdce76150 dirty=True |

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
