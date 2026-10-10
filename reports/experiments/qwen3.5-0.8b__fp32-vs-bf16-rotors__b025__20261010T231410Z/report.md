# diagnosis: qwen3.5-0.8b / fp32-vs-bf16-rotors / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__fp32-vs-bf16-rotors__b025__20261010T231410Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:14:10+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | fp32-vs-bf16-rotors |
| Compression target | b025 |

## Summary

Isolating serialization error from fit error on identical random activation probes; not language-quality evaluation.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| projections | [{'module': 'model.layers.0.mlp.up_proj', 'serialization_output_relative_error': 0.024189544841647148, 'max_abs_error': 0.03938949108123779, 'master_rms': 0.18510207533836365, 'stored_rms': 0.1851869374513626}, {'module': 'model.layers.0.mlp.down_proj', 'serialization_output_relative_error': 0.025139568373560905, 'max_abs_error': 0.08172118663787842, 'master_rms': 0.3532646596431732, 'stored_rms': 0.3535565733909607}, {'module': 'model.layers.12.mlp.up_proj', 'serialization_output_relative_error': 0.02937217243015766, 'max_abs_error': 0.09107768535614014, 'master_rms': 0.18624769151210785, 'stored_rms': 0.18624676764011383}, {'module': 'model.layers.22.mlp.up_proj', 'serialization_output_relative_error': 0.02630573697388172, 'max_abs_error': 0.05212794989347458, 'master_rms': 0.2256893515586853, 'stored_rms': 0.2255914956331253}] |

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
