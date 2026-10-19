# diagnosis: qwen3.5-0.8b / collapse-svd-capacity-screen / b025-b075

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__collapse-svd-capacity-screen__b025-b075__20261010T232447Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:24:47+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | collapse-svd-capacity-screen |
| Compression target | b025-b075 |

## Summary

Measured singular spectrum of all three layer-0 FFN matrices; SVD factor payload accounting only, no serialized dtype simulation or held-out quality.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| rows | [18 items] |

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

These weight-space lower bounds cannot establish preserved composed FFN or language behavior.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
