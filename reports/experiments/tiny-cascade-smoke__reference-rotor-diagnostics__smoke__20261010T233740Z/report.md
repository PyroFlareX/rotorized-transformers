# Collapse diagnosis: tiny-cascade-smoke

| Field | Value |
|---|---|
| Run id | `tiny-cascade-smoke__reference-rotor-diagnostics__smoke__20261010T233740Z` |
| Status | **failed** |
| Created (UTC) | 2026-10-10T23:37:40+00:00 |
| Kind | collapse-diagnosis |
| Model | tiny-cascade-smoke |
| Method | reference-rotor-diagnostics |
| Compression target | smoke |

## Summary

_(none provided)_

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

_(no measured metrics)_

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

```
Traceback (most recent call last):
  File "/mnt/Projects/rotorized-transformers/src/rotor/evaluation/collapse.py", line 359, in run_collapse_diagnostics
    _run_smoke(run, args)
  File "/mnt/Projects/rotorized-transformers/src/rotor/evaluation/collapse.py", line 329, in _run_smoke
    spec_layer.residual.U.normal_(0.0, 0.1, generator=generator)
    ^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'U'

```

## Interpretation

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
