# diagnosis: qwen3.5-0.8b / activation-collapse-trace / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__activation-collapse-trace__b025__20261010T230910Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-10T23:09:10+00:00 |
| Kind | diagnosis |
| Model | qwen3.5-0.8b |
| Method | activation-collapse-trace |
| Compression target | b025 |

## Summary

Single matched prompt layer activation and next-token logit trace.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| trace.base.activations | [120 items] |
| trace.base.logits_finite | True |
| trace.base.top_tokens | [10 items] |
| trace.base.logits_rms | 3.52971 |
| trace.converted.activations | [120 items] |
| trace.converted.logits_finite | True |
| trace.converted.top_tokens | [10 items] |
| trace.converted.logits_rms | 5.17256 |

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
| torch | 2.10.0+rocm7.0 cuda=True |
| git | 458a96e35541d2de7a9303732e4e405fdce76150 dirty=True |

Validity warnings for this host:

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
