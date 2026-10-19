# conversion: qwen3.5-0.8b / algebraic-ga-rotors / unrestricted

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__algebraic-ga-rotors__unrestricted__20261011T040143Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-11T04:01:43+00:00 |
| Kind | conversion |
| Model | qwen3.5-0.8b |
| Method | algebraic-ga-rotors |
| Compression target | unrestricted |

## Summary

Algebraically converted and verified 1 explicitly selected tensors to GA rotors; no training, dense runtime fallback, or byte-budget objective.

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| tensors.model.language_model.layers.0.mlp.up_proj.weight.weight.relative_l2 | 8.15616e-15 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.weight.max_absolute | 5.10009e-16 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.weight.reference_norm | 16.4287 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.weight.residual_norm | 1.33995e-13 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.direct_action.relative_l2 | 8.27571e-15 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.direct_action.max_absolute | 9.92088e-15 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.direct_action.reference_norm | 66.4313 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.direct_action.residual_norm | 5.49767e-13 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.passed | True |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.exact_full_rank | True |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.max_relative_error | 1e-10 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.elementwise_gate.atol | 1e-10 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.elementwise_gate.rtol | 1e-10 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.seed | 0 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.probe_rows | 16 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.basis_chunk | 256 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.probes_sha256 | 05aafd6fcb8c12253ece60016da827acf5858b73ed738b6bd0ab0b34dcc2d3ab |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.conversion_and_verification_seconds | 70.8247 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.source_dtype | torch.bfloat16 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.source_tensor_sha256 | 1c576515da6b8e11e78da098ed7425d7f5cb798d9e526189b5b757993ce3a427 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.source_scalars | 3670016 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.stored_scalars | 3672064 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.scalar_ratio | 1.00056 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.source_tensor_bytes | 7340032 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.stored_tensor_bytes | 29376512 |
| tensors.model.language_model.layers.0.mlp.up_proj.weight.compactness.tensor_byte_ratio | 4.00223 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.weight.relative_l2 | 8.15616e-15 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.weight.max_absolute | 5.10009e-16 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.weight.reference_norm | 16.4287 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.weight.residual_norm | 1.33995e-13 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.direct_action.relative_l2 | 8.27571e-15 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.direct_action.max_absolute | 9.92088e-15 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.direct_action.reference_norm | 66.4313 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.direct_action.residual_norm | 5.49767e-13 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.passed | True |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.exact_full_rank | True |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.max_relative_error | 1e-10 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.elementwise_gate.atol | 1e-10 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.elementwise_gate.rtol | 1e-10 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.seed | 0 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.probe_rows | 16 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.basis_chunk | 256 |
| export_load.model.language_model.layers.0.mlp.up_proj.weight.probes_sha256 | 05aafd6fcb8c12253ece60016da827acf5858b73ed738b6bd0ab0b34dcc2d3ab |
| checkpoint | /mnt/NAS/MediaHome/Local-Models/temp/qwen35-l0-up-algebraic-ga-angles-f64-proof |
| serialized.checkpoint_bytes | 29382180 |
| serialized.manifest_bytes | 4836 |
| serialized.manifest_overhead_bytes | 4836 |
| serialized.num_scalars | 3672064 |
| serialized.num_tensors | 5 |
| serialized.padding_bytes | 0 |
| serialized.safetensors_header_bytes | 824 |
| serialized.tensor_payload_bytes | 29376512 |
| serialized.weights_file_bytes | 29377344 |
| serialized.weights_sha256 | 27190bd02ed54b7a7979fe979bbd236638b589d5267205d973086bdb8161bf5e |
| provenance.path | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| provenance.hf_id | Qwen/Qwen3.5-0.8B-Base |
| provenance.revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| provenance.file_sha256.model.safetensors-00001-of-00001.safetensors | c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c |
| provenance.file_sha256.config.json | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

_(none)_

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.07 GiB |
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

Full-rank algebraic correctness is separate from compression or performance. Explicit rank truncation, if requested, is deterministic SVD approximation with measured error.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
