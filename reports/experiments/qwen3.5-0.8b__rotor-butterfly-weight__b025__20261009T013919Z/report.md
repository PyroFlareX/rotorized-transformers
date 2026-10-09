# convert: qwen3.5-0.8b / rotor-butterfly-weight / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-butterfly-weight__b025__20261009T013919Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T01:39:19+00:00 |
| Kind | convert |
| Model | qwen3.5-0.8b |
| Method | rotor-butterfly-weight |
| Compression target | b025 |

## Summary

Converted 3 projections (ffn.down, ffn.gate, ffn.up) of `qwen3.5-0.8b` at target `b025` (byte ratio 0.25). Serialized weights file 18384 B vs 73728 B dense converted weights (file ratio 0.2493). Output: `/tmp/rotor-prompt-smoke-2y25zy0p/derived/converted`.

## Per-layer conversion

| module | family | in | out | dense_dtype | dense_bytes | num_rotors | structured_bytes | byte_ratio | angles | scales | residual | rel_fro | fit_wall_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model.layers.0.mlp.down_proj | ffn.down | 96 | 64 | F32 | 24576 | 7 | 5728 | 0.233073 | 2128 | 736 | 0 | 1.29225 | 0.782733 |
| model.layers.0.mlp.gate_proj | ffn.gate | 64 | 96 | F32 | 24576 | 7 | 5728 | 0.233073 | 2128 | 736 | 0 | 1.28648 | 0.0680827 |
| model.layers.0.mlp.up_proj | ffn.up | 64 | 96 | F32 | 24576 | 7 | 5728 | 0.233073 | 2128 | 736 | 0 | 1.29885 | 0.0300987 |

## Byte accounting

| item | bytes |
|---|---|
| original checkpoint (sum of *.safetensors) | 1182160 |
| converted tensors, dense (checkpoint dtype) | 73728 |
| converted tensors, serialized tensor payload | 17184 |
| rotor_weights.safetensors on disk | 18384 |
|   safetensors header | 1192 |
|   alignment padding | 0 |
| rotor_manifest.json | 9638 |
| index bytes (deterministic topology) | 0 |
| quantization metadata (float storage) | 0 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| storage.original_checkpoint_bytes | 1182160 |
| storage.converted_dense_bytes | 73728 |
| storage.structured_tensor_bytes | 17184 |
| storage.layer_byte_ratio | 0.233073 |
| storage.derived_weights_file_bytes | 18384 |
| storage.derived_manifest_bytes | 9638 |
| storage.derived_checkpoint_bytes | 28022 |
| storage.file_compression_ratio | 0.249349 |
| storage.safetensors_header_bytes | 1192 |
| storage.padding_bytes | 0 |
| storage.learned_scalar_count.angles | 6384 |
| storage.learned_scalar_count.scales | 2208 |
| storage.learned_scalar_count.residual | 0 |
| storage.learned_scalar_count.bias | 0 |
| storage.learned_scalar_count.total | 8592 |
| storage.index_bytes | 0 |
| storage.quantization_metadata_bytes | 0 |
| resources.wall_seconds | 0.960146 |
| resources.cpu_seconds | 1.17401 |
| resources.peak_rss_bytes | 539336704 |
| resources.peak_device_bytes |  |
| resources.measured | True |
| resources.scope | base-shard hashing, calibration capture and all layer fits; excludes the final write |
| resources.peak_rss_note | process high-water mark (getrusage), an upper bound for this region |
| write_wall_seconds | 0.00298097 |
| layers | [{'module': 'model.layers.0.mlp.down_proj', 'family': 'ffn.down', 'in': 96, 'out': 64, 'dense_dtype': 'F32', 'dense_params': 6144, 'dense_bytes': 24576, 'num_rotors': 7, 'structured_bytes': 5728, 'byte_ratio': 0.23307291666666666, 'angles': 2128, 'scales': 736, 'residual': 0, 'bias': 0, 'rel_fro': 1.2922492871101285, 'fit_wall_seconds': 0.7827327560007689}, {'module': 'model.layers.0.mlp.gate_proj', 'family': 'ffn.gate', 'in': 64, 'out': 96, 'dense_dtype': 'F32', 'dense_params': 6144, 'dense_bytes': 24576, 'num_rotors': 7, 'structured_bytes': 5728, 'byte_ratio': 0.23307291666666666, 'angles': 2128, 'scales': 736, 'residual': 0, 'bias': 0, 'rel_fro': 1.28648127823987, 'fit_wall_seconds': 0.06808268100212445}, {'module': 'model.layers.0.mlp.up_proj', 'family': 'ffn.up', 'in': 64, 'out': 96, 'dense_dtype': 'F32', 'dense_params': 6144, 'dense_bytes': 24576, 'num_rotors': 7, 'structured_bytes': 5728, 'byte_ratio': 0.23307291666666666, 'angles': 2128, 'scales': 736, 'residual': 0, 'bias': 0, 'rel_fro': 1.2988531073261267, 'fit_wall_seconds': 0.03009866299908026}] |
| quality.mean_weight_rel_fro | 1.29253 |
| quality.max_weight_rel_fro | 1.29885 |
| quality.min_weight_rel_fro | 1.28648 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

| key | value |
|---|---|
| whole_model_bytes_if_unconverted_tensors_kept | 1136454 |
| whole_model_ratio_if_unconverted_tensors_kept | 0.961337 |
| note | arithmetic on measured byte counts: original checkpoint with the converted tensors replaced by the derived file; not a measured file |
| flops_per_token_total | 40512 |
| flops_per_token_dense_total | 36864 |

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.79 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.14.1+cpu cuda=False |
| git | ada677fe96ec412a1c1af45312c00bf875e03c60 dirty=True |

Validity warnings for this host:

- torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host
- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.8 GiB): memory-heavy timings may be distorted
- git working tree is dirty: results are not tied to a clean commit

## Failures

None.

## Interpretation

Only weight-space (and, for the activation objective, layer-input) reconstruction errors of the fitted projections are reported here; no model-quality (perplexity/task) result is implied. Compare against equal-byte baselines (truncated SVD etc.) before attributing any quality to the rotor structure.

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
