# convert: qwen3.5-0.8b / rotor-butterfly-weight / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-butterfly-weight__b025__20261009T043252Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T04:32:52+00:00 |
| Kind | convert |
| Model | qwen3.5-0.8b |
| Method | rotor-butterfly-weight |
| Compression target | b025 |

## Summary

Converted 3 projections (ffn.down, ffn.gate, ffn.up) of `qwen3.5-0.8b` at target `b025` (byte ratio 0.25). Serialized weights file 5406472 B vs 22020096 B dense converted weights (file ratio 0.2455). Output: `/mnt/NAS/MediaHome/Local-Models/temp/l0-gpu-probe-100steps`.

## Per-layer conversion

| module | family | in | out | dense_dtype | dense_bytes | num_rotors | structured_bytes | byte_ratio | angles | scales | residual | rel_fro | fit_wall_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model.layers.0.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.889812 | 107.321 |
| model.layers.0.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.879908 | 26.1581 |
| model.layers.0.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.891697 | 27.4967 |

## Byte accounting

| item | bytes |
|---|---|
| original checkpoint (sum of *.safetensors) | 1746942600 |
| converted tensors, dense (checkpoint dtype) | 22020096 |
| converted tensors, serialized tensor payload | 5405184 |
| rotor_weights.safetensors on disk | 5406472 |
|   safetensors header | 1280 |
|   alignment padding | 0 |
| rotor_manifest.json | 10923 |
| index bytes (deterministic topology) | 0 |
| quantization metadata (float storage) | 0 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| storage.original_checkpoint_bytes | 1746942600 |
| storage.converted_dense_bytes | 22020096 |
| storage.structured_tensor_bytes | 5405184 |
| storage.layer_byte_ratio | 0.245466 |
| storage.derived_weights_file_bytes | 5406472 |
| storage.derived_manifest_bytes | 10923 |
| storage.derived_checkpoint_bytes | 5417395 |
| storage.file_compression_ratio | 0.245524 |
| storage.safetensors_header_bytes | 1280 |
| storage.padding_bytes | 0 |
| storage.learned_scalar_count.angles | 2301696 |
| storage.learned_scalar_count.scales | 400896 |
| storage.learned_scalar_count.residual | 0 |
| storage.learned_scalar_count.bias | 0 |
| storage.learned_scalar_count.total | 2702592 |
| storage.index_bytes | 0 |
| storage.quantization_metadata_bytes | 0 |
| resources.wall_seconds | 215.726 |
| resources.cpu_seconds | 292.152 |
| resources.peak_rss_bytes | 1550254080 |
| resources.peak_device_bytes | 5269740032 |
| resources.measured | True |
| resources.scope | base-shard hashing, calibration capture and all layer fits; excludes the final write |
| resources.peak_rss_note | process high-water mark (getrusage), an upper bound for this region |
| write_wall_seconds | 9.30194 |
| layers | [{'module': 'model.layers.0.mlp.down_proj', 'family': 'ffn.down', 'in': 3584, 'out': 1024, 'dense_dtype': 'BF16', 'dense_params': 3670016, 'dense_bytes': 7340032, 'num_rotors': 37, 'structured_bytes': 1801728, 'byte_ratio': 0.24546595982142858, 'angles': 767232, 'scales': 133632, 'residual': 0, 'bias': 0, 'rel_fro': 0.8898122233946845, 'fit_wall_seconds': 107.32134542700078}, {'module': 'model.layers.0.mlp.gate_proj', 'family': 'ffn.gate', 'in': 1024, 'out': 3584, 'dense_dtype': 'BF16', 'dense_params': 3670016, 'dense_bytes': 7340032, 'num_rotors': 37, 'structured_bytes': 1801728, 'byte_ratio': 0.24546595982142858, 'angles': 767232, 'scales': 133632, 'residual': 0, 'bias': 0, 'rel_fro': 0.8799080233774852, 'fit_wall_seconds': 26.158133308999822}, {'module': 'model.layers.0.mlp.up_proj', 'family': 'ffn.up', 'in': 1024, 'out': 3584, 'dense_dtype': 'BF16', 'dense_params': 3670016, 'dense_bytes': 7340032, 'num_rotors': 37, 'structured_bytes': 1801728, 'byte_ratio': 0.24546595982142858, 'angles': 767232, 'scales': 133632, 'residual': 0, 'bias': 0, 'rel_fro': 0.8916971367889569, 'fit_wall_seconds': 27.49669205400278}] |
| quality.mean_weight_rel_fro | 0.887139 |
| quality.max_weight_rel_fro | 0.891697 |
| quality.min_weight_rel_fro | 0.879908 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

| key | value |
|---|---|
| whole_model_bytes_if_unconverted_tensors_kept | 1730339899 |
| whole_model_ratio_if_unconverted_tensors_kept | 0.990496 |
| note | arithmetic on measured byte counts: original checkpoint with the converted tensors replaced by the derived file; not a measured file |
| flops_per_token_total | 14211072 |
| flops_per_token_dense_total | 22020096 |

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.88 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.10.0+rocm7.0 cuda=True |
| git | ada677fe96ec412a1c1af45312c00bf875e03c60 dirty=True |

Validity warnings for this host:

- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.9 GiB): memory-heavy timings may be distorted
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
