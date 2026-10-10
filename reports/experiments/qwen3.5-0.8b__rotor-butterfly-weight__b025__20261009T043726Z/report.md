# convert: qwen3.5-0.8b / rotor-butterfly-weight / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-butterfly-weight__b025__20261009T043726Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T04:37:26+00:00 |
| Kind | convert |
| Model | qwen3.5-0.8b |
| Method | rotor-butterfly-weight |
| Compression target | b025 |

## Summary

Converted 1 projections (ffn.up) of `qwen3.5-0.8b` at target `b025` (byte ratio 0.25). Serialized weights file 1802184 B vs 7340032 B dense converted weights (file ratio 0.2455). Output: `/mnt/NAS/MediaHome/Local-Models/temp/l0-up-full2000-probe`.

## Per-layer conversion

| module | family | in | out | dense_dtype | dense_bytes | num_rotors | structured_bytes | byte_ratio | angles | scales | residual | rel_fro | fit_wall_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model.layers.0.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.744064 | 578.318 |

## Byte accounting

| item | bytes |
|---|---|
| original checkpoint (sum of *.safetensors) | 1746942600 |
| converted tensors, dense (checkpoint dtype) | 7340032 |
| converted tensors, serialized tensor payload | 1801728 |
| rotor_weights.safetensors on disk | 1802184 |
|   safetensors header | 448 |
|   alignment padding | 0 |
| rotor_manifest.json | 7932 |
| index bytes (deterministic topology) | 0 |
| quantization metadata (float storage) | 0 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| storage.original_checkpoint_bytes | 1746942600 |
| storage.converted_dense_bytes | 7340032 |
| storage.structured_tensor_bytes | 1801728 |
| storage.layer_byte_ratio | 0.245466 |
| storage.derived_weights_file_bytes | 1802184 |
| storage.derived_manifest_bytes | 7932 |
| storage.derived_checkpoint_bytes | 1810116 |
| storage.file_compression_ratio | 0.245528 |
| storage.safetensors_header_bytes | 448 |
| storage.padding_bytes | 0 |
| storage.learned_scalar_count.angles | 767232 |
| storage.learned_scalar_count.scales | 133632 |
| storage.learned_scalar_count.residual | 0 |
| storage.learned_scalar_count.bias | 0 |
| storage.learned_scalar_count.total | 900864 |
| storage.index_bytes | 0 |
| storage.quantization_metadata_bytes | 0 |
| resources.wall_seconds | 633.703 |
| resources.cpu_seconds | 876.131 |
| resources.peak_rss_bytes | 1516036096 |
| resources.peak_device_bytes | 1484390912 |
| resources.measured | True |
| resources.scope | base-shard hashing, calibration capture and all layer fits; excludes the final write |
| resources.peak_rss_note | process high-water mark (getrusage), an upper bound for this region |
| write_wall_seconds | 11.4197 |
| layers | [{'module': 'model.layers.0.mlp.up_proj', 'family': 'ffn.up', 'in': 1024, 'out': 3584, 'dense_dtype': 'BF16', 'dense_params': 3670016, 'dense_bytes': 7340032, 'num_rotors': 37, 'structured_bytes': 1801728, 'byte_ratio': 0.24546595982142858, 'angles': 767232, 'scales': 133632, 'residual': 0, 'bias': 0, 'rel_fro': 0.744063576997419, 'fit_wall_seconds': 578.318195246}] |
| quality.mean_weight_rel_fro | 0.744064 |
| quality.max_weight_rel_fro | 0.744064 |
| quality.min_weight_rel_fro | 0.744064 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

| key | value |
|---|---|
| whole_model_bytes_if_unconverted_tensors_kept | 1741412684 |
| whole_model_ratio_if_unconverted_tensors_kept | 0.996835 |
| note | arithmetic on measured byte counts: original checkpoint with the converted tensors replaced by the derived file; not a measured file |
| flops_per_token_total | 4737024 |
| flops_per_token_dense_total | 7340032 |

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.85 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.10.0+rocm7.0 cuda=True |
| git | 458a96e35541d2de7a9303732e4e405fdce76150 dirty=True |

Validity warnings for this host:

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
