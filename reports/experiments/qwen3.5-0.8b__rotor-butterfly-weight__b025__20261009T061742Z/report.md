# convert: qwen3.5-0.8b / rotor-butterfly-weight / b025

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__rotor-butterfly-weight__b025__20261009T061742Z` |
| Status | **completed** |
| Created (UTC) | 2026-10-09T06:17:42+00:00 |
| Kind | convert |
| Model | qwen3.5-0.8b |
| Method | rotor-butterfly-weight |
| Compression target | b025 |

## Summary

Converted 72 projections (ffn.down, ffn.gate, ffn.up) of `qwen3.5-0.8b` at target `b025` (byte ratio 0.25). Serialized weights file 129754992 B vs 528482304 B dense converted weights (file ratio 0.2455). Output: `/mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3`.

## Per-layer conversion

| module | family | in | out | dense_dtype | dense_bytes | num_rotors | structured_bytes | byte_ratio | angles | scales | residual | rel_fro | fit_wall_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model.layers.0.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.72895 | 2255.99 |
| model.layers.0.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.726859 | 542.887 |
| model.layers.0.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.744064 | 528.097 |
| model.layers.1.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.732465 | 2055.89 |
| model.layers.1.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.712828 | 500.491 |
| model.layers.1.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.740121 | 496.845 |
| model.layers.2.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.710423 | 2058.91 |
| model.layers.2.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.686723 | 496.626 |
| model.layers.2.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.743315 | 506.363 |
| model.layers.3.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.708112 | 2060.14 |
| model.layers.3.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.677871 | 497.089 |
| model.layers.3.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.736374 | 497.041 |
| model.layers.4.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.709869 | 2248.3 |
| model.layers.4.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.667291 | 951.167 |
| model.layers.4.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.726946 | 934.593 |
| model.layers.5.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.687367 | 2743.42 |
| model.layers.5.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.671154 | 520.519 |
| model.layers.5.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.723483 | 497.013 |
| model.layers.6.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.688046 | 2068.49 |
| model.layers.6.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.647385 | 497.136 |
| model.layers.6.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.698271 | 505.866 |
| model.layers.7.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.697671 | 2076.15 |
| model.layers.7.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.668968 | 497.2 |
| model.layers.7.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.712341 | 497.098 |
| model.layers.8.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.683953 | 2073.49 |
| model.layers.8.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.65474 | 506.503 |
| model.layers.8.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.713272 | 497.394 |
| model.layers.9.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.680467 | 2075.91 |
| model.layers.9.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.662515 | 497.122 |
| model.layers.9.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.696771 | 517.349 |
| model.layers.10.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.671878 | 2067.73 |
| model.layers.10.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.628204 | 497.057 |
| model.layers.10.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.70061 | 497.827 |
| model.layers.11.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.670966 | 2089.51 |
| model.layers.11.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.646225 | 497.639 |
| model.layers.11.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.70689 | 497.554 |
| model.layers.12.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.68932 | 2081.04 |
| model.layers.12.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.621485 | 520.334 |
| model.layers.12.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.706061 | 497.265 |
| model.layers.13.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.681214 | 2083.37 |
| model.layers.13.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.626269 | 497.059 |
| model.layers.13.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.708687 | 513.183 |
| model.layers.14.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.694031 | 2093.5 |
| model.layers.14.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.590509 | 507.047 |
| model.layers.14.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.708093 | 507.345 |
| model.layers.15.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.698946 | 2136.99 |
| model.layers.15.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.619045 | 541.125 |
| model.layers.15.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.709484 | 518.378 |
| model.layers.16.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.70472 | 2142.23 |
| model.layers.16.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.64341 | 508.768 |
| model.layers.16.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.716484 | 541.011 |
| model.layers.17.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.705694 | 2130.72 |
| model.layers.17.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.664231 | 508.57 |
| model.layers.17.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.722371 | 530.455 |
| model.layers.18.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.702027 | 2143.47 |
| model.layers.18.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.671672 | 507.99 |
| model.layers.18.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.728452 | 508.405 |
| model.layers.19.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.723765 | 2173.84 |
| model.layers.19.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.657079 | 508.993 |
| model.layers.19.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.716235 | 508.97 |
| model.layers.20.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.723874 | 2139.6 |
| model.layers.20.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.669667 | 569.433 |
| model.layers.20.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.715792 | 508.43 |
| model.layers.21.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.717012 | 2127.37 |
| model.layers.21.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.679409 | 507.633 |
| model.layers.21.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.718454 | 531.681 |
| model.layers.22.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.708129 | 2147.74 |
| model.layers.22.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.655016 | 508.052 |
| model.layers.22.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.721454 | 508.116 |
| model.layers.23.mlp.down_proj | ffn.down | 3584 | 1024 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.71207 | 2165.46 |
| model.layers.23.mlp.gate_proj | ffn.gate | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.678303 | 507.767 |
| model.layers.23.mlp.up_proj | ffn.up | 1024 | 3584 | BF16 | 7340032 | 37 | 1801728 | 0.245466 | 767232 | 133632 | 0 | 0.695338 | 507.819 |

## Byte accounting

| item | bytes |
|---|---|
| original checkpoint (sum of *.safetensors) | 1746942600 |
| converted tensors, dense (checkpoint dtype) | 528482304 |
| converted tensors, serialized tensor payload | 129724416 |
| rotor_weights.safetensors on disk | 129754992 |
|   safetensors header | 30568 |
|   alignment padding | 0 |
| rotor_manifest.json | 112625 |
| index bytes (deterministic topology) | 0 |
| quantization metadata (float storage) | 0 |

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| storage.original_checkpoint_bytes | 1746942600 |
| storage.converted_dense_bytes | 528482304 |
| storage.structured_tensor_bytes | 129724416 |
| storage.layer_byte_ratio | 0.245466 |
| storage.derived_weights_file_bytes | 129754992 |
| storage.derived_manifest_bytes | 112625 |
| storage.derived_checkpoint_bytes | 129867617 |
| storage.file_compression_ratio | 0.245524 |
| storage.safetensors_header_bytes | 30568 |
| storage.padding_bytes | 0 |
| storage.learned_scalar_count.angles | 55240704 |
| storage.learned_scalar_count.scales | 9621504 |
| storage.learned_scalar_count.residual | 0 |
| storage.learned_scalar_count.bias | 0 |
| storage.learned_scalar_count.total | 64862208 |
| storage.index_bytes | 0 |
| storage.quantization_metadata_bytes | 0 |
| resources.wall_seconds | 76845.8 |
| resources.cpu_seconds | 143170 |
| resources.peak_rss_bytes | 2543779840 |
| resources.peak_device_bytes | 5269740032 |
| resources.measured | True |
| resources.scope | base-shard hashing, calibration capture and all layer fits; excludes the final write |
| resources.peak_rss_note | process high-water mark (getrusage), an upper bound for this region |
| write_wall_seconds | 21.3644 |
| layers | [72 items] |
| quality.mean_weight_rel_fro | 0.6926 |
| quality.max_weight_rel_fro | 0.744064 |
| quality.min_weight_rel_fro | 0.590509 |

## Theoretical estimates

Formula-derived values. These are **not measurements**.

| key | value |
|---|---|
| whole_model_bytes_if_unconverted_tensors_kept | 1348327913 |
| whole_model_ratio_if_unconverted_tensors_kept | 0.771822 |
| note | arithmetic on measured byte counts: original checkpoint with the converted tensors replaced by the derived file; not a measured file |
| flops_per_token_total | 341065728 |
| flops_per_token_dense_total | 528482304 |

## Environment

| item | value |
|---|---|
| host | PyrosPC |
| cpu | AMD Ryzen 5 7600X 6-Core Processor (12 threads) |
| ram | 62.0 GiB |
| swap used | 3.44 GiB |
| gpus (pci) | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (amdgpu), Raphael (amdgpu) |
| torch | 2.10.0+rocm7.0 cuda=True |
| git | 458a96e35541d2de7a9303732e4e405fdce76150 dirty=True |

Validity warnings for this host:

- nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run
- swap in use (3.4 GiB): memory-heavy timings may be distorted
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
