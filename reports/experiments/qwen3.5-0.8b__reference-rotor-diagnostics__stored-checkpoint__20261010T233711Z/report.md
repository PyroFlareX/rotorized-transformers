# Collapse diagnosis: qwen3.5-0.8b

| Field | Value |
|---|---|
| Run id | `qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T233711Z` |
| Status | **failed** |
| Created (UTC) | 2026-10-10T23:37:11+00:00 |
| Kind | collapse-diagnosis |
| Model | qwen3.5-0.8b |
| Method | reference-rotor-diagnostics |
| Compression target | stored-checkpoint |

## Summary

_(none provided)_

_(no additional sections)_

## Measured metrics

All values below are measurements from this run (full data: `metrics.json`).

| key | value |
|---|---|
| provenance.checkpoint | /mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3 |
| provenance.checkpoint_manifest_sha256 | 8a89ec346e40a81e3bdd65f7f3f585d8b48f6e6581e1adc81526da8dc51d4cba |
| provenance.base_model_path | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| provenance.base_config_sha256 | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |
| provenance.base_hf_id | Qwen/Qwen3.5-0.8B-Base |
| provenance.base_revision | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |
| provenance.tokenizer_name | /mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base |
| provenance.tokenizer_config_sha256 | fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927 |
| provenance.prompts_path | /mnt/Projects/rotorized-transformers/configs/evaluation/prompts-256.jsonl |
| provenance.prompts_sha256 | 73036763ff58b3eb3f95ffc4dc05dbc79ce3604ee2c6e284ef2ac381fe5e6641 |
| provenance.prompt_fingerprints.science-001 | c68bb9a2b044f2e274f57adca453f9a24c991f60d1bd0461bce7f1b09669a9c9 |
| provenance.seed | 0 |
| provenance.device | cpu |
| provenance.dtype | float32 |
| provenance.rotor_backend | reference |
| provenance.snapshot.path | /mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3.snapshots/snapshot-1791623497413929547-49eb563c4c464f4699da11f2e945cf3e |
| provenance.snapshot.identity_sha256 | ad41cb8f75cf40e461edc28ae0ec1f41b73b371861fda23c135aae4fe7a1c6aa |
| provenance.snapshot.completed_matching_projections | [69 items] |
| provenance.approved_heldout_corpus |  |
| provenance.heldout_corpus_sha256 |  |
| corpus_status | no approved held-out corpus configured; prompts are diagnostic inputs, not held-out benchmark data |
| stored_projection_count | 72 |
| fp32_master_projection_count | 69 |
| snapshot.path | /mnt/NAS/MediaHome/Local-Models/temp/qwen3.5-0.8b-full-ffn-b025-gpu-r3.snapshots/snapshot-1791623497413929547-49eb563c4c464f4699da11f2e945cf3e |
| snapshot.identity_sha256 | ad41cb8f75cf40e461edc28ae0ec1f41b73b371861fda23c135aae4fe7a1c6aa |
| snapshot.completed_matching_projections | [69 items] |
| heldout_nll.status | not measured: no approved corpus supplied |
| operator_parity.model.layers.0.mlp.down_proj.elements | 8192 |
| operator_parity.model.layers.0.mlp.down_proj.finite_reference | 8192 |
| operator_parity.model.layers.0.mlp.down_proj.finite_actual | 8192 |
| operator_parity.model.layers.0.mlp.down_proj.finite_pairs | 8192 |
| operator_parity.model.layers.0.mlp.down_proj.max_abs_error | 4.35233e-06 |
| operator_parity.model.layers.0.mlp.down_proj.mean_abs_error | 5.21157e-07 |
| operator_parity.model.layers.0.mlp.down_proj.max_relative_error | 0.0160564 |
| operator_parity.model.layers.0.mlp.down_proj.relative_l2_error | 1.97737e-06 |
| operator_parity.model.layers.0.mlp.down_proj.cosine_similarity | 1 |
| operator_parity.model.layers.0.mlp.down_proj.reference_rms | 0.353557 |
| operator_parity.model.layers.0.mlp.down_proj.actual_rms | 0.353557 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_reference.min | 0.341975 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_reference.p50 | 0.35252 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_reference.p90 | 0.364559 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_reference.p99 | 0.36998 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_reference.max | 0.370583 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_actual.min | 0.341975 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_actual.p50 | 0.35252 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_actual.p90 | 0.364559 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_actual.p99 | 0.36998 |
| operator_parity.model.layers.0.mlp.down_proj.row_rms_actual.max | 0.370583 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_reference.min | 0.00948789 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_reference.p50 | 0.330671 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_reference.p90 | 0.487563 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_reference.p99 | 0.625814 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_reference.max | 1.56566 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_actual.min | 0.00948789 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_actual.p50 | 0.330672 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_actual.p90 | 0.487563 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_actual.p99 | 0.625814 |
| operator_parity.model.layers.0.mlp.down_proj.channel_rms_actual.max | 1.56566 |
| operator_parity.model.layers.0.mlp.down_proj.nonfinite_reference | 0 |
| operator_parity.model.layers.0.mlp.down_proj.nonfinite_actual | 0 |
| operator_parity.model.layers.0.mlp.down_proj.finite_pair_matrix | 8192 |
| operator_parity.model.layers.0.mlp.down_proj.dense_basis_chunk | 64 |
| operator_parity.model.layers.0.mlp.down_proj.dense_max_temporary_basis_elements | 229376 |
| operator_parity.model.layers.0.mlp.down_proj.weight_relative_frobenius_error | 0.729152 |
| operator_parity.model.layers.0.mlp.down_proj.weight_max_abs_error | 0.104605 |
| operator_parity.model.layers.0.mlp.down_proj.weight_reference_frobenius | 16.3108 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.elements | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.finite_reference | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.finite_actual | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.finite_pairs | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.max_abs_error | 0.182462 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.mean_abs_error | 0.0123461 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.max_relative_error | 13437.1 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.relative_l2_error | 0.705951 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.cosine_similarity | 0.708333 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.reference_rms | 0.0225545 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.actual_rms | 0.0157485 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_reference.min | 0.0137982 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_reference.p50 | 0.0209988 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_reference.p90 | 0.0296076 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_reference.p99 | 0.0326595 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_reference.max | 0.0338055 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_actual.min | 0.00854685 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_actual.p50 | 0.0150951 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_actual.p90 | 0.0200293 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_actual.p99 | 0.0241606 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.row_rms_actual.max | 0.0249575 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_reference.min | 0.00825493 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_reference.p50 | 0.0208273 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_reference.p90 | 0.0258858 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_reference.p99 | 0.0307209 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_reference.max | 0.243019 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_actual.min | 0.000402995 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_actual.p50 | 0.0137722 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_actual.p90 | 0.0181938 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_actual.p99 | 0.0221027 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.channel_rms_actual.max | 0.228461 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.nonfinite_reference | 0 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.nonfinite_actual | 0 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_output_error.finite_pair_matrix | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.random_probe_rows | 16 |
| operator_parity.model.layers.0.mlp.down_proj.teacher_activation_rows | 33 |
| operator_parity.model.layers.0.mlp.down_proj.family | ffn.down |
| operator_parity.model.layers.0.mlp.down_proj.layer | 0 |
| operator_parity.model.layers.0.mlp.down_proj.spec.in_features | 3584 |
| operator_parity.model.layers.0.mlp.down_proj.spec.out_features | 1024 |
| operator_parity.model.layers.0.mlp.down_proj.spec.num_rotors | 37 |
| operator_parity.model.layers.0.mlp.down_proj.spec.stages_per_rotor | 12 |
| operator_parity.model.layers.0.mlp.down_proj.spec.topology.kind | butterfly |
| operator_parity.model.layers.0.mlp.down_proj.spec.topology.seed | 0 |
| operator_parity.model.layers.0.mlp.down_proj.spec.adapter | pad_truncate |
| operator_parity.model.layers.0.mlp.down_proj.spec.residual_rank | 0 |
| operator_parity.model.layers.0.mlp.down_proj.loaded_parameter_dtype | torch.float32 |
| operator_parity.model.layers.0.mlp.down_proj.stored_storage_dtype | bfloat16 |
| operator_parity.model.layers.0.mlp.down_proj.serialized_vs_base_weight |  |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.elements | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.finite_reference | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.finite_actual | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.finite_pairs | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.max_abs_error | 0.00354633 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.mean_abs_error | 0.000147218 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.max_relative_error | 283.398 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.relative_l2_error | 0.0131798 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.cosine_similarity | 0.999933 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.reference_rms | 0.0157221 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.actual_rms | 0.0156211 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_reference.min | 0.0085224 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_reference.p50 | 0.0150735 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_reference.p90 | 0.020016 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_reference.p99 | 0.0241053 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_reference.max | 0.0249018 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_actual.min | 0.00848436 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_actual.p50 | 0.014957 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_actual.p90 | 0.0198917 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_actual.p99 | 0.0239738 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.row_rms_actual.max | 0.0247734 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_reference.min | 0.000402672 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_reference.p50 | 0.0137688 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_reference.p90 | 0.0181829 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_reference.p99 | 0.0221816 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_reference.max | 0.226954 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_actual.min | 0.000400701 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_actual.p50 | 0.0136693 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_actual.p90 | 0.0180571 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_actual.p99 | 0.0220033 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.channel_rms_actual.max | 0.225679 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.nonfinite_reference | 0 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.nonfinite_actual | 0 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.fp16.finite_pair_matrix | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.elements | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.finite_reference | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.finite_actual | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.finite_pairs | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.max_abs_error | 0.0197768 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.mean_abs_error | 0.00114238 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.max_relative_error | 2951.68 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.relative_l2_error | 0.101839 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.cosine_similarity | 0.995932 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.reference_rms | 0.0157221 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.actual_rms | 0.014912 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_reference.min | 0.0085224 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_reference.p50 | 0.0150735 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_reference.p90 | 0.020016 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_reference.p99 | 0.0241053 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_reference.max | 0.0249018 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_actual.min | 0.00809752 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_actual.p50 | 0.0141417 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_actual.p90 | 0.0189865 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_actual.p99 | 0.0229016 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.row_rms_actual.max | 0.0237622 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_reference.min | 0.000402672 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_reference.p50 | 0.0137688 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_reference.p90 | 0.0181829 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_reference.p99 | 0.0221816 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_reference.max | 0.226954 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_actual.min | 0.000375699 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_actual.p50 | 0.0130012 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_actual.p90 | 0.0171409 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_actual.p99 | 0.0207765 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.channel_rms_actual.max | 0.219645 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.nonfinite_reference | 0 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.nonfinite_actual | 0 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.bfloat16.finite_pair_matrix | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.elements | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.finite_reference | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.finite_actual | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.finite_pairs | 33792 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.max_abs_error | 0.00443462 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.mean_abs_error | 0.000249616 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.max_relative_error | 595.81 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.relative_l2_error | 0.022271 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.cosine_similarity | 0.999754 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.reference_rms | 0.0157221 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.actual_rms | 0.0157485 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_reference.min | 0.0085224 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_reference.p50 | 0.0150735 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_reference.p90 | 0.020016 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_reference.p99 | 0.0241053 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_reference.max | 0.0249018 |
| operator_parity.model.layers.0.mlp.down_proj.serialization_comparison.serialized_vs_fp32_master.row_rms_actual.min | 0.00854685 |

_17 more keys in metrics.json_

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
  File "/mnt/Projects/rotorized-transformers/src/rotor/evaluation/collapse.py", line 470, in run_collapse_diagnostics
    torch.cat([torch.zeros_like(random_probe[:, : entry.spec.out_features]), all_teacher_proj_outputs[module]], dim=0),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 1024 but got size 3584 for tensor number 1 in the list.

```

## Interpretation

_(not yet interpreted)_

## Figures

_(none)_

## Logs and artifacts

- `logs/run.log`
- /mnt/Projects/rotorized-transformers/reports/prompt-evaluations/qwen3.5-0.8b__reference-rotor-diagnostics__stored-checkpoint__20261010T233711Z.jsonl
- `config.yaml` — resolved configuration
- `environment.json` — full hardware/software probe
- `metrics.json` — machine-readable results
