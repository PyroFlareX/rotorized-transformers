# Experiment configs

Each `*.yaml` here describes an experiment: model, selected FFN projections, a byte budget, topology, fit objective and schedule, and calibration inputs. Model keys resolve through `configs/models.yaml`; storage paths resolve only through `rotor.storage`. Budget experiments may use named byte-fraction presets and optionally list comparison baselines and target sweeps.

## Schema

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Derived output directory name (under the rotor-derived root). |
| `model` | string | Key in `configs/models.yaml` (e.g. `qwen3.5-0.8b`). |
| `families` | list | Projection families to convert. Only `ffn.gate`, `ffn.up`, `ffn.down` are allowed. |
| `layers` | null or list[int] | `null` = all layers; otherwise explicit layer indices. |
| `budget.byte_ratio` | float | Structured stored bytes / dense bytes per projection. Dense bytes use checkpoint dtype (BF16 = 2 B). Use this or `budget.target`, not both. |
| `budget.target` | string | Named preset for the structured-byte fraction; alternative to `byte_ratio` (exactly one is required). |
| `budget.storage_dtype` | string | Storage dtype of structured parameters (`bfloat16`, `float16`, `float32`). |
| `baselines` | map | Optional comparison/sweep configuration. `baselines.methods` lists methods to compare; `baselines.targets` lists target presets to sweep. |
| `cascade.topology` | `{kind, seed}` | Givens pairing topology: `brick`, `butterfly`, or `seeded`; `seed` used by `seeded`. |
| `cascade.stages_per_rotor` | int or `auto` | Stages per rotor; `auto` = `max(1, ceil(log2(width)))`. |
| `cascade.residual_rank` | int | Rank of optional low-rank residual `U V` (0 = none). |
| `cascade.adapter` | string | Rectangular handling; `pad_truncate`. |
| `fit.objective` | `weight` or `activation` | Weight-space Frobenius fit, or activation-space fit on calibration inputs. |
| `fit.seed` | int | Seed for initialization. |
| `fit.device` | string | `cpu` (or `cuda` if available). |
| `fit.init_angle_std` | float | Std of initial rotation angles. |
| `fit.schedule.phases` | list | Each phase: `{train: [angles|scales|residual ...], steps: N}`, run in order. |
| `fit.schedule.lrs` | map | Learning rate per parameter group. |
| `fit.schedule.cosine_decay` | bool | Cosine learning-rate decay across steps. |
| `fit.schedule.log_every` | int | Logging interval in steps. |
| `calibration.source` | `random` or `text_file` | Calibration inputs. `random` uses seeded random token windows; `text_file` tokenizes `path`. Unused for `weight`. |
| `calibration.path` | string or null | Text file for `text_file`. Must be user-approved before running. |
| `calibration.seq_len` | int | Window length in tokens. |
| `calibration.num_windows` | int | Number of windows. |
| `calibration.seed` | int | Window sampling seed. |
| `calibration.max_rows` | int | Max input rows per projection kept for activation fitting (deterministic subsampling). |

### Compression target presets

Presets denote fractions of the original stored bytes:

| Preset | Fraction |
|---|---:|
| `b075` | 0.75 |
| `b050` | 0.50 |
| `b035` | 0.35 |
| `b025` | 0.25 |
| `b015` | 0.15 |

For example, `budget: {target: b025, storage_dtype: bfloat16}` selects a 25% target. The equivalent numeric setting is `byte_ratio: 0.25`.

`baselines.methods` can name `dense`, `svd`, `svd_givens`, `magnitude_sparse`, `butterfly`, `rotor_weight`, and `rotor_residual`; target sweeps use `baselines.targets` with one or more presets from the table above. These are comparison/sweep declarations, not a claim that runs have completed.

## Running

Dry run (prints the allocation and byte plan, no fitting, nothing written):

```bash
.venv/bin/python scripts/convert_model.py --config configs/experiments/qwen3.5-0.8b-ffn-weight.yaml --dry-run
```

Full run (publishes atomically under the derived root; refuses to overwrite):

```bash
.venv/bin/python scripts/convert_model.py --config configs/experiments/qwen3.5-0.8b-ffn-weight.yaml
```

The base checkpoint must already exist at the path given in `configs/models.yaml`; the converter refuses to run otherwise and reports the HF id and revision to download. It never downloads weights.
