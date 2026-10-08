# Experiment configs

Each `*.yaml` here describes one conversion run: which model, which FFN projections, the byte budget, the rotor cascade topology, the fit objective and its schedule, and the calibration inputs. Model keys resolve through `configs/models.yaml`; storage paths resolve only through `rotor.storage`.

## Schema

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Derived output directory name (under the rotor-derived root). |
| `model` | string | Key in `configs/models.yaml` (e.g. `qwen3.5-0.8b`). |
| `families` | list | Projection families to convert. Only `ffn.gate`, `ffn.up`, `ffn.down` are allowed. |
| `layers` | null or list[int] | `null` = all layers; otherwise explicit layer indices. |
| `budget.byte_ratio` | float | Structured stored bytes / dense bytes per projection. Dense bytes use the checkpoint dtype (BF16 = 2 B). |
| `budget.storage_dtype` | string | Storage dtype of the structured parameters (`bfloat16`, `float16`, `float32`). |
| `cascade.topology` | `{kind, seed}` | Givens pairing topology: `brick`, `butterfly`, or `seeded`; `seed` used by `seeded`. |
| `cascade.stages_per_rotor` | int or `auto` | Stages per rotor; `auto` = `max(1, ceil(log2(width)))`. |
| `cascade.residual_rank` | int | Rank of the optional low-rank residual `U V` (0 = none). |
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
