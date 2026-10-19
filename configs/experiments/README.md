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

### Periodic snapshots and resume

Conversion saves a resumable snapshot every 1,800 seconds (30 minutes), keeping the newest three.
Snapshots live beside the final output, in `<output-name>.snapshots/`, under the configured writable NAS root.
Publication is atomic; older snapshots are removed only after a new snapshot is published successfully.
The final inference checkpoint remains a separate, atomic output and is never overwritten.

Snapshots retain completed projections plus the active fit's FP32 parameters, Adam moments, schedule phase,
next step, and loss history. Resume continues the original cosine schedule and skips completed projections.
A snapshot is taken at an optimizer-step boundary; a long step can delay the interval.
Only use snapshots from trusted runs. Configuration and base-checkpoint identity must match.

```bash
.venv-rocm/bin/python scripts/convert_model.py \
  --config configs/experiments/qwen3.5-0.8b-ffn-weight.yaml --device cuda \
  --name <original-output-name> --resume <snapshot-directory> \
  --snapshot-interval-seconds 1800 --snapshot-keep 3 \
  --telemetry <new-telemetry-jsonl-path>
```

`--device` overrides `fit.device`; ROCm PyTorch uses the `cuda` device spelling.
Use a fresh telemetry path on every launch. Snapshots are recovery artifacts, not loadable inference checkpoints.

### Baseline execution device

`scripts/compare_baselines.py --device cuda:0` selects the device for rotor fitting and teacher activation capture.
Analytic SVD/Givens/sparse methods, random controls, and the free-block butterfly fit still execute on CPU.
The default is `--device cpu`; device selection does not turn synthetic/random probes into language-quality evidence.
Baseline errors currently describe FP32 masters, not serialized storage-dtype rounding; payload byte budgets exclude
checkpoint headers and manifests. Qualify serialization separately before claiming an equal-byte quality result.

### Output-collapse diagnosis (plan phases 0–2)

`scripts/diagnose_collapse.py` publishes each run through `ExperimentRun` and writes a separate progressive JSONL outside model directories. It uses the existing reference `RotorLinear` backend; device is selectable (`cpu` by default), but do not share an active GPU evaluation. No checkpoint or dataset downloads occur. Projection parity enumerates every checkpoint-manifest layer, streams bounded identity-row chunks, and independently applies each cascade stage equation; it does not allocate a full dense weight or identity matrix. The report includes deterministic random probes, actual dense-teacher prompt activations, finite/error/RMS distributions, weight error, FFN/block traces, first-pass logits, token agreement/KL, and visible-whitespace generations.

Run the tiny, known-correct CPU parity smoke first:

```bash
.venv/bin/python scripts/diagnose_collapse.py --smoke
```

Then compare every stored projection on CPU without generation or ablations:

```bash
.venv/bin/python scripts/diagnose_collapse.py \
  --ckpt <existing-rotor-checkpoint> --device cpu --dtype float32 \
  --prompt-limit 1 --probe-rows 16 --basis-chunk 64 --parity-only \
  --snapshot <trusted-snapshot-under-configured-NAS>
```

For bounded phase-2 screening, omit `--parity-only` and select only the plan's layers:

```bash
.venv/bin/python scripts/diagnose_collapse.py \
  --ckpt <existing-rotor-checkpoint> --device cpu --dtype float32 \
  --prompt-limit 1 --layers 0,12,23 --max-new-tokens 32
```

The runner does not use the curated diagnostic prompts as held-out NLL. NLL is omitted unless an explicitly supplied `--held-out-text` file is approved; the current evaluation corpus config is null, and no approved corpus is configured. FP32 master substitutions require `--snapshot`; only snapshot projections with matching base provenance, source tensor, CascadeSpec identity, and trusted weights-only payload are compared. Whole-model FP32 replacement is never inferred from missing masters. FP16/BF16 arms hold topology and master weights fixed; FP32 uses twice the parameter payload bytes.

FP64 smoke compares against the independent dense-action stream at `rtol=atol=1e-10`, matching `tests/test_math.py`. Full checkpoint FP32 numerical discrepancies are reported, not judged using an invented tolerance. All JSONL records are flushed and fsynced; choose a fresh output path when using `--output`.

### Joint FFN output fitting

Phase 4B uses a standalone runner rather than `scripts/convert_model.py`: `scripts/fit_joint_ffn.py` fits the gate,
up, and down projections together against the composed teacher FFN output. Its YAML has a distinct schema from the
standard per-projection conversion schema above; the joint objective is fixed to normalized squared composed FFN output error.
The schema and `fit_joint_ffn` API are used by `load_joint_config` / `run_joint_ffn` in
`rotor.converters.joint_ffn_runner`. The checked-in example is
[`qwen3.5-0.8b-joint-ffn.yaml`](qwen3.5-0.8b-joint-ffn.yaml). It deliberately leaves both data paths null because no
approved corpus is currently configured. Copy it to a run-specific YAML and set both paths to existing, user-approved
local text files before invoking the runner. Relative text paths resolve against the YAML file's directory; each file
must contain enough tokens for its requested windows. The runner has no random-token or synthetic-row fallback.

Each input file is tokenized with the model's local tokenizer and split into seeded, non-overlapping windows. The runner
requires matching calibration/held-out `seq_len` values and rejects duplicate token windows between these sets. It
captures dense teacher MLP input rows for each set and deterministically subsamples each capture to `max_rows`. The fit
API additionally rejects exact duplicate rows across the resulting training and held-out tensors. Calibration rows
alone drive optimization; held-out rows are evaluation-only. Curated evaluation prompts and random tokens are not
training data or language-quality evidence.

The runner requires a local base checkpoint already configured in `configs/models.yaml`; it never downloads model
weights. One target layer must have exactly one gate, up, and down projection, all with the same source weight dtype.
The joint budget is `floor(target_ratio * combined_dense_bytes)` over all three dense weights **and present biases**.
After reserving one cascade per projection and all serialized biases, remaining bytes add rotors to the projection with
the lowest current structured-weight-parameter / dense-weight-parameter ratio. The allocator stops
when no further rotor fits; it does not divide the remaining bytes into three independent per-projection budgets.
`budget.target` and `budget.byte_ratio` are alternatives.
`budget.storage_dtype` is the serialized parameter/bias dtype. The budget/fit payload counts include biases and exclude
safetensors headers and manifest metadata; final checkpoint metadata and file bytes are recorded separately.

Run from the repository root with:

```bash
.venv/bin/python scripts/fit_joint_ffn.py --config configs/experiments/qwen3.5-0.8b-joint-ffn.yaml
```

The CLI accepts `--device`, `--steps-scale`, `--resume`, `--snapshot-interval-seconds` (default 1800), and
`--snapshot-keep` (default 3). For example, use `--device cpu` for CPU execution or `--device cuda` for ROCm PyTorch.
Do not compete with an active GPU evaluation. The deterministic tiny-model CPU behavioral tests call the fit API
directly; they do not exercise a no-data runner mode or establish language behavior. Every runner launch publishes an
`ExperimentRun` report as required by the experiment contract.

`fit_joint_ffn(X_train, weights, specs, schedule, *, biases=None, heldout_X=None, seed=0, init_angle_std=0.01,
init=None, device="cpu", log=None, resume_state=None, progress_callback=None, storage_dtype="bfloat16",
combined_byte_budget=None)` and the composition helpers `joint_ffn_output(x, weights, biases=None)` and
`joint_ffn_student_output(x, modules)` are in `rotor.converters.fit`. Weight/spec mappings must have exactly the keys
`gate`, `up`, `down`; optional bias/init mappings may contain those keys only. The fit returns the three `RotorLinear`
modules and metrics. It minimizes normalized **squared** composed FFN output error, applying the original projection
biases with the dense teacher's source checkpoint dtypes and the exact `down(SiLU(gate(x)) * up(x))` convention;
student modules and trainable biases remain FP32 masters, with loss accumulation in FP32.

Metrics include `train_output_relative_error` and `heldout_output_relative_error` (normalized squared Frobenius error),
per-projection and combined dense/payload byte counts, `combined_byte_budget`, and `combined_byte_ratio`. Payload
counts are at `storage_dtype`, include present biases, and exclude checkpoint metadata.

The runner exports only the three fitted projections in the standard `rotor-scaler` format to
`storage.derived_path(name)` after `storage.assert_writable`, using `atomic_dir`; it refuses to overwrite an existing
destination. Every runner launch publishes an `ExperimentRun` report containing config, base provenance, environment,
input/window/row fingerprints, byte plan, fit metrics, resources, and serialized checkpoint accounting under
`reports/experiments/` (or `ROTOR_REPORTS_DIR`). This reports a fixed-row teacher FFN function fit, not end-to-end
language-model quality.

Resumable snapshots live under the configured storage root beside the checkpoint at
`<name>.snapshots/joint-snapshot-*`. Each step-boundary snapshot is atomically published in the
`rotor-joint-ffn-snapshot` format; retention removes only owned snapshots after publication succeeds. State contains all
three FP32 modules and their shared optimizer/schedule progress. Resume validates the resolved config, base provenance,
projection specs and source tensors, and the token-window and captured-row fingerprints for both calibration and
held-out data. Use the same config, base, and data split with `--resume <snapshot-directory>`. Snapshots are recovery
artifacts, not inference checkpoints, and use PyTorch restricted `weights_only=True` loading. The final checkpoint is
separate and stored in the configured storage dtype; assess its serialization rounding using established dtype
tolerances.
