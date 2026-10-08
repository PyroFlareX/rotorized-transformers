# Rotor Transformer

Research code testing whether the linear weights of pretrained Transformers can be stored in a smaller *compositional* form (products of Givens rotations and diagonal scalings on a deterministic topology) than as dense coefficients, without substantial loss in language-model quality. The hypothesis is that pretrained linear maps `W` admit such representations at fewer stored bytes than dense `W`; this repository exists to test that, not to assume it. Every comparison is made at equal stored bytes against ordinary truncated-SVD low-rank baselines, so that gains from plain rank truncation are never credited to rotors.

## Representation

For a dense weight `W` of shape `out x in`:

```
W ≈ D_0 R_1 D_1 R_2 D_2 … R_k D_k
```

- `R_i`: product of disjoint 2D Givens rotation stages whose index pairs come from a deterministic topology (`brick`, `butterfly`, `seeded`); only angles are stored, never indices.
- `D_i`: learned diagonal scalings.
- Rectangular layers: pad to `max(in, out)` and truncate (`pad_truncate`).
- Optional low-rank residual `U V`, always reported separately.

Normative forward semantics and tensor layout: `src/rotor/formats/spec.py`. Derivation and parameter counts: `docs/RESEARCH_PLAN.md`.

## What is and is not claimed

- **No results yet.** No benchmark, compression ratio, or quality number is claimed anywhere in this repository. Run records (JSON under `reports/experiments/`) are the only source of measured numbers; formula-derived values are labeled `estimate`.
- Initial scope: text-only, FFN projections only (`ffn.gate`, `ffn.up`, `ffn.down`). Attention, DeltaNet, norms, embeddings, RoPE, MTP, and vision tensors are untouched.
- Whether rotor application is ever faster than a dense GEMM is an open, unmeasured question (`docs/INFERENCE_ENGINE_PLAN.md`).
- Status: milestone M0 (scaffold, storage, math, format v1, converters, evaluation, tests) is complete; M1 onward is open. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Target models

From `configs/models.yaml` (revisions pinned from the Hugging Face Hub API; sizes are published safetensors totals):

| Role | Key | Hugging Face id | Pinned revision | Published size (bytes) | Layers / hidden / intermediate |
|---|---|---|---|---|---|
| A: development | `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 1,746,882,752 | 24 / 1024 / 3584 |
| B: main | `qwen3.5-2b` | `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | 4,548,144,832 | 24 / 2048 / 6144 |
| C: scaling | `qwen3.5-4b` | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | 9,319,737,856 | 32 / 2560 / 9216 |

No weights are downloaded by this repository's code, and none are present on this machine unless a user placed them.

## Model storage

All base and derived models live under `/mnt/NAS/MediaHome/Local-Models/` (base: `base/`, derived: `derived/rotor-scaler/`), never in this repository. Paths are resolved only through `configs/storage.yaml` via `rotor.storage.load_storage()`. The full rules (read-only NAS directories, atomic publishing, download policy) are in [`AGENTS.md`](AGENTS.md).

## Installation

Python ≥ 3.11. CPU-only PyTorch is sufficient (the development host has no NVIDIA driver):

```bash
python3 -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch && .venv/bin/pip install -e '.[dev]'
```

Verified versions for the Qwen3.5 model classes: `transformers>=5.19` (see `pyproject.toml`).

## Quickstart

1. **Inspect a model remotely (header-only, nothing downloaded).** Reads `config.json` and safetensors headers over HTTP range requests; reports tensor groups, layer types, and FFN projections:

   ```bash
   .venv/bin/python scripts/inspect_model.py --remote --model qwen3.5-0.8b
   ```

2. **Equal-byte baseline comparison on a synthetic matrix (no model needed).** Truncated SVD vs SVD→Givens vs rotor fits, plus random controls, at the same stored-byte budget:

   ```bash
   .venv/bin/python scripts/compare_baselines.py --synthetic 48,32 --steps 50
   ```

3. **End-to-end on a tiny random checkpoint.** `rotor.models.tiny.make_tiny_checkpoint` writes a random Qwen3.5 checkpoint (hidden 64, 4 layers, < 1 MB) in the real on-disk layout. `tests/test_conversion_integration.py` converts one, publishes it atomically into a temporary storage root, reloads it, and checks the safety refusals:

   ```bash
   .venv/bin/python -m pytest tests/test_conversion_integration.py
   ```

   The full suite (CPU-only; excludes `network` and `nas` markers by default): `.venv/bin/python -m pytest`.

4. **Real conversion (only after the user approves the download).** Downloading is never automatic. Once approved, place the pinned revision of `Qwen/Qwen3.5-0.8B-Base` at `/mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base` (the `local_path` in `configs/models.yaml`; or point `local_path` at an existing copy elsewhere under the NAS root via `configs/models.local.yaml`). Then print the allocation and byte plan without fitting or writing anything:

   ```bash
   .venv/bin/python scripts/convert_model.py --config configs/experiments/qwen3.5-0.8b-ffn-weight.yaml --dry-run
   ```

   Dropping `--dry-run` fits every selected projection and publishes the result atomically under `/mnt/NAS/MediaHome/Local-Models/derived/rotor-scaler/<name>/` (refuses to overwrite). Evaluation and recovery then use `scripts/evaluate_model.py` and `scripts/train_recovery.py`; experiment YAML fields are documented in [`configs/experiments/README.md`](configs/experiments/README.md).

### Scripts

| Script | Purpose |
|---|---|
| `scripts/inspect_model.py` | Tensor groups, layer types, FFN projections (local or `--remote` headers) |
| `scripts/compare_baselines.py` | Per-matrix equal-byte comparison: SVD, SVD→Givens, rotor, random controls |
| `scripts/sweep_compression.py` | Weight-space sweep over byte ratios, topologies, residual ranks (`--dry-run` lists the plan) |
| `scripts/convert_model.py` | Convert FFN projections from an experiment YAML |
| `scripts/evaluate_model.py` | Next-token NLL of base/converted model on identical token windows |
| `scripts/train_recovery.py` | KL-distillation recovery of the replaced layers |
| `scripts/benchmark.py` | Microbenchmark dense vs rotor vs low-rank linear at equal stored bytes |

Every script supports `--help`.

## Repository layout

```
src/rotor/        math, formats, models, converters, evaluation, training, storage, data
scripts/          CLI entry points
configs/          storage.yaml, models.yaml, experiments/*.yaml
runtime/          C++ interface headers (kernels/ and fork/ are empty)
tests/            pytest suite (CPU-only)
docs/             architecture, research plan, decisions, roadmap, inference engine plan
reports/          run records, figures, papers, literature notes
```

## Documentation

- [`AGENTS.md`](AGENTS.md): repository guidelines, mandatory storage rules, research-integrity rules, commands, conventions
- [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md): hypothesis, representation, experiment matrix, falsification criteria
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): package layout, dependency rules, data flow, format, safety mechanisms
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decisions D1–D10
- [`docs/ROADMAP.md`](docs/ROADMAP.md): milestones M0–M7
- [`docs/INFERENCE_ENGINE_PLAN.md`](docs/INFERENCE_ENGINE_PLAN.md): engine selection and kernel plan (none measured)
- [`runtime/README.md`](runtime/README.md): C++ runtime status
