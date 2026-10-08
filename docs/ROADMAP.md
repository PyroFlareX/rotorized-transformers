# Rotor Transformer Roadmap

Status legend: `[x]` done, `[ ]` open, `[~]` in progress.

Scope reminder: text-only FFN projections (gate/up/down) are the initial target. Attention, DeltaNet, norms, embeddings, RoPE, MTP, and vision stay untouched. See `docs/DECISIONS.md`.

---

## [x] M0 — Repo scaffold, storage config, rotor math, format v1, tiny-model tests

**Goal:** A working, testable Python core that can build rotor-parameterized matrices and round-trip them through the format v1 spec, without touching real models.

**Deliverables:**
- Repo layout under `src/rotor/` (math, formats, storage config loader).
- `configs/storage.yaml` centralizing NAS paths (D1).
- Deterministic index-free topologies: brick, butterfly, seeded SplitMix64 (D3).
- Rectangular pad/truncate handling (D4).
- Format v1 spec at `src/rotor/formats/spec.py` (D8).
- Tiny-model tests (synthetic matrices, no downloads).
- Qwen3.5 adapter: header-only tensor discovery and projection classification (gate/up/down vs. untouched families).
- Converter pipeline and CLIs under `scripts/*.py`.
- `runtime/interfaces` C++ reference headers, with a parity test against the Python reference.

**Exit criteria:**
- Rotor matrices reproduce from `(kind, seed, shape, params)` alone.
- Format v1 write/read round trip reproduces layer outputs exactly for float32 storage and within a documented tolerance for bfloat16.
- Storage paths resolve only through `configs/storage.yaml`.
- Tests pass on CPU-only PyTorch (D9).

**Dependencies:** none.

---

## [ ] M1 — Inspect Model A (`Qwen/Qwen3.5-0.8B-Base`)

**Goal:** Record Model A's FFN tensor inventory from remote headers, then load it text-only for reference inference.

**Deliverables:**
- (a) Remote header inspection, no download permission needed: `scripts/inspect_model.py --remote --model qwen3.5-0.8b` reads config and safetensors headers via HTTP range requests. Produces per-layer gate/up/down shapes, parameter counts, FFN share of total, and tensor inventory.
- (b) Local copy at `/mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-0.8B-Base`, text-only loader via `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` with tied embeddings preserved (D7), and reference forward pass on a fixed prompt.

**Exit criteria:**
- (a) Remote inventory matches the published tensor layout.
- (b) Reference forward pass on a fixed prompt produces stable logits across two loads.

**Dependencies:** M0.

**Gate (b only):** Downloading ~1.75 GB (published total `1746882752` bytes) requires explicit user permission before starting. Part (a) runs without permission and fetches only config and header byte ranges. Until permission is granted, (b) stays open.

---

## [ ] M2 — Weight-space equal-byte sweeps on Model A

**Goal:** Compare rotor FFN replacement against truncated SVD at equal stored bytes, in weight space only.

**Deliverables:**
- Sweep harness with arms: truncated SVD; SVD→Givens (correctness baseline: truncated SVD whose orthonormal factors are stored as Givens angles); rotor weight-fit (direct cascade D_0 R_1 D_1 … R_k D_k, no SVD; topologies brick, butterfly, seeded; varying rotors and stages; optional residual reported separately) (D4, D5).
- Per-layer, per-projection relative Frobenius error versus stored bytes curves.
- Controls: spectrum-matched random matrices (same singular values, Haar-random singular vectors) and iid Gaussian matrices of equal Frobenius norm. Controls separate structure exploitation from spectrum effects.

**Exit criteria:**
- Every rotor point has a truncated-SVD point at identical stored bytes.
- Results reported with compute cost separately from storage cost.
- Rotor results are reported against both control types; a rotor gain counts only where it exceeds what spectrum-matched controls achieve.

**Dependencies:** M1.

---

## [ ] M3 — Activation-aware fitting and validation loss on Model A

**Goal:** Fit rotor parameters against activation statistics and measure end-to-end validation loss.

**Deliverables:**
- Activation capture for gate/up/down inputs on a fixed calibration set.
- Activation-weighted fitting objective.
- Validation loss/perplexity for: base, rotor, truncated SVD, controls, at matched bytes.
- Optional residual term reported separately from the main rotor result (D4).

**Exit criteria:**
- Validation loss reported for all arms at identical stored-byte budgets.
- Calibration and validation sets are disjoint and recorded by hash.

**Dependencies:** M2.

---

## [ ] M4 — First CUDA operator microbenchmarks

**Goal:** Measure rotor apply cost against dense and low-rank matmuls on GPU.

**Deliverables:**
- CUDA kernels for butterfly and brick rotor application.
- Microbenchmarks versus dense GEMM and truncated-SVD matmul at FFN shapes.
- Benchmark harness recording device, driver, and clock state.

**Exit criteria:**
- Kernel outputs match the CPU reference within a documented tolerance.
- Throughput and memory-traffic numbers recorded for each FFN shape.

**Dependencies:** M0 (format and math), M1 (shapes). Blocked on a working NVIDIA driver; current dev host has `nvidia-smi` failing (D9).

---

## [ ] M5 — Model B (`Qwen/Qwen3.5-2B-Base`) scaling study

**Goal:** Repeat the compression-quality study on a larger model and add recovery fine-tuning, baselines, decoder accuracy/throughput, and longer context.

**Deliverables:**
- Model B local copy at `/mnt/NAS/MediaHome/Local-Models/base/Qwen/Qwen3.5-2B-Base` (published total `4548144832` bytes; requires explicit permission before download).
- Compression-quality curves for Model B matching the M2/M3 arms.
- Recovery fine-tuning of replaced projections only.
- Decoder accuracy and throughput on the same prompts used for Model A.
- Longer-context evaluation.

**Exit criteria:**
- Curves for all arms at matched stored bytes.
- Recovery fine-tune improves validation loss without modifying untouched modules.
- Derived checkpoint follows D2 (replaced projections plus manifest, no base duplication).

**Dependencies:** M3, M4 (for throughput numbers).

---

## [ ] M6 — Model C (`Qwen/Qwen3.5-4B-Base`) scaling

**Goal:** Confirm that the M5 trends hold at a third model size.

**Deliverables:**
- Model C local copy under `base/Qwen/Qwen3.5-4B-Base` (published total `9319737856` bytes; requires explicit permission before download).
- Same arm set and metrics as M5.
- Cross-model summary of quality versus stored bytes.

**Exit criteria:**
- Scaling summary covering A, B, C with identical protocol.
- Any deviation from M5 trends documented with evidence.

**Dependencies:** M5.

---

## [ ] M7 — Inference engine selection and fork under `runtime/fork`

**Goal:** Choose an inference engine for rotor-replaced models and fork it under `runtime/fork` implementing the format v1 contract.

**Deliverables:**
- Engine evaluation note (candidate engines, rotor-kernel integration cost, licensing).
- Fork at `runtime/fork` with C++ loader matching `runtime/interfaces` headers (D8).
- Parity test: C++ runtime output matches Python reference on fixed prompts.

**Exit criteria:**
- Runtime loads derived checkpoints without access to Python code.
- Logit parity with the Python reference within documented tolerance.

**Dependencies:** M0 (format v1), M4 (kernels), M5 (a model worth serving).

---

## [ ] Later — Attention and DeltaNet projections

**Goal:** Extend rotor replacement beyond FFN to attention and DeltaNet projections.

**Deliverables:**
- Per-projection-type analysis of error sensitivity.
- Format v1 extension or v2 spec, if required, recorded as a new ADR.
- Quality-versus-bytes curves for attention and DeltaNet projections.

**Exit criteria:**
- Each projection type has an explicit go/no-go decision with evidence.

**Dependencies:** M5, M7. Scope change from D6 requires a new ADR.
