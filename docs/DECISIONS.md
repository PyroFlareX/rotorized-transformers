# Decisions

ADR-style log. All entries dated 2026-10-08.

---

## D1 — NAS storage root and layout

**Date:** 2026-10-08

**Context:** Model weights and derived checkpoints are large and must not live in the repo. A NAS is available, but parts of it already hold unrelated content.

**Decision:** Use `/mnt/NAS/MediaHome/Local-Models/` as the root, with `base/` for downloaded upstream checkpoints (each model in its own containing directory) and `temp/<run-name>/` for rotor-derived artifacts, as requested by the user on 2026-10-08. All paths are centralized in `configs/storage.yaml`. Unrelated pre-existing NAS directories are read-only; new conversion directories under `temp/` are authorized, and existing destinations are never overwritten.

**Consequences:**
- No code hardcodes NAS paths; everything resolves through `configs/storage.yaml`.
- Moving the root requires one config change.
- Accidental writes outside the project subtrees are a policy violation, not just a bug.

**Status:** Accepted.

---

## D2 — Derived checkpoints store only replaced projections

**Date:** 2026-10-08

**Context:** Base checkpoints are multi-gigabyte. Copying untouched tensors into every derived artifact wastes NAS space and makes provenance unclear.

**Decision:** A derived checkpoint stores only the replaced projections plus a manifest. The manifest references the base model by path, revision, and sha256. Untouched tensors are never duplicated.

**Consequences:**
- Derived checkpoints are small and cheap to produce.
- Loading requires both the base and the derived artifact. The manifest should be verified before use; see the drift note below.
- Base revision drift can be detected by re-hashing: the manifest records per-shard sha256, and comparing against the current base files reveals drift. Verification on load is not yet automatic; it must be run explicitly until implemented.

**Status:** Accepted.

---

## D3 — Index-free deterministic topologies

**Date:** 2026-10-08

**Context:** Storing explicit index lists for rotor pairings costs bytes that count against the compression budget and complicates format versioning.

**Decision:** Rotor topologies are deterministic and index-free: brick, butterfly, and seeded SplitMix64 pairings. Only `(kind, seed)` is stored; indices are regenerated at load time.

**Consequences:**
- Stored overhead per projection is a few bytes.
- Any change to the topology generator is a format break and requires a version bump.
- Reproducibility depends on the SplitMix64 implementation being identical in Python and C++ (D8).

**Status:** Accepted.

---

## D4 — Rectangular matrices via pad-to-max and truncate

**Date:** 2026-10-08

**Context:** FFN projections are rectangular (gate/up map hidden→intermediate, down maps back). Rotor transforms act on square orthogonal blocks.

**Decision:** Rectangular matrices are handled by padding the smaller dimension to `max(in, out)`, applying the square rotor, then truncating to the target shape. The main experiment uses no low-rank bottleneck. An optional residual is fitted and reported separately from the main result.

**Consequences:**
- Padding adds compute cost that must be reported in M4 benchmarks.
- Main results are not confounded by a hidden low-rank term.
- Residual gains, if any, are visible as a separate line item and never folded into the headline number.

**Status:** Accepted.

---

## D5 — Equal stored-byte comparisons with mandatory baselines

**Date:** 2026-10-08

**Context:** Comparing methods at different byte budgets, or without a truncated-SVD reference, can make any new parameterization look better than it is.

**Decision:** All quality comparisons use equal stored bytes. Truncated SVD is a mandatory baseline in every sweep. Controls are exactly two kinds: spectrum-matched random matrices (same singular values, Haar-random singular vectors) and iid Gaussian matrices of equal Frobenius norm. Controls separate structure exploitation from spectrum effects. Compute cost is measured and reported separately from storage cost.

**Consequences:**
- Every rotor result has a matched SVD point; unmatched results are not reported as wins.
- Sweeps cost more to run.
- Compute and storage trade-offs are presented as two axes, not one blended score.

**Status:** Accepted.

---

## D6 — Initial scope is text-only FFN (gate/up/down)

**Date:** 2026-10-08

**Context:** Attention, DeltaNet, norms, embeddings, RoPE, MTP heads, and vision encoders each carry distinct sensitivity profiles. Touching them at once would confound results.

**Decision:** The initial scope is text-only FFN projections (gate, up, down). Attention, DeltaNet, norms, embeddings, RoPE, MTP, and vision are left untouched. Extending scope requires a new ADR.

**Consequences:**
- Results isolate one projection family.
- Quality numbers do not represent whole-model compression until later milestones.
- Roadmap item "Attention and DeltaNet projections" is gated on this ADR being superseded.

**Status:** Accepted.

---

## D7 — Text-only loading through transformers

**Date:** 2026-10-08

**Context:** Qwen3.5 checkpoints include vision and MTP components. Loading the full multimodal stack adds memory and code paths that the experiments do not use. Transformers 5.19 is verified to expose the text model.

**Decision:** Load models with `AutoModelForCausalLM`, which resolves to `Qwen3_5ForCausalLM`. Tied input and output embeddings are preserved as the upstream config defines them.

**Consequences:**
- Vision and MTP weights are not materialized in memory during experiments.
- Loader behavior depends on transformers 5.19; upgrades require re-verification of the class resolution and tie handling.
- Tied embeddings must be checked in M1 inventory.

**Status:** Accepted.

---

## D8 — Python converter independent of C++ runtime; shared format spec

**Date:** 2026-10-08

**Context:** The research pipeline is Python. The eventual inference runtime is C++ (M5). Coupling the converter to runtime internals would block either side from changing independently.

**Decision:** The Python converter is independent of the C++ runtime. The only shared contract is the format spec at `src/rotor/formats/spec.py`, mirrored in `runtime/interfaces` headers. Any spec change updates both mirrors in the same change.

**Consequences:**
- The runtime can be developed and tested without Python present.
- Spec drift is possible; `tests/test_cxx_parity.py` checks parity now, with engine conformance validation in M5.
- Format versioning is explicit in the spec file.

**Status:** Accepted.

---

## D9 — CPU-only PyTorch on the current host

**Date:** 2026-10-08

**Context:** The current host has no NVIDIA GPU on the PCI bus. It has an AMD Radeon RX 7900 XT/XTX (Navi 31, 24 GiB VRAM) and AMD Raphael iGPU, both through `amdgpu`. There is no ROCm or `hipcc`; the Python environment uses torch 2.14.1+cpu. `nvcc` 12.0 is present for compile-only checks. Vulkan loader and Mesa radeon ICD are installed; `vulkaninfo` is absent.

**Decision:** Treat this host as CPU-only for torch execution and GPU benchmarking. CUDA kernels may be compiled here with `nvcc`, but CUDA runtime correctness and performance benchmarking require another host with an NVIDIA GPU. AMD GPU runtime work is unavailable until a ROCm-capable environment is provisioned.

**Consequences:**
- Local tests and inventory work run on CPU; measurements here are not GPU performance claims.
- `nvcc` compilation alone does not satisfy the M4 runtime correctness/performance gate.
- Vulkan loader/ICD presence does not establish application-level Vulkan availability; `vulkaninfo` is not installed.

**Status:** Accepted for the current host; revisit when GPU runtime capability changes.

---

## D10 — Git excludes weights and large binaries

**Date:** 2026-10-08

**Context:** Model weights and derived checkpoints are gigabytes each. Committing them would make the repo unusable and leak large artifacts into history.

**Decision:** `.gitignore` excludes weights and binary model files, including `*.safetensors`, `*.bin`, `*.gguf`, and similar. Weights live only on the NAS under D1. The repo holds code, configs, docs, and small fixtures.

**Consequences:**
- Clones do not contain weights; users must restore from NAS or re-download with permission.
- Accidentally added weight files are caught by ignore rules, but ignore rules alone do not stop force-adds; reviewers must check.
- Small test fixtures must stay well under the size that would make them weights-like.

**Status:** Accepted.
---

## D11 — Run-directory experiment reporting

**Date:** 2026-10-08

**Context:** Flat JSON run records do not provide a durable, self-contained home for configuration, environment, figures, logs, interpretation, or failure evidence.

**Decision:** Every experiment is published under `reports/experiments/<model>__<method>__<target>__<UTC>/` with `report.md`, `metrics.json`, `config.yaml`, `environment.json`, `figures/`, and `logs/`. Failed runs are published with status `failed`; this replaces flat JSON run records.

**Consequences:**
- Each run is auditable and independently tied to its configuration and environment.
- Failure information remains reportable rather than disappearing with the failed process.
- Consumers must use run directories rather than the retired flat-record convention.

**Status:** Accepted.

---

## D12 — Record environment in manifests and runs

**Date:** 2026-10-08

**Context:** Hardware, software, storage, and compiler capabilities vary by host and affect reproducibility and benchmark interpretation.

**Decision:** Detect and record the actual environment in every model manifest and experiment run via `rotor.environment`.

**Consequences:**
- Benchmark records can distinguish CPU, GPU, CUDA, Vulkan, storage, and compiler capabilities instead of assuming them.
- GPU claims must remain separate from CPU fallback or compile-only observations.
- Environment capture becomes part of routine manifest and run production.

**Status:** Accepted.

---

## D13 — Named compression targets and serialized-byte accounting

**Date:** 2026-10-08

**Context:** Angle/parameter counts are not equivalent to actual checkpoint size, and inconsistent budget labels make experiment comparisons difficult.

**Decision:** Use named compression-target presets `b075`, `b050`, `b035`, `b025`, and `b015` for fractions of original stored bytes. Report checkpoint file/header/padding/index/quantization-metadata bytes separately from per-layer tensor bytes.

**Consequences:**
- Budget comparisons use serialized layer tensor bytes, while checkpoint-level overhead remains visible separately.
- Reports can distinguish deterministic-topology/index and float-storage/quantization-metadata overhead rather than hiding zero-valued categories.
- Preset names are stable, human-readable identifiers for experiment configs and reports.

**Status:** Accepted.

---

## D14 — Equal-byte baseline set and controls

**Date:** 2026-10-08

**Context:** A rotor result is not meaningful without relevant alternatives at equal storage and controls that separate structural gains from spectrum effects.

**Decision:** Compare dense, truncated SVD, SVD→Givens, magnitude sparse, butterfly, rotor/scaler, and rotor/scaler+residual at equal stored-byte budgets; include spectrum-matched random and Gaussian controls. Report compute separately from storage and do not attribute SVD-truncation gains to rotors.

**Consequences:**
- Baseline and control rows are part of the experiment contract, not optional context for headline results.
- Dense is the reference; residual and SVD contributions remain identifiable.
- More expensive sweeps are required, but prevent unsupported structural-compression claims.

**Status:** Accepted.

---

## D15 — Paper toolchain uses pdflatex, bibtex, and natbib

**Date:** 2026-10-08

**Context:** The current host provides `pdflatex` and `bibtex`; `biblatex`, `biber`, and `latexmk` are not installed.

**Decision:** Author papers for the available `pdflatex` + `bibtex` + `natbib` toolchain; do not require `biblatex`/`biber`.

**Consequences:**
- Paper builds use explicit pdflatex/bibtex passes rather than latexmk or biber.
- A future host with a different toolchain does not silently change project dependencies.

**Status:** Accepted.

---

## D16 — Fixed prompt outputs and sampled runtime telemetry

**Date:** 2026-10-08

**Context:** The user requested a 256-entry list of input/output results and a consistent runtime measurement standard before comparing the 0.8B conversion.

**Decision:** Keep 256 curated Base-model continuation prompts in the repository, and record deterministic greedy output text, input/output token IDs and counts, settings, and fingerprints in per-run JSONL listings. Reuse the base tokenizer for converted runs; reject comparisons with mismatched inputs or settings. Record TTFT, prefill and decode throughput with their existing definitions separately from low-rate (default 1 Hz) process, host, and device telemetry. Attribute telemetry to loading/inference/conversion phases; unavailable sensors are null rather than zero.

**Consequences:**
- Output agreement is observable but is not a substitute for language-model quality metrics or equal-byte baselines.
- Measurements retain actual hardware/software and timing provenance; CPU execution is never described as GPU performance.
- Progressive listings remain inspectable during long runs, and failures retain their evidence.
- The full configured conversion is launched without reducing projection scope, target budget, or fit steps. A process memory cap and no additional swap allocation protect the interactive host.

**Status:** Accepted.

---

## D17 — Resumable conversion snapshots

**Date:** 2026-10-08

**Context:** Full-model fitting takes hours; interruption must not discard completed projections or restart Adam.

**Decision:** Publish recovery snapshots atomically under the configured NAS output root every 30 minutes,
retaining the newest three per run. Store completed projections and the active FP32 fit, Adam state, phase,
next optimizer step, and loss history. Resume validates conversion and base identity before continuing.
Recovery snapshots are separate from the inference checkpoint format; no format-version change is required.

**Consequences:**
- Publication occurs at consistent optimizer-step boundaries, so intervals may be delayed by a long step.
- Retention removes only owned recovery snapshots after successful publication; experiment reports remain intact.
- The final checkpoint remains exclusive and atomically published; base checkpoints remain read-only.

**Status:** Accepted.
