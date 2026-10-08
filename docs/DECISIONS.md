# Decisions

ADR-style log. All entries dated 2026-10-08.

---

## D1 — NAS storage root and layout

**Date:** 2026-10-08

**Context:** Model weights and derived checkpoints are large and must not live in the repo. A NAS is available, but parts of it already hold unrelated content.

**Decision:** Use `/mnt/NAS/MediaHome/Local-Models/` as the root, with `base/` for downloaded upstream checkpoints and `derived/rotor-scaler/` for rotor-derived artifacts. All paths are centralized in `configs/storage.yaml`. Pre-existing NAS directories are read-only to this project; only the `base/` and `derived/rotor-scaler/` subtrees are written.

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

**Context:** The research pipeline is Python. The eventual inference runtime is C++ (M7). Coupling the converter to runtime internals would block either side from changing independently.

**Decision:** The Python converter is independent of the C++ runtime. The only shared contract is the format spec at `src/rotor/formats/spec.py`, mirrored in `runtime/interfaces` headers. Any spec change updates both mirrors in the same change.

**Consequences:**
- The runtime can be developed and tested without Python present.
- Spec drift is possible; parity tests in M7 are the guard.
- Format versioning is explicit in the spec file.

**Status:** Accepted.

---

## D9 — CPU-only PyTorch on the current host

**Date:** 2026-10-08

**Context:** The development host has no working NVIDIA driver: `nvidia-smi` fails, so CUDA is unavailable. M0 and M1 do not need a GPU.

**Decision:** The `.venv` on this host uses a CPU-only PyTorch wheel. GPU work (M4) is deferred until a host with a working driver is available.

**Consequences:**
- Local tests and inventory work run on CPU.
- Throughput numbers from this host are not valid for GPU claims.
- Switching to a CUDA wheel is a host change, recorded here when it happens.

**Status:** Accepted for M0–M3. Revisit when a GPU host is available for M4.

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
