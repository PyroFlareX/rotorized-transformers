# Inference engine plan

Status: **plan only**. No engine is integrated, no kernel exists, and **every speed or memory benefit mentioned
here is unmeasured**. Statements about third-party engines describe what must be *checked*, not what is known.

## 1. Goal

Run `rotor-scaler` v1 checkpoints (`src/rotor/formats/spec.py`) in a native engine for the hybrid Qwen3.5 family
(Gated DeltaNet linear-attention layers interleaved with gated full-attention layers; initial scope replaces only
FFN `gate/up/down` projections). The question to answer is whether a structured layer with the same stored bytes as
a dense or truncated-SVD layer is competitive in latency and throughput, and at what quality. Quality comparisons
follow the equal-stored-bytes rule from the project contract; this document concerns only the runtime.

## 2. Candidate engines

Support for the hybrid Qwen3.5 architecture (Gated DeltaNet recurrent state + gated attention, possibly multimodal
and MTP tensors in the checkpoint) **must be verified against the exact upstream commit and model revision before
selection**; it is not assumed for any row below.

| Engine | Potential advantages | Potential drawbacks / risks | To verify |
|---|---|---|---|
| llama.cpp / ggml | Small C/C++ code base; CPU and many GPU backends; simple graph-building code makes adding a custom op in one FFN site tractable; easy to run on modest hardware | GGUF is its native weight container, so a loader for `rotor_weights.safetensors` or a GGUF conversion step is needed; the recurrent-state (Gated DeltaNet) architecture needs a model implementation and kernels per backend; custom ops must be written for each backend separately | Does the target commit load Qwen3.5 text weights and match reference logits? Which backends implement the DeltaNet op? |
| vLLM | Mature CUDA serving stack, paged KV, batching, Python-side model registry; custom CUDA ops are straightforward to register; torch interop eases golden comparison | Python/torch dependency conflicts with the "independent of Python" goal for the runtime (acceptable only for a benchmarking harness); heavy build; hybrid linear-attention support and state management depend on version | Is Qwen3.5 hybrid supported in the pinned release, including cache layout for DeltaNet state? |
| SGLang | Fast serving runtime, flexible scheduling, extensible model code | Same Python/torch coupling as vLLM; hybrid-model support and kernel provenance depend on version; smaller maintenance surface | Same as vLLM; also check whether its DeltaNet kernels come from an external library that must be pinned |
| Custom minimal engine | Full control of format, layout and fusion; no upstream churn; smallest harness for isolating rotor-vs-dense layer cost | Must implement Qwen3.5 end to end (DeltaNet, gated attention, tokenizer, sampling) correctly, a large correctness burden; no ecosystem | Cost of reproducing reference logits for the full model vs. benchmarking FFN layers in isolation |

Recommended order (a proposal, not a finding): (1) microbenchmark the FFN layer alone with a standalone kernel
harness under `runtime/kernels/` against dense and truncated-SVD baselines; (2) only if that shows a plausible
win, pick the engine that is verified to run Qwen3.5 correctly and cheapest to extend; (3) integrate through the
format, never through ad hoc exports.

## 3. Integration via the versioned format

- The engine reads `rotor_manifest.json` (format name/version checked against `ROTOR_FORMAT_NAME` /
  `ROTOR_FORMAT_VERSION`) and `rotor_weights.safetensors`, plus the untouched dense tensors from the base
  checkpoint recorded in the manifest (path, revision, sha256).
- Per layer it builds a `rotor_cascade_spec` and regenerates stage pairs from `(kind, width, stage, seed)` via
  `interfaces/rotor_topology.hpp`; no coordinate indices are stored. Pair tables may be cached per
  `(kind, width, stage, seed)` since they are layer-independent.
- An unknown `FORMAT_VERSION`, topology kind, adapter kind or storage dtype is a hard load error.
- Any format change bumps the version in Python and C simultaneously and updates `tests/test_cxx_parity.py`.
- Engine-specific repackaging (e.g. GGUF) is a derived artifact of the versioned format, produced by a converter that
  records its source manifest hash; it is never the source of truth.

## 4. Kernel design notes (hypotheses, none measured)

A cascade of `k` rotors with `m` stages each applies `k*m` passes over an `n`-vector, each pass touching `n` values
and `n/2` angles with about 6 flops per pair (4 mul + 2 add) plus one multiply per diagonal. Arithmetic intensity per
pass is low, so the layer is expected to be **memory-/latency-bound**, not compute-bound; the dense GEMM it competes
with is compute-bound at large token batches and weight-bandwidth-bound at batch 1. Which regime wins is an empirical
question.

- **Stage fusion.** Executing stages one kernel launch at a time forces a full round trip of the activation
  through global memory per stage. Fuse all stages of a rotor (and the following diagonal) in one kernel that keeps
  the activation tile in registers/shared memory. Butterfly strides above the warp size need shared-memory exchange;
  strides below can use warp shuffles. Brick pairs are local (neighbouring lanes). Seeded pairs are arbitrary
  permutations and are the hardest to fuse (bank conflicts, gathers); expect them to be slowest.
- **Angles in shared memory.** One rotor row is `m*~n/2` angles; for hidden sizes in the thousands this fits in
  shared memory for a tile of tokens, amortizing the global read across the token batch. Verify capacity per target
  GPU; if a rotor row does not fit, tile over stages.
- **sincos: precompute vs on-the-fly.** Options: (a) store angles and compute `sincos` on the fly (stored bytes
  minimal, extra SFU/ALU work, precision of fast intrinsics must be gated); (b) precompute `cos/sin` at load time
  (2x working memory, no stored-byte change since the checkpoint still holds angles; extra bandwidth at run time).
  Whether (a) or (b) is better depends on the measured balance of SFU throughput vs. memory bandwidth and must be
  decided per GPU.
- **Batching across tokens.** Amortize angle/pair loads over many tokens by processing a block of tokens per CTA with
  token as the contiguous or outer dimension; compare layouts. At batch 1 (decode) the layer is latency-bound by
  the serial dependency of stages, so launch overhead and fusion across rotors, or across gate/up projections that
  share an input, matter most. Consider CUDA graphs to remove launch cost.
- **Residual.** `U(Vx)` is two small GEMMs; fuse into the epilogue or run concurrently.
- **Precision.** Master angles are float32; the stored dtype may be bf16/fp16. Accumulate in fp32; decide through
  correctness gates (section 5) whether bf16 angle storage is acceptable, since angle rounding error compounds
  through `k*m` stages.
- **Pad/truncate.** The adapter is index-free; avoid materializing padding: a pair touching only padded zeros can be
  skipped for stages where both coordinates are provably zero (optimization, verify bit-equivalence first).
- **Backward/training kernels** are out of scope; Python autograd handles fitting.

## 5. Correctness gates

Each gate must pass before the next stage proceeds and results are recorded with the commit hashes of both repos.

1. **Topology parity**: C++ `stage_pairs` equals Python exactly for all kinds, including odd and
   non-power-of-two widths (`tests/test_cxx_parity.py`).
2. **Reference-cascade parity**: C++ double reference matches `RotorCascade` float64 within 1e-10.
3. **Golden outputs**: Python generates, from a fixed seed, a golden set per layer shape of interest
   (inputs, parameters, expected outputs in float64 and the engine's dtype). Each kernel is compared with an
   explicit tolerance derived from dtype (record the observed max abs / relative error, do not hand-wave).
4. **Layer-level end-to-end**: converted tiny checkpoint (`rotor.models.tiny`) executed in the engine reproduces
   Python logits within a stated tolerance, and argmax tokens agree on a fixed prompt set.
5. **Model-level**: for the real model, engine perplexity on a fixed token set matches the Python evaluation of the
   same checkpoint within a stated tolerance, and the dense baseline in the engine matches the Python dense baseline,
   which proves the engine itself is correct before any rotor claim is made.

## 6. Performance methodology

- **Measured vs. estimated** are always labeled separately. FLOP counts and byte counts from formulas
  (`CascadeSpec.flops_per_token`, `param_breakdown`) are `estimate`; wall-clock timings are `measured` and carry
  hardware, driver, engine commit, dtype, batch size, sequence length, warmup and iteration counts, and
  median/p10/p90.
- **Equal stored bytes.** Compare a rotor layer only with a dense layer, and with a truncated-SVD layer, whose stored
  bytes are the same (same storage dtype). A rotor layer that wins only against an *uncompressed* dense layer says
  nothing about structure. Report the SVD rank used and the byte budget for each arm.
- **Report compute separately** from storage: FLOPs (estimate), achieved bandwidth and FLOP/s (measured where counters
  are available), and latency. A byte-matched layer that uses more compute than the baseline is reported as such.
- **Regimes.** Decode (batch 1..few), prefill (many tokens), and several FFN shapes of the target model. Include
  layer-only microbenchmarks and end-to-end tokens/s, but do not infer one from the other.
- **Quality alongside speed.** Every performance row names its quality row (perplexity / NLL delta and the baseline
  it is compared with); a fast layer with unacceptable quality is not a result.
- **Controls.** Include a dense baseline in the same engine and a truncated-SVD baseline at equal bytes. Pin clocks or
  report variance; run on an otherwise idle device; repeat across process restarts.
- **Negative results are reported.** If the structured layer is not faster at equal bytes, that is the finding.

## 7. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Interfaces: `rotor_format.h`, `rotor_topology.hpp`, parity test | Gates 1-2 pass (this repo, present) |
| M1 | Standalone CPU kernel in `runtime/kernels/` (vectorized, multi-token) | Gate 3 on CPU; first *measured* CPU layer microbench vs dense and equal-bytes SVD |
| M2 | Standalone CUDA fused-stage kernel (butterfly/brick first, seeded after) | Gate 3 on GPU; measured layer microbench across decode and prefill batches, sincos variants compared (needs a GPU host; none is available currently) |
| M3 | Engine selection record: verify hybrid Qwen3.5 support at a pinned commit for each candidate; write the choice into `docs/DECISIONS.md` | Dense Qwen3.5 text logits in the candidate match the Python reference (gate 5 dense arm) |
| M4 | Fork/submodule under `runtime/fork/` with a loader for the versioned format and a rotor FFN op | Gate 4 on the tiny checkpoint |
| M5 | Real-model run: converted FFN layers, quality and speed at equal stored bytes | Gate 5 plus a measured performance table with every row labeled; negative outcome acceptable |

M2 onward requires hardware that is not available on the current host (no GPU driver); schedule it only when such
hardware exists, and until then claim nothing about GPU performance.
