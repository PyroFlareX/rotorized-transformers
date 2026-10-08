# Research plan

## Hypothesis

> Pretrained Transformer linear transformations may admit significantly smaller compositional representations using rotors and scalings than their ordinary dense coefficient representations, without substantial degradation in language-model quality.

This is a hypothesis to be tested, not a result. Nothing in this repository claims it holds. Every number in this
document that comes from a formula is an **estimate**; measured numbers appear in run directories under
`reports/experiments/<model>__<method>__<target>__<UTCstamp>/`.

## Representation

A dense linear weight `W` (shape `out x in`, `y = x W^T`) is approximated by

```
W ≈ D_0 R_1 D_1 R_2 D_2 … R_k D_k
```

- `R_i`: a product of `stages_per_rotor` stages, each stage a set of disjoint 2D Givens rotations whose index pairs come
  from a deterministic `Topology` (`brick`, `butterfly`, `seeded`). Pair indices are never stored; only angles.
- `D_i`: learned diagonal scalings.
- Rectangular layers use the `pad_truncate` adapter: the working width is `n = max(in, out)`; the input is zero-padded
  to `n`, and the first `out` coordinates are kept at the end.
- Optional low-rank residual `U V` (rank `r`) is added in parallel and is always reported separately.

The cascade imposes no rank bottleneck: interior width is `n`, so the operator rank is limited only by `min(in, out)`.

### Exact parameter counts (from the `CascadeSpec` docstring, `src/rotor/math/cascade.py`)

With `k = num_rotors`, `n = max(in, out)`, `pairs(n, s)` the pair set of stage `s`:

| Term      | Count                                  |
|-----------|----------------------------------------|
| angles    | `k * sum_s |pairs(n, s)|`              |
| diagonals | `in + out + (k - 1) * n`               |
| residual  | `r * (in + out)`                       |

Total is `CascadeSpec.param_count()`; the dense reference is `CascadeSpec.dense_param_count()` (`in * out`). Stored bytes
are parameters times `bytes_per_param(storage_dtype)` (float32 4 B, bfloat16/float16 2 B). Compute
(`CascadeSpec.flops_per_token()`) is a separate axis and is never folded into the byte budget.

## Why SVD→Givens is only a correctness baseline

`src/rotor/math/decompose.py` reproduces the earlier experiment: truncate the SVD to rank `r`, then re-express the
orthonormal factors `U_r`, `V_r` as Givens sequences. Per its docstring, an `m x r` Stiefel factor costs
`r*m - r*(r+1)/2` parameters (angles; plus `r` signs) versus `m*r` dense coefficients. The saving over dense factors is
only the Stiefel-manifold dimension; the dominant saving comes from **rank truncation itself**, which truncated SVD gets
without any rotors. Therefore SVD→Givens is used (a) to validate that the Givens machinery is exact and (b) to measure
how much of any gain is just truncation. Any gain explained by rank truncation MUST NOT be reported as a rotor gain.
The rotor cascade is a different proposal: full-rank, learned by gradient descent, with a fixed sparse topology.

## Models

Shapes read from published safetensors headers on 2026-10-08 (BF16; no weights downloaded). Revisions and
per-family text parameters are pinned in `configs/models.yaml`.

| Model | Key (`configs/models.yaml`) | Role | hidden | ffn | gate/up | down | layers |
|---|---|---|---|---|---|---|---|
| A: Qwen3.5-0.8B-Base | `qwen3.5-0.8b` | development | 1024 | 3584 | 3584x1024 | 1024x3584 | 24 (18 linear_attention, 6 full_attention) |
| B: Qwen3.5-2B-Base | `qwen3.5-2b` | main | 2048 | 6144 | 6144x2048 | 2048x6144 | 24 |
| C: Qwen3.5-4B-Base | `qwen3.5-4b` | scaling | 2560 | 9216 | 9216x2560 | 2560x9216 | 32 |

Other shapes (not in initial scope, listed for later phases):

| Model | linear_attn in_proj_qkv | in_proj_z | out_proj | attn q (query+gate) | attn k / v | attn o |
|---|---|---|---|---|---|---|
| 0.8B | 6144x1024 | 2048x1024 | 1024x2048 | 4096x1024 | 512x1024 | 1024x2048 |
| 2B | 6144x2048 | 2048x2048 | 2048x2048 | 4096x2048 | 512x2048 | 2048x2048 |
| 4B | 8192x2560 | 4096x2560 | 2560x4096 | 8192x2560 | 1024x2560 | 2560x4096 |

0.8B additionally has tied embeddings (248320x1024), one MTP layer, and a vision tower. Checkpoint tensor names are
`model.language_model.layers.N.mlp.{gate,up,down}_proj.weight`, `model.visual.*`, `mtp.*`; no `lm_head` tensor (tied).

**Initial scope:** text-only, FFN projections only (`ffn.gate`, `ffn.up`, `ffn.down`). Attention, DeltaNet, norms,
embeddings, RoPE, vision, and MTP are untouched. Dense FFN size per layer is `3 * hidden * ffn` parameters
(estimate of the target pool: 0.8B ≈ 11.0M, 2B ≈ 37.7M, 4B ≈ 70.8M per layer).

## Experiment matrix

| ID | Question | Method | Primary output |
|---|---|---|---|
| M1 | Is the math correct? | Unit/property tests: Givens orthogonality, topology determinism and disjointness, `RotorCascade` forward vs `dense_weight()`, exact SVD→Givens reconstruction, format round-trip, parameter-count formulas vs actual tensors. Tiny random checkpoint (`make_tiny_checkpoint`). | pass/fail test log |
| M2 | What do the real weights look like and how do methods compare on 0.8B? | Header-only inspection; conversion of selected FFNs; equal-stored-byte comparisons: dense, truncated SVD, SVD→Givens, magnitude sparse, butterfly, rotor/scaler, rotor/scaler+residual; controls: spectrum-matched random and Gaussian. Evaluate whole-model NLL/perplexity on identical windows. | baseline comparison and model-quality report directories |
| M3 | Do controlled 2B experiments support the hypothesis? | Matched-byte sweeps, activation-aware error, LM quality, byte accounting, and recovery training on 2B. | reproducible reports and paper-ready figures |
| M4 | Can structured CUDA kernels be correct and competitive? | Isolated kernels with numerical oracle, compare dense GEMV/GEMM over projection shapes and decode/prefill batch regimes. | correctness and performance evidence on a capable GPU host |
| M5 | Can an engine directly execute derived checkpoints? | Engine selection/integration; structured projections without dense materialization; semantics, numerical and generation validation. | engine conformance evidence |
| M6 | How does the approach scale and publish? | 4B experiments, long-context tests, ablations, memory/performance comparisons, full paper draft. | scaling results and paper |

### Falsification criteria

The hypothesis is unsupported at a given byte budget if any of the following is measured (pre-registered):

1. On real weights, the rotor fit never beats truncated SVD at equal stored bytes (in weight error and M2 whole-model NLL),
   across tested topologies, stage counts and residual ranks, and it also does not beat truncated SVD on control matrices.
   (Then rotors show no advantage either from structure in real weights or from the spectrum.)
2. Rotor fit error on real weights is not lower than on spectrum-matched random controls — no evidence that trained-weight
   structure is being exploited. An equal advantage on controls indicates a spectrum/capacity effect, not pretrained structure.
3. At the byte budget where perplexity degradation is "not substantial" (threshold fixed before M2, e.g. a stated relative
   ΔNLL on M2 windows), the rotor representation requires at least as many bytes as truncated SVD with the same recovery training.
4. Gains vanish after M3 recovery is applied equally to both methods, or do not persist at 2B/4B (M3/M6).

Partial outcomes (e.g. only at some byte ratios, only some families) MUST be reported as such.

## Metrics and reporting rules

- **Equal stored bytes** is the only budget axis for quality comparisons: actual serialized layer tensor bytes vs dense checkpoint bytes
  (BF16 = 2 B/param) and vs truncated-SVD bytes (same storage dtype, includes both factors). Report ratio and checkpoint-level
  file/header/index/padding/quantization-metadata overhead explicitly. Named target presets are `b075`, `b050`, `b035`,
  `b025`, and `b015` (fractions of original stored bytes). Any difference between BF16 and float32 master parameters is stated.
- **Compute is reported separately** (`flops_per_token`, measured latency); never traded against bytes in one number.
- Weight metrics: relative Frobenius, relative spectral, max abs. Activation metrics: relative Frobenius, mean row
  cosine. Implemented evaluation modules also provide mean NLL, perplexity, token count, `windows_sha256`, greedy generation
  output/token ids and time-to-first-token, prefill/decode tokens/s, multiple-choice accuracy, and synthetic passkey retrieval accuracy.
- The brief's full execution list also includes memory traffic, kernel latency, and GPU utilization. These are not yet
  implemented as evaluation metrics and require a GPU-capable host; the current host's available GPU/runtime capabilities
  must be detected rather than assumed.
- Each `ExperimentRun` directory carries environment, versions, hardware, UTC timestamp, seeds, config, and base provenance
  (`hf_id`/revision/shard sha256), alongside measured metrics and separately labeled estimates.
- Evaluate base and all variants on identical windows; fingerprint the windows.
- Multiple seeds for stochastic parts (init, `seeded` topology, random controls, window sampling); report spread, not
  just best.
- **Negative results are recorded** in `reports/experiments/` with the same detail as positive ones. Failed or
  diverged fits are published with status `failed`, not dropped.
- Formula-derived numbers (parameter counts, FLOPs, byte estimates) are labeled `estimate`; measurements are labeled
  `measured`. No result is quoted that is not backed by a run directory.
- SVD-rank-truncation gains are never attributed to rotors.

## Dataset policy

No dataset is downloaded without explicit permission, and no model weights are downloaded by code in this repo
(header-only remote reads are allowed). Until permission is given, M1 uses random token windows
(`random_windows`) and the tiny checkpoint; calibration may use `source: random`. Candidate evaluation corpora are
**proposals only**, to be confirmed with the user before any fetch: WikiText-103 validation split, a held-out slice of
FineWeb-Edu, a held-out code slice (e.g. The Stack subset), and a long-document set for long-context M6. Any
user-provided text file is consumed through `tokenize_text_file`.

## Open questions

- Which topology/stage count (default `auto = ceil(log2 n)`) best matches FFN weights; is `seeded` ever better than
  `butterfly`?
- What byte ratio is achievable at "not substantial" degradation, and what threshold defines that (fixed before E4)?
- Does the interior-width-`n` design for rectangular FFN matrices (`n = ffn`) waste capacity relative to a square
  chain plus a thin adapter?
- Is the optimization landscape for angles tractable at width ≥ 3584 without SVD-based initialization?
- How do activation-aware fits trade weight error for LM loss, and how many calibration rows are needed?
- Can the optimal comparison baseline include quantization (e.g. int8/int4 dense) at equal bytes?
- Which evaluation corpora does the user permit, and is a GPU available for E5–E7 (none on the development host)?
- Extending scope beyond FFN (attention, DeltaNet projections) and handling the MTP and vision tower.
