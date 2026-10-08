# Research plan

## Hypothesis

> Pretrained Transformer linear transformations may admit significantly smaller compositional representations using rotors and scalings than their ordinary dense coefficient representations, without substantial degradation in language-model quality.

This is a hypothesis to be tested, not a result. Nothing in this repository claims it holds. Every number in this
document that comes from a formula is an **estimate**; measured numbers only appear in run records under
`reports/experiments/`.

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
| E0 | Is the math correct? | Unit/property tests: Givens orthogonality, topology determinism and disjointness, `RotorCascade` forward vs `dense_weight()`, exact SVD→Givens reconstruction, format round-trip, parameter-count formulas vs actual tensors. Tiny random checkpoint (`make_tiny_checkpoint`). | pass/fail test log |
| E1 | What do the real weights look like? | Header-only inspection of models A/B/C (`remote_tensors`); shapes, dtypes, tied embeddings, layer types. If weights are locally present: singular-spectrum summaries of FFN matrices. | shape table; spectrum stats |
| E2 | In weight space, at **equal stored bytes**, does a rotor cascade beat truncated SVD? | `equal_budget_baselines`: truncated SVD, SVD→Givens, rotor weight-fit, across topologies (brick/butterfly/seeded), stages per rotor, number of rotors, residual rank; byte ratios swept. **Controls:** spectrum-matched random (same singular values, Haar singular vectors) and iid Gaussian (same Frobenius norm) to separate exploitation of learned structure from mere spectrum. | rel. Frobenius/spectral error vs bytes per matrix and control |
| E3 | Do activation-aware fits help? | Fit on `‖XW^T − XŴ^T‖` with calibration inputs X from forward hooks (`collect_inputs`); compare to weight-fit and activation-aware SVD at equal bytes. | activation rel. error, row cosine |
| E4 | LM quality | `convert` + `load_converted`; `evaluate_lm` on **identical token windows** (same `windows_sha256`) for base and each derived model; validation NLL/perplexity; long-context `nll_by_position`. | mean NLL, perplexity Δ vs base and vs equal-byte truncated SVD |
| E5 | Can recovery close the gap? | KL distillation (`train_recovery`) updating only `RotorLinear` parameters, base frozen as teacher; same budget for the SVD baseline's trainable factors. | loss history, post-recovery NLL |
| E6 | Does it scale? | Repeat E2–E5 on 2B (B) and 4B (C) with configs fixed from A. | trend vs model size |
| E7 | Throughput / long context | `linear_microbench`, `time_callable`; Python reference first, C++ runtime later. Memory traffic and latency at batch sizes of interest; long-context behavior. | median/p10/p90 ms |

### Falsification criteria

The hypothesis is unsupported at a given byte budget if any of the following is measured (pre-registered):

1. On real weights, the rotor fit never beats truncated SVD at equal stored bytes (in weight error **and** in E4 NLL),
   across all tested topologies, stage counts and residual ranks, **and** it also does not beat truncated SVD on the
   control matrices. (Then rotors give no advantage either from structure in real weights or from the spectrum.)
2. Rotor fit error on real weights is not lower than on spectrum-matched random controls — no evidence that the
   structure present in trained weights is being exploited. A rotor advantage that appears equally on controls is a
   spectrum/capacity effect, not evidence about pretrained weights.
3. At the byte budget where perplexity degradation is "not substantial" (threshold fixed before running, e.g. a
   stated relative ΔNLL on E4 windows), the rotor representation requires at least as many bytes as truncated SVD
   with the same recovery training.
4. Gains vanish after E5 recovery is applied equally to both methods, or do not persist at 2B/4B (E6).

Partial outcomes (e.g. only at some byte ratios, only some families) MUST be reported as such.

## Metrics and reporting rules

- **Equal stored bytes** is the only budget axis for quality comparisons: `structured_bytes(spec, storage_dtype)` vs
  the dense checkpoint bytes (BF16 = 2 B/param) and vs truncated-SVD bytes (same storage dtype, includes both
  factors). Report the byte ratio explicitly. Any difference between BF16 and float32 master parameters is stated.
- **Compute is reported separately** (`flops_per_token`, measured latency); never traded against bytes in one number.
- Weight metrics: relative Frobenius, relative spectral, max abs. Activation metrics: relative Frobenius, mean row
  cosine. LM metrics: mean NLL, perplexity, token count, `windows_sha256`.
- Every run record (`write_run_record`) carries environment, versions, hardware, UTC timestamp, seeds (fit seed,
  topology seed, window seed), config, and the base `hf_id`/`revision`/shard sha256.
- Evaluate base and all variants on identical windows; fingerprint the windows.
- Multiple seeds for stochastic parts (init, `seeded` topology, random controls, window sampling); report spread, not
  just best.
- **Negative results are recorded** in `reports/experiments/` with the same detail as positive ones. Failed or
  diverged fits are kept, not dropped.
- Formula-derived numbers (parameter counts, FLOPs, byte estimates) are labeled `estimate`; measurements are labeled
  `measured`. No result is quoted that is not backed by a run record.
- SVD-rank-truncation gains are never attributed to rotors.

## Dataset policy

No dataset is downloaded without explicit permission, and no model weights are downloaded by code in this repo
(header-only remote reads are allowed). Until permission is given, E0 uses random token windows
(`random_windows`) and the tiny checkpoint; calibration may use `source: random`. Candidate evaluation corpora are
**proposals only**, to be confirmed with the user before any fetch: WikiText-103 validation split, a held-out slice of
FineWeb-Edu, a held-out code slice (e.g. The Stack subset), and a long-document set for long-context E7. Any
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
