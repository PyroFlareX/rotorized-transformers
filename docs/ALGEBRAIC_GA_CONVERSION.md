# Algebraic tensor conversion to GA rotors

## Scope

`scripts/convert_ga_rotors.py` converts explicitly selected local floating-point two-dimensional tensors.
No training, gradient optimization, calibration, downloads, or byte-budget objective is involved. Full thin-SVD
rank is the default. Explicit truncation is an approximation and must pass the caller's error gate.

The new `ga-rotor-svd` representation is separate from the learned fixed-topology `rotor-scaler` representation.
It does not reinterpret that format, its topology, or its existing checkpoints. This construction is ordinary
linear algebra expressed as geometric-algebra rotors; it is not a claim of novel compression or faster inference.

## Derivation

Use Euclidean Clifford algebra Cl(n,0), with orthonormal basis vectors satisfying
`e_i e_j + e_j e_i = 2 delta_ij`. For a plane i<j, let `B=e_i e_j`, so `B^2=-1`.
A plane rotor is `R=a+b B`; its reverse is `R~=a-b B`, and `R R~=a^2+b^2`.

The vector sandwich is `x' = R x R^-1`, with `R^-1=R~/(a^2+b^2)`. Expanding the products gives

```
c = (a^2 - b^2) / (a^2 + b^2)
s = -2 a b / (a^2 + b^2)
x_i' = c x_i - s x_j
x_j' = s x_i + c x_j
```

Coordinates outside the plane are unchanged. Thus `R=cos(theta/2)-sin(theta/2) e_i e_j` executes the conventional
Givens rotation of angle theta. The minus sign depends on the stated basis and sandwich convention; it is not
optional. Normalizing by the rotor norm makes storage-rounding behavior explicit. Signs/reflections are stored
separately because a product of rotors alone cannot represent an orientation-reversing orthogonal map.

For W of shape (out,in), thin SVD gives `W=U diag(sigma) V^T`, where U and V have r orthonormal columns.
At full rank `r=min(out,in)`, this reconstructs any finite real W to numerical precision, including rank-deficient
matrices. With explicit r<min(out,in), it is the deterministic truncated-SVD approximation.

For either thin orthonormal factor Q of shape (m,r), eliminate entries below its diagonal with adjacent-row
Givens rotations. For column j and row i>j, take the current values u=Q[i-1,j], v=Q[i,j] and store
the single angle `theta=atan2(v,u)` in radians. When both are zero, store zero (identity).
The rotor is implicitly `R=cos(theta/2)+sin(theta/2) B`; execution uses `c=cos(theta)`, `s=-sin(theta)`.
This annihilates the lower entry without storing two coefficients. After elimination, `G Q = E diag(signs)`, where E embeds r
coordinates into m coordinates. Therefore `Q=G^-1 E diag(signs)` and `Q^T=diag(signs) E^T G`.

Direct W execution consists of:
1. Apply V's elimination rotations G_V to the input, then keep r coordinates and apply V signs.
2. Multiply by singular values.
3. Apply U signs and embed r coordinates into out coordinates.
4. Apply inverse U elimination rotations in reverse execution order.
5. Add the original bias, if present.

No dense U, V, or W is retained by the runtime operator.

## Batched elimination order

For elimination cell (j,i), use wave `t=(m-1-i)+2*j`, i>j. Within a wave, enumerate increasing j.
The plane is **adjacent rows (i-1,i)**, not rows (j,i). Equal-t pairs differ by two row positions per unit j,
so they are disjoint. The next elimination above in the same column occurs at t+1; prerequisites from the
previous column occur earlier. This permits batched disjoint rotations without a Python loop per coefficient.
The inverse factor executes waves in reverse order with inverse plane rotations.

The wave schedule and coefficient packing order are part of the new format's versioned contract.

GA format version 2 stores `left_angles` and `right_angles`, one scalar per rotation. Version 1 paired
coefficients are rejected by the current loader; historical v1 outputs are preserved, not overwritten.
For an (m,n) weight of retained rank r, the stored count excluding bias is
`r*(m+n)-r*(r+1)+3*r`. At full rank this is `m*n+2*r`: slightly more than dense, not compression.
This halves the rotation storage versus v1, but does not change the full-rank capacity or introduce fitting.

## Verification and publication

The converter checks:
- Whole weight reconstruction using bounded basis chunks, against the original tensor.
- Direct rotor action on deterministic probes, against dense action of the original tensor and original bias.
- Export/load action and reconstruction before the temporary checkpoint is atomically published.
- Finite angles, valid shapes/signs, and actual scalar counts and payload/file sizes.

The exact full-rank defaults require relative norm error <=1e-10 for FP64 or <=1e-5 for FP32 storage, plus
corresponding elementwise absolute/relative tolerances. These are new converter acceptance requirements,
not retroactive relaxation of existing FP64 math tests. Explicitly requested truncated rank uses the stated
relative-error gate and reports elementwise errors without claiming exact conversion. Thresholds and actual
errors are recorded. A failure produces an ExperimentRun failure report and no published final checkpoint.

The probe seed and fingerprint, source tensor and shard checksums, source revision when catalogued, algorithm,
rank, device, dtype, and measured errors are recorded. Probe agreement alone is not the only check: all weight
coefficients are checked by reconstruction. No language-model quality or performance claim follows from these
numerical checks.

## Usage

```bash
.venv/bin/python scripts/convert_ga_rotors.py \
  --model qwen3.5-0.8b \
  --tensor model.language_model.layers.0.mlp.up_proj.weight \
  --name qwen35-l0-up-algebraic-ga-new \
  --storage-dtype float64 --chunk-size 256
```

Repeat `--tensor` for additional matrices. `--base-dir` can select an existing safetensors directory under the
configured NAS root instead of a model key. Biases corresponding to `.weight` names are preserved automatically.
For ROCm, use `.venv-rocm/bin/python` and `--device cuda:0` after verifying GPU ownership. CPU is the default;
unavailable CUDA/HIP requests fail rather than silently falling back. Do not compete with existing GPU runs.

Default output is a new storage-configured derived directory, with atomic no-overwrite publication. Existing
base checkpoints and pre-existing protected NAS models are read-only. Optional approximation requires explicit
`--rank` and an acceptable `--max-relative-error`; omission of rank keeps every component.

To execute the saved representation:

```python
from rotor.formats.ga import load_ga_checkpoint

manifest, tensors = load_ga_checkpoint(checkpoint_path)
y = tensors["model.language_model.layers.0.mlp.up_proj.weight"](x)
```

This is a tensor operator, not automatic whole-model replacement. A verified tensor conversion does not yet
establish composed-FFN or whole-model parity. Model integration must preserve input/output dtype, biases,
SiLU/product conventions, and original module paths; test those separately before scaling.

## Historical version-1 full-projection verification

The CPU run [qwen3.5-0.8b__algebraic-ga-rotors__unrestricted__20261011T034657Z](../reports/experiments/qwen3.5-0.8b__algebraic-ga-rotors__unrestricted__20261011T034657Z/report.md)
converted the complete layer-0 `mlp.up_proj.weight` (3584 x 1024), retaining all 1024 SVD components in FP64.
The original BF16 checkpoint was read in place; no training or downloads were used.

| Check against original tensor | Relative L2 error | Maximum absolute error |
|---|---:|---:|
| Whole-weight reconstruction | 9.594744e-15 | 5.585810e-16 |
| Direct action, 16 seeded probes | 9.717941e-15 | 1.204592e-14 |
| Serialized reload, whole-weight reconstruction | 9.594744e-15 | 5.585810e-16 |
| Serialized reload, direct action | 9.717941e-15 | 1.204592e-14 |

The checkpoint was published at the configured derived location
`temp/qwen35-l0-up-algebraic-ga-f64-20261011-proof/` after both verification passes.
Its tensor payload is 58,728,448 bytes; the checkpoint total is 58,733,712 bytes.
The source projection contains 7,340,032 BF16 tensor bytes (shape x dtype size), so this correctness-first
representation is larger, not compressed. Conversion plus the initial verification took 80.65 seconds;
this is not an inference-speed measurement. Whole-model generation and composed-FFN parity were not evaluated.

Integrated verification: 41 GA tests passed; the repository suite passed with 380 tests, six skipped,
and one deselected. Ruff passed. The suite emitted an existing requires-grad-to-scalar warning in `test_math.py`.

## Version-2 angle verification and compactness

The [angle run](../reports/experiments/qwen3.5-0.8b__algebraic-ga-rotors__unrestricted__20261011T040143Z/report.md)
converted the same complete 3584 x 1024 layer-0 up projection at full rank in FP64, with the same probe fingerprint
and source tensor checksum as the historical paired-coefficient proof. Its whole-weight relative error was
8.156160e-15 and direct-action error 8.275712e-15; serialized reload reproduced both measurements.

| Representation | Stored scalars | Tensor payload bytes |
|---|---:|---:|
| Original dense BF16 | 3,670,016 | 7,340,032 |
| Version 1 paired rotors FP64 | 7,341,056 | 58,728,448 |
| Version 2 angles FP64 | 3,672,064 | 29,376,512 |

Angles remove 3,668,992 stored scalars (49.979% of the v1 total). The v2 total is still 2,048 scalars
above dense (+0.0558%), because signs and singular values accompany the angles. It is not fewer parameters
than the original, and FP64 tensor bytes remain 4.0022 times the BF16 source. At equal dtype, scalar count
and byte ratios coincide; lower dtype byte savings would not establish parameter compression.

The new checkpoint is `temp/qwen35-l0-up-algebraic-ga-angles-f64-proof/`, total 29,382,180 bytes.
Historical checkpoints remain untouched. GA tests: 48 passed; full suite: 387 passed, six skipped,
one deselected; Ruff passed. Whole-model language quality and inference performance remain unmeasured.

## Clifford block-projection pilot

`scripts/explore_clifford.py` tests a different restricted family: blockwise left multiplication of
multivectors in Euclidean Cl(2,0) or Cl(3,0). These have q=4 or q=8 blades, respectively, with basis order
given by ascending bit masks. This is not exhaustive rotor decomposition, and not a claim that pretrained
features already carry geometric meaning. Cl(0,2), not Cl(2,0), is the quaternion algebra; the pilot uses
positive-square generators explicitly.

Group a dense weight into q x q blocks M. The blade multiplication matrices L_b obey
`<L_b,L_c>_F=q*delta_bc`. The best Frobenius projection onto left geometric multiplication is therefore
`c_b=<M,L_b>_F/q`, with reconstructed block `sum_b c_b L_b`. This is a linear orthogonal projection,
not gradient training. The residual must be orthogonal to every L_b, which is verified on small matrices.
Each block stores q coefficients instead of q², so an m x n projection stores mn/q coefficients.
Bias, if supported by a future conversion, must be counted separately; this weight-only pilot does not
replace or execute any original biases or whole-model modules.

Contiguous grouping and fixed seeded input/output permutations test sensitivity to coordinate assignment.
Every full-projection arm uses identical Gaussian input probes; these are numerical diagnostics, not a
language-quality corpus. Direct compressed execution is checked against the reconstructed approximation
separately from approximation error against the original weight. Random norm-matched weights provide a
structure control; matched coefficient-budget SVD provides a different restricted-family comparison.
This pilot does not export an inference checkpoint or establish a full-model go decision.

### Measured first pilot

The [recorded pilot](../reports/experiments/qwen3.5-0.8b__clifford-block-projection__model.language_model.layers.0.mlp.up_proj.weight.probes-64__20261011T184547Z/report.md)
used the complete layer-0 up projection (3584 x 1024), FP64 coefficients, 64 identical Gaussian probes,
and contiguous plus three fixed seeded groupings. No checkpoint was exported.

| Contiguous arm | Coefficients | Weight relative error | Probe action relative error | Retained squared weight norm |
|---|---:|---:|---:|---:|
| q=4 Clifford | 917,504 | 0.866250 | 0.865462 | 24.961% |
| q=8 Clifford | 458,752 | 0.935543 | 0.936111 | 12.476% |
| SVD, rank 199, within q=4 scalar budget | 917,191 | 0.795848 | 0.795561 | 36.663% |
| SVD, rank 99, within q=8 scalar budget | 456,291 | 0.892804 | 0.892286 | 20.290% |

Clifford coefficients alone occupy 7,340,032 bytes (q=4) or 3,670,016 bytes (q=8); explicit permutation
arrays add 36,864 bytes in this implementation. Thus FP64 q=4 coefficients equal the source BF16 tensor
size despite a fourfold coefficient reduction; q=8 coefficients halve source BF16 bytes before metadata.
These are computed tensor sizes, not serialized checkpoint sizes.

All eight source arms and eight Gaussian-control arms passed direct operator/reconstruction parity,
with source-arm maximum absolute mismatch at most 1.610e-15. Grouping changes did not rescue reconstruction:
source retained-energy ranges were 24.954–24.961% for q=4 and 12.461–12.493% for q=8. Gaussian controls
retained 24.952–25.050% and 12.480–12.525%, respectively. An isotropic weight's expected retained fraction
is the subspace dimension ratio q/q²=1/q (estimate). The observed source is consistent with this background
level; the pilot provides no evidence of useful alignment with this fixed Clifford family.

Decision: do not expand this single-left-product family to full-model conversion on these measurements.
This is not a rejection of all GA factorization or alternative bases/compositions. Four-step bounded q-by-q
baseline rows are recorded separately and do not establish a full-projection trained-method ranking.
Verification: 19 Clifford tests passed; full repository suite 406 passed, six skipped, one deselected;
Ruff passed. Random probes are not language-quality evaluation.
