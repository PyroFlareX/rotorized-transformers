# Rotor-scaler conversion ↔ runtime interface (normative)

This document defines the boundary between the Python converter and any native inference runtime. The Python format definition in [`src/rotor/formats/spec.py`](../../src/rotor/formats/spec.py) is authoritative; the C-compatible declarations are mirrored in [`rotor_format.h`](rotor_format.h), and deterministic topology/reference math is mirrored in [`rotor_topology.hpp`](rotor_topology.hpp). If these disagree, the Python definition and format version govern; resolve the mismatch before consuming affected checkpoints.

The runtime MUST remain independent of Python, PyTorch, NumPy, and Transformers. It MAY implement the manifest JSON and safetensors readers in native code or use native dependencies. The runtime MUST NOT import, link, or invoke Python or Transformers as part of loading or executing a derived checkpoint.

## Files and provenance

A derived checkpoint directory contains:

```text
rotor_manifest.json         UTF-8 JSON manifest
rotor_weights.safetensors   structured tensors for replaced projections
base_metadata/              copies of small base files (such as config/tokenizer)
```

Only converted projection tensors are stored in the derived checkpoint. The unmodified base model weights are not copied: the manifest's base provenance identifies the original base path, revision, shard hashes, tied-embedding flag, and architectures. A runtime must obtain all untouched tensors from that exact base checkpoint and must not modify it. Model-artifact path resolution and write protection remain governed by `rotor.storage`; the runtime must not infer alternative base files or substitute a different revision silently.

## Manifest and compatibility

The manifest format name is `rotor-scaler`; the current `FORMAT_VERSION` is **1**. Its JSON representation is the `Manifest` described in `src/rotor/formats/manifest.py`, including base provenance and per-layer entries. Each layer records its module path, source tensor, family, cascade spec, storage dtype, bias presence, dense/structured parameter and byte counts, and fit metrics. Treat counts/metrics as descriptive metadata, not as substitutes for tensor shape validation.

A reader MUST validate the format name and version before interpreting layer data. It MUST reject an unknown format name, a version newer than it supports, unknown topology/adapter names, unsupported dtype, malformed required fields, or inconsistent tensor shapes. A reader MUST NOT guess semantics for a newer version. Any semantic change to tensor layout, dtype meaning, topology, angle order, forward equation, or metadata interpretation MUST bump `FORMAT_VERSION` in Python and `rotor_format.h`, update this document, and update conformance/golden tests. Readers may support explicitly implemented older versions, but compatibility is never inferred; a reader that does not implement a version MUST reject it. A derived repackaging such as GGUF is not the source of truth and must identify the source manifest (for example by hash) and preserve provenance.

## Tensor keys, shapes, and dtypes

For a converted module path `<m>` (the Hugging Face module path, for example `model.layers.3.mlp.up_proj`), tensors in `rotor_weights.safetensors` use these exact keys and shapes. Let `in = spec.in_features`, `out = spec.out_features`, `n = max(in, out)`, `k = spec.num_rotors`, `m = spec.stages_per_rotor`, and `A = angle_count(kind, n, k*m, seed)` per rotor as defined by the topology implementation.

| Tensor key | Shape | Presence / meaning |
|---|---:|---|
| `<m>.angles` | `(k, A)` | Required; row `i` parameterizes rotor `R_(i+1)`. Within each row, stage-0 pair angles precede stage-1 angles, and so on, in canonical pair order. |
| `<m>.d_in` | `(in,)` | Required; input diagonal `D_k`. |
| `<m>.d_mid` | `(k-1, n)` | Required; row-major intermediate diagonals `D_1 ... D_(k-1)`. For `k=1`, the leading dimension is zero. |
| `<m>.d_out` | `(out,)` | Required; output diagonal `D_0`. |
| `<m>.residual.U` | `(out, r)` | Present only when residual rank `r > 0`. |
| `<m>.residual.V` | `(r, in)` | Present only when residual rank `r > 0`. |
| `<m>.bias` | `(out,)` | Present iff the layer spec says it has bias. |

Each tensor's stored dtype is the layer's manifest `storage_dtype`, one of `float32`, `bfloat16`, or `float16` in v1. Readers must preserve its declared numerical meaning and must not reinterpret raw bits as another dtype. No coordinate-index tensor is serialized. The loader must verify all required tensors exist, optional tensors agree with `residual_rank`/`has_bias`, and every shape matches both this table and the spec before dispatching execution.

## Forward semantics

For an input row/vector `x` of length `in`, the converted linear layer computes

```text
y = D_0 * P(R_1 D_1 R_2 ... D_(k-1) R_k E(D_k * x)) + U (V x) + bias
```

The residual and bias terms are omitted when absent. `E` zero-pads `in` coordinates to `n`; `P` keeps the first `out` coordinates. Diagonal multiplication is elementwise. The product convention is `R_i = S_1 S_2 ... S_m`; applied to a vector, `S_m` acts first. For a pair `(a,b)` and angle `t`, the stage updates `x_a' = cos(t) x_a - sin(t) x_b` and `x_b' = sin(t) x_a + cos(t) x_b`. Pair slots are disjoint within a stage. Accumulation/approximation policy in optimized kernels must preserve this defined operation within the kernel's declared numerical tolerance; the reference implementation is the correctness oracle.

## Deterministic topology generation

The runtime MUST regenerate all coordinate pairs from `(kind, n, stage, seed)` using `rotor_topology.hpp`'s `stage_pairs` algorithm; it MUST NOT require or invent serialized indices. `kind` is one of `brick`, `butterfly`, or `seeded`. The manifest seed is interpreted modulo `2^64` (two's-complement reduction for negative values). Seeded topology uses the exact SplitMix64/Fisher–Yates stream, including modulo-bounded draws; ordering is canonical and determines angle-slot assignment. See comments and implementation in [`rotor_topology.hpp`](rotor_topology.hpp), which mirrors [`src/rotor/math/topology.py`](../../src/rotor/math/topology.py). Cache generated pairs only with a complete key including topology kind, width, stage, and seed.

## Conformance requirements

Before a runtime claims compatibility, its reader and execution path MUST satisfy all of the following:

1. Pass [`tests/test_cxx_parity.py`](../../tests/test_cxx_parity.py): topology pair lists must exactly equal Python for the covered kinds, odd and non-power-of-two widths, stages, and seeds; reference cascade results must match `RotorCascade` float64 with the test's strict tolerance.
2. Execute fixed golden inputs and parameters produced by Python for representative square and rectangular shapes, all topology kinds, `k=1` and multi-rotor cases, optional residual, optional bias, and the supported storage dtypes. Compare against expected outputs with a declared tolerance and report max-absolute and relative error. Golden cases are required in addition to topology parity; do not replace them with a successful load-only check.
3. Validate malformed/mismatched checkpoint cases fail closed: newer version, unknown names/dtypes, missing required tensor, unexpected shape, and residual/bias presence inconsistent with spec.
4. At integration, compare converted tiny-checkpoint logits and fixed-prompt argmaxes against Python reference, then compare dense-base logits/perplexity before making any performance claim.

No runtime may claim full checkpoint compatibility based only on a standalone tensor kernel test. Numerical results and performance are separate: correctness tolerances are explicit, while all speed claims require measured benchmark evidence on named hardware.
