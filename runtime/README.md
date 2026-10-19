# runtime/

Native (C/C++/CUDA) side of the Rotor Transformer project. It exists so that structured `rotor-scaler`
checkpoints can eventually be executed by an inference engine **without Python**.

## Purpose

The Python package (`src/rotor/`) fits and evaluates structured layers. The runtime consumes the *result*:
a versioned directory (`rotor_manifest.json` + `rotor_weights.safetensors`, format `rotor-scaler` v1) and
applies `y = D_0 P(R_1 D_1 ... R_k E(D_k x)) + U(Vx) + bias` per layer. Whether this is ever faster than a
dense GEMM is an open, **unmeasured** question; see `docs/INFERENCE_ENGINE_PLAN.md`.

## Normative conversion/runtime contract

[`interfaces/README.md`](interfaces/README.md) is the normative specification for the converter↔runtime boundary: checkpoint files, tensor names/shapes/dtypes, execution semantics, deterministic topology regeneration, version compatibility, required conformance tests, and the prohibition on Python/Transformers runtime dependencies. Read it alongside the Python authority in `src/rotor/formats/spec.py`, `interfaces/rotor_format.h`, and `interfaces/rotor_topology.hpp` before implementing a loader or operator.

## Independence from Python

- Nothing under `runtime/` imports or links Python, torch, or numpy.
- The only shared contract is the on-disk format and the deterministic topology algorithms (SplitMix64
  permutation, brick / butterfly / seeded pair schedules). Checkpoints store no coordinate indices; the runtime
  regenerates pairs from `(kind, width, stage, seed)`.
- The Python spec in `src/rotor/formats/spec.py` is normative. `interfaces/rotor_format.h` mirrors its constants.
  `interfaces/rotor_topology.hpp` mirrors `src/rotor/math/topology.py` bit-exactly and carries a double/float
  reference of the cascade semantics. `tests/test_cxx_parity.py` (marker `cxx`) compiles the header with the
  system C++ compiler and checks pairs for exact equality and cascade outputs against `RotorCascade`
  (float64, atol 1e-10).

The engine decision, upstream support/license survey, CUDA/Vulkan kernel research plan, and sequencing gates are in
[`docs/INFERENCE_ENGINE_PLAN.md`](../docs/INFERENCE_ENGINE_PLAN.md).

## Layout

```
runtime/
  README.md
  interfaces/          # header-only contract shared with Python
    rotor_format.h     #   C-compatible constants + POD `rotor_cascade_spec`
    rotor_topology.hpp #   C++17 `rotor::` topology mirror + reference apply_stage / apply_cascade
  kernels/             # fused GPU/CPU kernels (empty)
  fork/                # inference-engine fork or submodule (empty)
```

## Status

**Optional Python inference backend:** The evaluator offers an explicit, opt-in fused rotor backend for
converted checkpoints, implemented separately from this native-runtime directory. Select it with
`python scripts/evaluate_prompts.py --ckpt <checkpoint> --rotor-backend triton --device cuda:0`.
The CLI defaults to `--rotor-backend reference`; base-model runs use the base backend and cannot select
`triton`. This optional backend requires Triton and a supported CUDA or ROCm GPU. Selecting it on an
unsupported device or without its dependencies fails explicitly; there is no automatic fallback. It is
inference-only. Compare its outputs with a compatible base/reference results JSONL using `--compare-to`.
The selected rotor backend is recorded as runtime provenance, not a generation setting, so otherwise-matched
reference and fused runs can be compared. No speedup or hardware qualification is claimed here.


- **Present:** the reference CPU headers in `interfaces/` only. They are correctness oracles (one vector at a
  time, no SIMD, no CUDA), not performance code.
- **Empty:** `kernels/` and `fork/`. No engine has been selected; selection is gated on verifying that a candidate
  actually supports the hybrid Qwen3.5 architecture (see the plan). Nothing here has been benchmarked.

## Plan for `fork/`

Engine code will live in an **independently versioned** fork, added either as a git submodule under
`runtime/fork/<engine>/` or as a separate repository checked out there, pinned to an explicit upstream commit.
Rotor-specific changes stay on a dedicated branch so upstream rebases remain mechanical, and this repo records only
the pinned commit plus the interface headers. Fork-local code includes `interfaces/` rather than copying it, so the
format and topology definitions have exactly one source of truth.

## Rules

- A change to the format or topology algorithms requires bumping `FORMAT_VERSION` (Python and `rotor_format.h`
  together) and updating the parity test.
- Model artifacts are read from the storage roots resolved by `rotor.storage`; the runtime never writes into the
  base model directory.
