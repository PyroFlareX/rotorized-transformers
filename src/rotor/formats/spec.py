"""Versioned structured-weight format ``rotor-scaler``. Shared contract with the future C++ runtime.

A derived checkpoint is a directory::

    <derived_root>/<name>/
        rotor_manifest.json         # this module's Manifest (UTF-8 JSON)
        rotor_weights.safetensors   # structured tensors, keys below
        base_metadata/              # copies of small base files (config, tokenizer) for provenance

It stores only the replaced projections; every untouched tensor is read from the base checkpoint
recorded in ``manifest.base`` (path + revision + sha256), so base weights are never duplicated.

Tensor keys, for module ``<m>`` (HF module path, e.g. ``model.layers.3.mlp.up_proj``)::

    <m>.angles       (num_rotors, angles_per_rotor)   rotor i (R_{i+1}) = row i
    <m>.d_in         (in_features,)                   D_k
    <m>.d_mid        (num_rotors - 1, width)          D_1 .. D_{k-1}
    <m>.d_out        (out_features,)                  D_0
    <m>.residual.U   (out_features, r)                optional
    <m>.residual.V   (r, in_features)                 optional
    <m>.bias         (out_features,)                  optional

Semantics (normative; ``runtime/interfaces/rotor_format.h`` mirrors this):

- ``y = D_0 * P(R_1 D_1 R_2 ... D_{k-1} R_k E(D_k * x)) + U (V x) + bias``
- ``E`` zero-pads ``in`` -> ``width = max(in, out)``, ``P`` keeps the first ``out`` coordinates.
- ``R_i = S_1 S_2 ... S_m``; applied to a vector, stage ``S_m`` acts first.
- Stage ``s`` pairs come from ``rotor.math.topology.stage_pairs(kind, width, s, seed)``; angle
  slots within a rotor row are stage 0 pairs, then stage 1 pairs, ... in canonical pair order.
- Pair ``(a, b)``, angle ``t``: ``x_a' = c x_a - s x_b``, ``x_b' = s x_a + c x_b``.
"""

from __future__ import annotations

FORMAT_NAME = "rotor-scaler"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "rotor_manifest.json"
WEIGHTS_FILENAME = "rotor_weights.safetensors"
BASE_METADATA_DIRNAME = "base_metadata"

TENSOR_SUFFIXES = ("angles", "d_in", "d_mid", "d_out", "residual.U", "residual.V", "bias")

SUPPORTED_STORAGE_DTYPES = {"float32": 4, "bfloat16": 2, "float16": 2}


def bytes_per_param(dtype: str) -> int:
    try:
        return SUPPORTED_STORAGE_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f"unsupported storage dtype {dtype!r}; expected {sorted(SUPPORTED_STORAGE_DTYPES)}") from e
