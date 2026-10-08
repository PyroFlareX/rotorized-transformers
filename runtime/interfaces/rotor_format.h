/*
 * rotor_format.h -- C-compatible constants and POD descriptors for the `rotor-scaler` format.
 *
 * Mirrors src/rotor/formats/spec.py (the normative Python definition). If the two disagree, the
 * Python module and FORMAT_VERSION are authoritative; bump ROTOR_FORMAT_VERSION on any change.
 *
 * A derived checkpoint is a directory:
 *     rotor_manifest.json         UTF-8 JSON manifest
 *     rotor_weights.safetensors   structured tensors named "<module>.<suffix>"
 *     base_metadata/              copies of small base files (config, tokenizer)
 *
 * Semantics of one layer (normative):
 *     y = D_0 * P(R_1 D_1 R_2 ... D_{k-1} R_k E(D_k * x)) + U (V x) + bias
 *   E zero-pads `in` -> width = max(in, out); P keeps the first `out` coordinates.
 *   R_i = S_1 S_2 ... S_m; applied to a vector, stage S_m acts first.
 *   Pair (a, b), angle t:  x_a' = cos(t) x_a - sin(t) x_b,  x_b' = sin(t) x_a + cos(t) x_b.
 */
#ifndef ROTOR_FORMAT_H
#define ROTOR_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROTOR_FORMAT_NAME "rotor-scaler"
#define ROTOR_FORMAT_VERSION 1

#define ROTOR_MANIFEST_FILENAME "rotor_manifest.json"
#define ROTOR_WEIGHTS_FILENAME "rotor_weights.safetensors"
#define ROTOR_BASE_METADATA_DIRNAME "base_metadata"

/* Tensor key = "<module>." + suffix, e.g. "model.layers.3.mlp.up_proj.angles". */
#define ROTOR_TENSOR_ANGLES "angles"           /* (num_rotors, angles_per_rotor); row i = R_{i+1} */
#define ROTOR_TENSOR_D_IN "d_in"               /* (in_features,)            D_k                    */
#define ROTOR_TENSOR_D_MID "d_mid"             /* (num_rotors - 1, width)   D_1 .. D_{k-1}         */
#define ROTOR_TENSOR_D_OUT "d_out"             /* (out_features,)           D_0                    */
#define ROTOR_TENSOR_RESIDUAL_U "residual.U"   /* (out_features, rank)      optional               */
#define ROTOR_TENSOR_RESIDUAL_V "residual.V"   /* (rank, in_features)       optional               */
#define ROTOR_TENSOR_BIAS "bias"               /* (out_features,)           optional               */

/* Topology kinds; the manifest stores the strings, the integer values are runtime-internal. */
#define ROTOR_TOPOLOGY_NAME_BRICK "brick"
#define ROTOR_TOPOLOGY_NAME_BUTTERFLY "butterfly"
#define ROTOR_TOPOLOGY_NAME_SEEDED "seeded"

enum rotor_topology_kind {
    ROTOR_TOPOLOGY_BRICK = 0,
    ROTOR_TOPOLOGY_BUTTERFLY = 1,
    ROTOR_TOPOLOGY_SEEDED = 2
};

/* Rectangular adapters; the manifest stores the string. */
#define ROTOR_ADAPTER_NAME_PAD_TRUNCATE "pad_truncate"

enum rotor_adapter_kind {
    ROTOR_ADAPTER_PAD_TRUNCATE = 0
};

/* Storage dtypes supported by the format (SUPPORTED_STORAGE_DTYPES in spec.py). */
enum rotor_storage_dtype {
    ROTOR_DTYPE_FLOAT32 = 0,
    ROTOR_DTYPE_BFLOAT16 = 1,
    ROTOR_DTYPE_FLOAT16 = 2
};

static inline int rotor_bytes_per_param(int dtype) {
    switch (dtype) {
        case ROTOR_DTYPE_FLOAT32: return 4;
        case ROTOR_DTYPE_BFLOAT16: return 2;
        case ROTOR_DTYPE_FLOAT16: return 2;
        default: return -1;
    }
}

/*
 * Structure of one cascade (no parameter values). Mirrors rotor.math.cascade.CascadeSpec plus the
 * bias flag. `topology_seed` is the manifest integer interpreted modulo 2^64 (two's complement
 * for negative values), which is exactly how the Python topology code reduces it.
 */
typedef struct rotor_cascade_spec {
    uint32_t in_features;
    uint32_t out_features;
    uint32_t num_rotors;        /* k >= 1 */
    uint32_t stages_per_rotor;  /* m >= 0 */
    uint32_t topology_kind;     /* enum rotor_topology_kind */
    uint32_t adapter_kind;      /* enum rotor_adapter_kind */
    uint32_t residual_rank;     /* r >= 0; 0 = no residual */
    uint32_t has_bias;          /* 0 / 1 */
    uint64_t topology_seed;
} rotor_cascade_spec;

/* Square working width n = max(in, out). */
static inline uint32_t rotor_cascade_width(const rotor_cascade_spec* s) {
    return s->in_features > s->out_features ? s->in_features : s->out_features;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROTOR_FORMAT_H */
