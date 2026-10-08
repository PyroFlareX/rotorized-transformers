// rotor_topology.hpp -- header-only C++17 reference mirror of the rotor-scaler math.
//
// Bit-exact mirror of src/rotor/math/topology.py (splitmix64, seeded_permutation, stage_pairs,
// angle_count) plus a *reference* CPU implementation of the normative cascade semantics from
// src/rotor/formats/spec.py. It is a correctness oracle for future kernels, not a fast path:
// no SIMD, no CUDA, one vector at a time.
//
//   y = D_0 * P(R_1 D_1 R_2 ... D_{k-1} R_k E(D_k * x)) + U (V x) + bias
//   R_i = S_1 S_2 ... S_m  (stage S_m acts first on a vector)
//   pair (a, b), angle t:  x_a' = cos(t) x_a - sin(t) x_b,  x_b' = sin(t) x_a + cos(t) x_b
//
// Any change here must be matched by src/rotor/math/topology.py (and vice versa);
// tests/test_cxx_parity.py compares the two exactly.
#ifndef ROTOR_TOPOLOGY_HPP
#define ROTOR_TOPOLOGY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rotor_format.h"

namespace rotor {

using TopologyKind = rotor_topology_kind;

struct Pair {
    uint32_t a;
    uint32_t b;
    bool operator==(const Pair& o) const { return a == o.a && b == o.b; }
};

inline TopologyKind topology_kind_from_string(const std::string& s) {
    if (s == ROTOR_TOPOLOGY_NAME_BRICK) return ROTOR_TOPOLOGY_BRICK;
    if (s == ROTOR_TOPOLOGY_NAME_BUTTERFLY) return ROTOR_TOPOLOGY_BUTTERFLY;
    if (s == ROTOR_TOPOLOGY_NAME_SEEDED) return ROTOR_TOPOLOGY_SEEDED;
    throw std::invalid_argument("unknown topology kind '" + s + "'; expected brick, butterfly or seeded");
}

// One SplitMix64 step. Advances `state` and returns the output.
inline uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Fisher-Yates permutation of 0..n-1; stream state = seed * 0x100000001B3 + stage (mod 2^64);
// bounded draw = output % (i + 1), i from n-1 down to 1.
inline std::vector<uint32_t> seeded_permutation(uint32_t n, uint64_t seed, uint64_t stage) {
    uint64_t state = seed * 0x100000001B3ULL + stage;
    std::vector<uint32_t> perm(n);
    for (uint32_t i = 0; i < n; ++i) perm[i] = i;
    for (uint32_t i = n; i-- > 1;) {
        const uint64_t out = splitmix64(state);
        const uint32_t j = static_cast<uint32_t>(out % (static_cast<uint64_t>(i) + 1));
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

// max(1, bit_length(n - 1)); n >= 2.
inline uint32_t num_strides(uint32_t n) {
    uint32_t bits = 0;
    for (uint32_t v = n - 1; v != 0; v >>= 1) ++bits;
    return bits < 1 ? 1 : bits;
}

// Disjoint pairs for rotation stage `stage` (0-based) of an n-dim cascade, in canonical order
// (sorted by first index). This order defines the layout of the flat angle vector.
inline std::vector<Pair> stage_pairs(TopologyKind kind, uint32_t n, uint64_t stage, uint64_t seed = 0) {
    std::vector<Pair> pairs;
    if (n < 2) return pairs;
    switch (kind) {
        case ROTOR_TOPOLOGY_BRICK: {
            const uint32_t off = static_cast<uint32_t>(stage % 2);
            for (uint32_t i = off; i + 1 < n; i += 2) pairs.push_back({i, i + 1});
            break;
        }
        case ROTOR_TOPOLOGY_BUTTERFLY: {
            const uint32_t d = uint32_t{1} << (stage % num_strides(n));
            for (uint32_t i = 0; i + d < n; ++i) {
                if ((i / d) % 2 == 0) pairs.push_back({i, i + d});
            }
            break;
        }
        case ROTOR_TOPOLOGY_SEEDED: {
            const std::vector<uint32_t> p = seeded_permutation(n, seed, stage);
            pairs.reserve(n / 2);
            for (uint32_t j = 0; j < n / 2; ++j) {
                const uint32_t u = p[2 * j], v = p[2 * j + 1];
                pairs.push_back({std::min(u, v), std::max(u, v)});
            }
            std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) {
                return x.a != y.a ? x.a < y.a : x.b < y.b;
            });
            break;
        }
        default:
            throw std::invalid_argument("unknown topology kind");
    }
    return pairs;
}

inline std::vector<Pair> stage_pairs(const std::string& kind, uint32_t n, uint64_t stage, uint64_t seed = 0) {
    return stage_pairs(topology_kind_from_string(kind), n, stage, seed);
}

// Total angle count of `num_stages` stages (= angles per rotor).
inline uint64_t angle_count(TopologyKind kind, uint32_t n, uint64_t num_stages, uint64_t seed = 0) {
    uint64_t total = 0;
    for (uint64_t s = 0; s < num_stages; ++s) total += stage_pairs(kind, n, s, seed).size();
    return total;
}

// Reference stage application on one vector x[0..n), in place. theta has pairs.size() entries.
// transpose == true applies the inverse stage (same as negating every angle).
template <class T>
void apply_stage(T* x, uint32_t n, const std::vector<Pair>& pairs, const T* theta, bool transpose = false) {
    for (std::size_t p = 0; p < pairs.size(); ++p) {
        const Pair pr = pairs[p];
        if (pr.a >= n || pr.b >= n) throw std::out_of_range("rotor pair index outside vector");
        const T c = std::cos(theta[p]);
        const T s = transpose ? -std::sin(theta[p]) : std::sin(theta[p]);
        const T xa = x[pr.a];
        const T xb = x[pr.b];
        x[pr.a] = c * xa - s * xb;
        x[pr.b] = s * xa + c * xb;
    }
}

// Non-owning view of one layer's parameters (format layout, see rotor_format.h).
//   angles : num_rotors * angles_per_rotor     (row i = R_{i+1}; stage 0 pairs first)
//   d_in   : in_features                       (D_k)
//   d_mid  : (num_rotors - 1) * width          (D_1 .. D_{k-1}, row-major)
//   d_out  : out_features                      (D_0)
//   res_u  : out_features * residual_rank      (row-major)   -- may be null if residual_rank == 0
//   res_v  : residual_rank * in_features       (row-major)   -- may be null if residual_rank == 0
//   bias   : out_features                      -- may be null if has_bias == 0
template <class T>
struct CascadeParams {
    const T* angles = nullptr;
    const T* d_in = nullptr;
    const T* d_mid = nullptr;
    const T* d_out = nullptr;
    const T* res_u = nullptr;
    const T* res_v = nullptr;
    const T* bias = nullptr;
};

// Reference forward of one cascade on one input vector x[0..in) -> y[0..out).
template <class T>
void apply_cascade(const rotor_cascade_spec& spec, const CascadeParams<T>& w, const T* x, T* y) {
    if (spec.num_rotors < 1) throw std::invalid_argument("num_rotors must be >= 1");
    if (spec.adapter_kind != ROTOR_ADAPTER_PAD_TRUNCATE) throw std::invalid_argument("unsupported adapter kind");
    const uint32_t in = spec.in_features, out = spec.out_features;
    const uint32_t n = rotor_cascade_width(&spec);
    const uint32_t k = spec.num_rotors, m = spec.stages_per_rotor;
    const TopologyKind kind = static_cast<TopologyKind>(spec.topology_kind);

    std::vector<std::vector<Pair>> stages(m);
    std::vector<uint64_t> offsets(m + 1, 0);
    for (uint32_t s = 0; s < m; ++s) {
        stages[s] = stage_pairs(kind, n, s, spec.topology_seed);
        offsets[s + 1] = offsets[s] + stages[s].size();
    }
    const uint64_t per_rotor = offsets[m];

    std::vector<T> h(n, T(0));
    for (uint32_t j = 0; j < in; ++j) h[j] = x[j] * w.d_in[j];

    for (uint32_t i = k; i-- > 0;) {  // R_k first, then D_{k-1}, ..., R_1
        const T* theta = w.angles + static_cast<uint64_t>(i) * per_rotor;
        for (uint32_t s = m; s-- > 0;) {  // R = S_1 ... S_m: S_m acts first
            apply_stage<T>(h.data(), n, stages[s], theta + offsets[s], false);
        }
        if (i > 0) {
            const T* d = w.d_mid + static_cast<uint64_t>(i - 1) * n;
            for (uint32_t j = 0; j < n; ++j) h[j] *= d[j];
        }
    }

    for (uint32_t j = 0; j < out; ++j) y[j] = h[j] * w.d_out[j];

    const uint32_t r = spec.residual_rank;
    if (r > 0) {
        std::vector<T> vx(r, T(0));  // V x on the raw (unscaled) input
        for (uint32_t q = 0; q < r; ++q) {
            T acc = T(0);
            for (uint32_t j = 0; j < in; ++j) acc += w.res_v[static_cast<uint64_t>(q) * in + j] * x[j];
            vx[q] = acc;
        }
        for (uint32_t o = 0; o < out; ++o) {
            T acc = T(0);
            for (uint32_t q = 0; q < r; ++q) acc += w.res_u[static_cast<uint64_t>(o) * r + q] * vx[q];
            y[o] += acc;
        }
    }
    if (spec.has_bias) {
        for (uint32_t o = 0; o < out; ++o) y[o] += w.bias[o];
    }
}

}  // namespace rotor

#endif  // ROTOR_TOPOLOGY_HPP
