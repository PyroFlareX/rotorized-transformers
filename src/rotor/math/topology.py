"""Deterministic rotation schedules.

A *stage* is a set of disjoint coordinate pairs ``(a, b)`` acted on by independent 2D Givens
rotations. Every topology here is a pure function of ``(kind, n, stage_index, seed)`` so a
checkpoint stores only those integers, never coordinate indices. The pair ordering within a
stage is canonical (sorted by ``a``) and defines the order of the flat angle vector.

The same algorithms are mirrored in ``runtime/interfaces/rotor_topology.hpp``; the PRNG is a
hand-written SplitMix64 so C++ reproduces it bit-exactly (no numpy/torch RNG dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

TOPOLOGY_KINDS = ("brick", "butterfly", "seeded")
_MASK64 = (1 << 64) - 1


def splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step. Returns ``(new_state, output)``."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


def seeded_permutation(n: int, seed: int, stage: int) -> list[int]:
    """Fisher-Yates permutation of ``range(n)`` driven by SplitMix64.

    The stream state is ``seed * 0x100000001B3 + stage`` (mod 2**64). Bounded draws use
    ``output % (i + 1)``; the modulo bias is irrelevant for topology purposes and keeps the
    C++ mirror trivial.
    """
    state = (seed * 0x100000001B3 + stage) & _MASK64
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        state, out = splitmix64(state)
        j = out % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _num_strides(n: int) -> int:
    return max(1, (n - 1).bit_length())


@lru_cache(maxsize=4096)
def stage_pairs(kind: str, n: int, stage: int, seed: int = 0) -> tuple[tuple[int, int], ...]:
    """Disjoint pairs for rotation stage ``stage`` (0-based) of an ``n``-dim cascade.

    - ``brick``: odd/even nearest-neighbour pattern; ``n`` stages are universal for SO(n).
    - ``butterfly``: stride ``2**(stage mod ceil(log2 n))``; pairs ``(i, i+d)`` with
      ``(i // d)`` even. Mixes globally in ``O(log n)`` stages; works for any ``n``.
    - ``seeded``: a SplitMix64 permutation per stage, consecutive elements paired.
    """
    if n < 2:
        return ()
    if kind == "brick":
        off = stage % 2
        pairs = [(i, i + 1) for i in range(off, n - 1, 2)]
    elif kind == "butterfly":
        d = 1 << (stage % _num_strides(n))
        pairs = [(i, i + d) for i in range(n - d) if (i // d) % 2 == 0]
    elif kind == "seeded":
        p = seeded_permutation(n, seed, stage)
        pairs = sorted((min(p[2 * j], p[2 * j + 1]), max(p[2 * j], p[2 * j + 1])) for j in range(n // 2))
    else:
        raise ValueError(f"unknown topology kind {kind!r}; expected one of {TOPOLOGY_KINDS}")
    return tuple(pairs)


@dataclass(frozen=True)
class Topology:
    """Deterministic schedule identity. Serialized verbatim into manifests."""

    kind: str = "butterfly"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in TOPOLOGY_KINDS:
            raise ValueError(f"unknown topology kind {self.kind!r}; expected one of {TOPOLOGY_KINDS}")

    def pairs(self, n: int, stage: int) -> tuple[tuple[int, int], ...]:
        return stage_pairs(self.kind, n, stage, self.seed)

    def angle_count(self, n: int, num_stages: int) -> int:
        return sum(len(self.pairs(n, s)) for s in range(num_stages))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: dict) -> Topology:
        return cls(kind=d["kind"], seed=int(d.get("seed", 0)))
