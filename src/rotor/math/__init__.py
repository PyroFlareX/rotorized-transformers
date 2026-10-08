"""Architecture-independent rotor mathematics. Must not import ``rotor.models`` or ``transformers``."""

from rotor.math.cascade import CascadeSpec, RectAdapter, RotorCascade, RotorLinear
from rotor.math.decompose import givens_decompose, givens_reconstruct, svd_givens_baseline
from rotor.math.givens import GivensStage, build_stages, stage_matrix
from rotor.math.lowrank import LowRankLinear, LowRankResidual, rank_for_budget, truncated_svd
from rotor.math.topology import TOPOLOGY_KINDS, Topology, seeded_permutation, splitmix64, stage_pairs

__all__ = [
    "CascadeSpec",
    "RectAdapter",
    "RotorCascade",
    "RotorLinear",
    "givens_decompose",
    "givens_reconstruct",
    "svd_givens_baseline",
    "GivensStage",
    "build_stages",
    "stage_matrix",
    "LowRankLinear",
    "LowRankResidual",
    "rank_for_budget",
    "truncated_svd",
    "TOPOLOGY_KINDS",
    "Topology",
    "seeded_permutation",
    "splitmix64",
    "stage_pairs",
]
