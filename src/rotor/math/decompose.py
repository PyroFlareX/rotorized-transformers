"""Exact Givens decomposition of matrices with orthonormal columns (the SVD->Givens baseline).

This reproduces the earlier experiment: truncate the SVD to rank ``r`` and re-express the
orthonormal factors ``U_r``/``V_r`` as Givens sequences. The rotation sequence is a fixed
function of ``(m, r)`` (column-wise elimination from the bottom), so no indices are stored;
only angles plus ``r`` signs. Parameter count for an ``m x r`` Stiefel factor is
``r*m - r*(r+1)/2`` -- any saving over ``m*r`` dense coefficients is the Stiefel-manifold
dimension, and the dominant saving comes from the rank truncation itself. This module exists
to *measure* that, not to claim it as a rotor gain.
"""

from __future__ import annotations

import torch


def stiefel_sequence(m: int, r: int) -> list[tuple[int, int]]:
    """Ordered pairs ``(i-1, i)`` used to zero entries below the diagonal of an ``m x r`` matrix."""
    seq = []
    for j in range(r):
        for i in range(m - 1, j, -1):
            seq.append((i - 1, i))
    return seq


def stiefel_param_count(m: int, r: int) -> int:
    return r * m - r * (r + 1) // 2


def givens_decompose(Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose ``Q`` (``m x r``, orthonormal columns) as ``Q = G^T [diag(signs); 0]``.

    Returns ``(angles, signs)``; ``angles[t]`` corresponds to ``stiefel_sequence(m, r)[t]``.
    Computed in float64.
    """
    m, r = Q.shape
    A = Q.double().clone()
    seq = stiefel_sequence(m, r)
    angles = torch.empty(len(seq), dtype=torch.float64)
    t = 0
    for j in range(r):
        for i in range(m - 1, j, -1):
            a, b = A[i - 1, j], A[i, j]
            theta = torch.atan2(-b, a)  # rotation zeroing A[i, j] into A[i-1, j]
            c, s = torch.cos(theta), torch.sin(theta)
            rows = A[[i - 1, i]].clone()
            A[i - 1] = c * rows[0] - s * rows[1]
            A[i] = s * rows[0] + c * rows[1]
            angles[t] = theta
            t += 1
    signs = torch.sign(torch.diagonal(A[:r, :r]))
    signs[signs == 0] = 1.0
    return angles, signs


def givens_reconstruct(angles: torch.Tensor, signs: torch.Tensor, m: int) -> torch.Tensor:
    """Inverse of ``givens_decompose``: rebuild the ``m x r`` matrix in float64."""
    r = signs.numel()
    A = torch.zeros(m, r, dtype=torch.float64)
    A[torch.arange(r), torch.arange(r)] = signs.double()
    seq = stiefel_sequence(m, r)
    for t in range(len(seq) - 1, -1, -1):  # apply G^T = reverse order, negated angles
        i0, i1 = seq[t]
        c, s = torch.cos(angles[t]), torch.sin(-angles[t])
        rows = A[[i0, i1]].clone()
        A[i0] = c * rows[0] - s * rows[1]
        A[i1] = s * rows[0] + c * rows[1]
    return A


def svd_givens_baseline(W: torch.Tensor, rank: int) -> dict:
    """Truncated SVD whose orthonormal factors are stored as Givens angles.

    Returns the reconstruction and an honest parameter breakdown.
    """
    out_f, in_f = W.shape
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    r = max(0, min(rank, S.numel()))
    Ur, Vr = U[:, :r], Vh[:r].T
    au, su = givens_decompose(Ur)
    av, sv = givens_decompose(Vr)
    W_hat = givens_reconstruct(au, su, out_f) @ torch.diag(S[:r]) @ givens_reconstruct(av, sv, in_f).T
    return {
        "W_hat": W_hat.to(W.dtype),
        "rank": r,
        "params": {
            "angles": au.numel() + av.numel(),
            "singular_values": r,
            "signs_bits": 2 * r,  # 1 bit each; reported, not counted as full coefficients
        },
        "param_count": au.numel() + av.numel() + r,
    }
