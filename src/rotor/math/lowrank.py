"""Low-rank pieces: the optional residual for cascades, and the truncated-SVD baseline."""

from __future__ import annotations

import math

import torch
from torch import nn


class LowRankResidual(nn.Module):
    """``y = x @ (U V).T`` with ``U: (out, r)``, ``V: (r, in)``. Zero-initialized output (``U = 0``)."""

    def __init__(self, in_features: int, out_features: int, rank: int, dtype=torch.float32, device=None):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.V = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        self.U.zero_()
        self.V.normal_(0.0, 1.0 / math.sqrt(max(self.V.shape[1], 1)), generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V.T) @ self.U.T


def truncated_svd(W: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-``rank`` factors ``(A, B)`` with ``W ~= A @ B`` (Eckart-Young), computed in float64.

    ``A = U_r diag(sqrt(s))``, ``B = diag(sqrt(s)) V_r^T`` (balanced split).
    """
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    r = max(0, min(rank, S.numel()))
    sq = S[:r].sqrt()
    return (U[:, :r] * sq).to(W.dtype), (sq[:, None] * Vh[:r]).to(W.dtype)


def lowrank_param_count(in_features: int, out_features: int, rank: int) -> int:
    return rank * (in_features + out_features)


def rank_for_budget(in_features: int, out_features: int, param_budget: int) -> int:
    return max(0, param_budget // (in_features + out_features))


class LowRankLinear(nn.Module):
    """Truncated-SVD baseline layer ``y = (x @ B.T) @ A.T (+ bias)``."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.A = nn.Parameter(A.detach().clone())
        self.B = nn.Parameter(B.detach().clone())
        self.in_features = B.shape[1]
        self.out_features = A.shape[0]
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    @classmethod
    def from_dense(cls, W: torch.Tensor, rank: int, bias: torch.Tensor | None = None) -> LowRankLinear:
        A, B = truncated_svd(W, rank)
        return cls(A, B, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x @ self.B.T.to(x.dtype)) @ self.A.T.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.A.shape[1]}"
