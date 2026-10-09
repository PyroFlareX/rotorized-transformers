"""Batched application of one stage of disjoint 2D Givens rotations.

Convention for a pair ``(a, b)`` with angle ``t`` acting on a row vector ``x``::

    x_a' = cos(t) * x_a - sin(t) * x_b
    x_b' = sin(t) * x_a + cos(t) * x_b

i.e. ``x' = G x`` with ``G[a,a]=G[b,b]=c, G[a,b]=-s, G[b,a]=s``. The transpose (inverse) is the
same stage with ``-t``. Coordinates not covered by any pair pass through unchanged.
"""

from __future__ import annotations

import torch

from rotor.math.topology import Topology


class GivensStage:
    """Index buffers for one stage; cached per (topology, n, stage, device)."""

    __slots__ = ("a", "b", "n", "num_pairs", "partner", "inv_perm", "n_pass")

    def __init__(self, pairs: tuple[tuple[int, int], ...], n: int, device: torch.device | None = None):
        self.n = n
        self.num_pairs = len(pairs)
        self.n_pass = n - 2 * len(pairs)
        if pairs:
            idx = torch.tensor(pairs, dtype=torch.long, device=device)
            self.a = idx[:, 0].contiguous()
            self.b = idx[:, 1].contiguous()
            covered = torch.zeros(n, dtype=torch.bool, device=device)
            covered[self.a] = True
            covered[self.b] = True
            rest = (~covered).nonzero().squeeze(1)
            # ``partner`` swaps each pair and fixes uncovered coordinates; ``inv_perm`` scatters the
            # [a..., b..., rest...]-ordered coefficient vectors back to coordinate order.
            self.partner = torch.arange(n, dtype=torch.long, device=device)
            self.partner[self.a] = self.b
            self.partner[self.b] = self.a
            self.inv_perm = torch.argsort(torch.cat([self.a, self.b, rest]))
        else:
            self.a = self.b = self.partner = self.inv_perm = torch.empty(0, dtype=torch.long, device=device)

    def apply_rows(self, h: torch.Tensor, theta: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Rotate axis 0 of ``h`` (shape ``(n, B)``): ``y = C * h + S * h[partner]`` with per-coordinate
        coefficients. Same arithmetic as :meth:`apply` (bitwise for finite inputs) but one whole-row gather and a
        fused multiply-add instead of ``clone`` + two ``index_copy_`` passes; whole contiguous rows move, which
        is what makes the gather cheap on CPU and coalesced on GPU.
        """
        if self.num_pairs == 0:
            return h
        c = torch.cos(theta).to(h.dtype)
        s = torch.sin(theta).to(h.dtype)
        if transpose:
            s = -s
        ones = torch.ones(self.n_pass, dtype=h.dtype, device=h.device)
        zeros = torch.zeros(self.n_pass, dtype=h.dtype, device=h.device)
        C = torch.cat([c, c, ones])[self.inv_perm]
        S = torch.cat([-s, s, zeros])[self.inv_perm]
        return C.unsqueeze(1) * h + S.unsqueeze(1) * h.index_select(0, self.partner)

    def apply(self, x: torch.Tensor, theta: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Rotate the last dim of ``x`` (shape ``(..., n)``). ``theta`` has shape ``(num_pairs,)``.

        Out-of-place and autograd-friendly (gradients flow to both ``x`` and ``theta``).
        """
        if self.num_pairs == 0:
            return x
        c = torch.cos(theta).to(x.dtype)
        s = torch.sin(theta).to(x.dtype)
        if transpose:
            s = -s
        xa = x.index_select(-1, self.a)
        xb = x.index_select(-1, self.b)
        ya = c * xa - s * xb
        yb = s * xa + c * xb
        out = x.clone()
        out.index_copy_(-1, self.a, ya)
        out.index_copy_(-1, self.b, yb)
        return out


def build_stages(topology: Topology, n: int, num_stages: int, device=None) -> list[GivensStage]:
    return [GivensStage(topology.pairs(n, s), n, device) for s in range(num_stages)]


def stage_matrix(pairs: tuple[tuple[int, int], ...], theta: torch.Tensor, n: int) -> torch.Tensor:
    """Dense ``n x n`` matrix of one stage (reference/testing only)."""
    G = torch.eye(n, dtype=theta.dtype, device=theta.device)
    for (a, b), t in zip(pairs, theta, strict=True):
        c, s = torch.cos(t), torch.sin(t)
        G[a, a] = c
        G[b, b] = c
        G[a, b] = -s
        G[b, a] = s
    return G
