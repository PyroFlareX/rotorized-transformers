"""Classic butterfly factorization (Dao et al. 2019): products of sparse butterfly factors with FREE 2x2 blocks.

A butterfly *factor* at stride ``d`` of an ``n``-dim vector (``n`` a power of two) acts on the coordinate pairs
``(i, i + d)`` with ``(i // d)`` even by an independent 2x2 matrix ``[[a, b], [c, e]]``: ``n / 2`` blocks and
``4 * n / 2 = 2 n`` free parameters. The blocks are NOT constrained to rotations (contrast ``rotor.math.givens``,
where each pair carries one angle). One butterfly *matrix* is the product of ``log2(n)`` factors with strides
``1, 2, 4, ..., n / 2``; a product of ``m`` butterfly matrices is ``m * log2(n)`` factors, strides repeating.

Factor ``f`` uses stride ``2 ** (f mod log2(n))`` and factors are applied in index order (factor 0 first), so
``forward(x) = x @ B.T`` with ``B = F_{last} ... F_1 F_0``.

Parameter count (exact)::  ``num_factors * 2 * n``.   FLOP count (estimate)::  ``num_factors * (n / 2) * 6``
(4 multiplies + 2 adds per 2x2 block).

Rectangular matrices use the pad/truncate convention: input coordinates occupy ``[0, in)`` of the ``n``-dim
vector, the remaining ones are zero, and the output keeps ``[0, out)`` (see ``dense_weight``).
"""

from __future__ import annotations

import torch
from torch import nn


def next_pow2(n: int) -> int:
    """Smallest power of two ``>= max(n, 2)``."""
    return 1 << max(1, (int(n) - 1).bit_length())


def butterfly_param_count(n_pow2: int, num_factors: int) -> int:
    """Exact free-parameter count of ``num_factors`` butterfly factors of width ``n_pow2`` (``2 n`` each)."""
    return int(num_factors) * 2 * int(n_pow2)


class ButterflyMatrix(nn.Module):
    """``num_factors`` butterfly factors of width ``n_pow2`` with learnable 2x2 blocks (identity at construction).

    ``blocks`` has shape ``(num_factors, n / 2, 2, 2)``; block ``k`` of a factor with stride ``d`` acts on the pair
    ``(g * 2d + j, g * 2d + j + d)`` with ``g, j = divmod(k, d)``.
    """

    def __init__(self, n_pow2: int, num_factors: int, dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        n = int(n_pow2)
        if n < 2 or n & (n - 1):
            raise ValueError(f"n_pow2 must be a power of two >= 2, got {n_pow2}")
        if int(num_factors) < 1:
            raise ValueError(f"num_factors must be >= 1, got {num_factors}")
        self.n = n
        self.num_factors = int(num_factors)
        self.log_n = n.bit_length() - 1
        self.blocks = nn.Parameter(torch.empty(self.num_factors, n // 2, 2, 2, dtype=dtype, device=device))
        self.reset_identity()

    # --- structure ------------------------------------------------------------------------------------------
    @property
    def param_count(self) -> int:
        return butterfly_param_count(self.n, self.num_factors)

    def estimated_flops_per_token(self) -> int:
        """Formula estimate (not a measurement): 6 flops per 2x2 block, ``n / 2`` blocks per factor."""
        return self.num_factors * (self.n // 2) * 6

    def stride(self, factor: int) -> int:
        return 1 << (factor % self.log_n)

    @torch.no_grad()
    def reset_identity(self) -> None:
        """Every block is the 2x2 identity, so the whole product is the identity matrix."""
        self.blocks.copy_(torch.eye(2, dtype=self.blocks.dtype).expand_as(self.blocks))

    @torch.no_grad()
    def init_from_dense(
        self,
        W: torch.Tensor,
        noise_std: float = 0.01,
        generator: torch.Generator | None = None,
    ) -> None:
        """Deterministic warm start for fitting ``W`` (``out x in``), mirroring the rotor cascade's init.

        Blocks are the 2x2 identity plus ``N(0, noise_std)`` noise; factor 0 is then scaled so the truncated product
        ``B[:out, :in]`` has the Frobenius norm of ``W``.
        """
        out_f, in_f = W.shape
        if max(out_f, in_f) > self.n:
            raise ValueError(f"W {tuple(W.shape)} does not fit width {self.n}")
        self.reset_identity()
        if noise_std > 0:
            noise = torch.randn(self.blocks.shape, generator=generator, dtype=torch.float64) * noise_std
            self.blocks.add_(noise.to(self.blocks.dtype))
        norm = self.dense_weight(out_f, in_f).norm()
        target = W.detach().to(self.blocks.dtype).norm()
        if norm > 0 and target > 0:
            self.blocks[0].mul_(target / norm)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Single parameter kind, exposed as ``"scales"`` so ``rotor.training.layerwise.FitSchedule`` can drive it.

        A butterfly has no separate angles/scales/residual: ``blocks`` is trained in every schedule phase that trains
        any kind, with the ``scales`` learning rate.
        """
        return {"scales": [self.blocks]}

    # --- application ----------------------------------------------------------------------------------------
    def _apply_factor(self, x: torch.Tensor, f: int) -> torch.Tensor:
        d = self.stride(f)
        g = self.n // (2 * d)
        xv = x.reshape(*x.shape[:-1], g, 2, d)
        B = self.blocks[f].to(x.dtype).reshape(g, d, 2, 2)
        x0, x1 = xv[..., 0, :], xv[..., 1, :]
        y0 = B[..., 0, 0] * x0 + B[..., 0, 1] * x1
        y1 = B[..., 1, 0] * x0 + B[..., 1, 1] * x1
        return torch.stack((y0, y1), dim=-2).reshape(x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ B.T`` on the last dimension (which must equal ``n``)."""
        if x.shape[-1] != self.n:
            raise ValueError(f"last dim must be {self.n}, got {x.shape[-1]}")
        for f in range(self.num_factors):
            x = self._apply_factor(x, f)
        return x

    def dense_weight(self, out_features: int | None = None, in_features: int | None = None) -> torch.Tensor:
        """Materialize ``B`` (``n x n``), or its pad/truncate window ``B[:out, :in]``. Differentiable."""
        o = self.n if out_features is None else int(out_features)
        i = self.n if in_features is None else int(in_features)
        if not (0 < o <= self.n and 0 < i <= self.n):
            raise ValueError(f"window ({o}, {i}) must lie within width {self.n}")
        eye = torch.eye(self.n, dtype=self.blocks.dtype, device=self.blocks.device)[:i]
        return self.forward(eye)[:, :o].T

    def extra_repr(self) -> str:
        return f"n={self.n}, num_factors={self.num_factors}, params={self.param_count}"
