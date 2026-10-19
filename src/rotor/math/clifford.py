"""Blockwise Clifford left-geometric-product projections for real linear maps.

A block of ``q = 2**dimensions`` features is identified with a Euclidean
Clifford multivector whose bit-mask blade order is ascending.  A projected
matrix block is represented by the ``q`` left-multiplication operators, so
its coefficient count is exactly one ``q``-th of its dense scalar count.
"""

from __future__ import annotations

from operator import index as _index

import torch
from torch import nn

_MAX_DIMENSIONS = 6


def _dimension(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"dimensions must be an integer in [1, {_MAX_DIMENSIONS}]")
    try:
        result = _index(value)
    except TypeError as exc:
        raise ValueError(f"dimensions must be an integer in [1, {_MAX_DIMENSIONS}]") from exc
    if not 1 <= result <= _MAX_DIMENSIONS:
        raise ValueError(f"dimensions must be in [1, {_MAX_DIMENSIONS}]")
    return result


def multiplication_basis(d: int) -> torch.Tensor:
    """Return ``table[b, i, j]`` for the coefficient of blade ``i`` in ``e_b*e_j``.

    Blades use ascending bit masks and the Euclidean convention ``e_i**2 = +1``.
    Each left-multiplication matrix is a signed permutation matrix.  The
    returned CPU tensor has dtype ``int8`` and shape ``(q, q, q)``, where
    ``q = 2**d``.
    """
    dimensions = _dimension(d)
    q = 1 << dimensions
    table = torch.zeros((q, q, q), dtype=torch.int8)
    for left in range(q):
        for right in range(q):
            swaps = sum(
                (right & ((1 << axis) - 1)).bit_count()
                for axis in range(dimensions)
                if left & (1 << axis)
            )
            result = left ^ right
            table[left, result, right] = -1 if swaps & 1 else 1
    return table


class CliffordLinear(nn.Module):
    """Frozen blockwise Clifford projection of a dense linear map.

    ``coefficients`` has shape ``(out_features // q, in_features // q, q)``.
    The public permutations list the original feature indices in grouped order.
    Forward applies signed blade routing and coefficient-sized block matmuls;
    it never forms a dense projected weight.  ``reconstruct()`` is provided
    only as an explicit diagnostic operation.
    """

    def __init__(
        self,
        coefficients: torch.Tensor,
        dimensions: int,
        input_permutation: torch.Tensor,
        output_permutation: torch.Tensor,
    ) -> None:
        super().__init__()
        dimensions = _dimension(dimensions)
        q = 1 << dimensions
        if not isinstance(coefficients, torch.Tensor) or coefficients.ndim != 3:
            raise ValueError("coefficients must be a rank-three tensor")
        if coefficients.dtype != torch.float64:
            raise TypeError("Clifford coefficients must use float64")
        if coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.shape[2] != q:
            raise ValueError(f"coefficients must have shape (out_blocks, in_blocks, {q})")
        if not torch.isfinite(coefficients).all().item():
            raise ValueError("coefficients must contain only finite values")

        out_features = coefficients.shape[0] * q
        in_features = coefficients.shape[1] * q
        for name, permutation, size in (
            ("input_permutation", input_permutation, in_features),
            ("output_permutation", output_permutation, out_features),
        ):
            if not isinstance(permutation, torch.Tensor) or permutation.ndim != 1:
                raise ValueError(f"{name} must be a rank-one tensor")
            if permutation.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(f"{name} must use an integer dtype")
            if permutation.numel() != size:
                raise ValueError(f"{name} must contain {size} indices")
            permutation = permutation.to(device=coefficients.device, dtype=torch.long)
            if not torch.equal(
                torch.sort(permutation).values,
                torch.arange(size, device=coefficients.device),
            ):
                raise ValueError(f"{name} must be a permutation of [0, {size})")
            if name == "input_permutation":
                input_permutation = permutation
            else:
                output_permutation = permutation

        self.dimensions = dimensions
        self.q = q
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("coefficients", coefficients.detach().clone().contiguous())
        self.register_buffer("input_permutation", input_permutation.detach().clone().contiguous())
        self.register_buffer("output_permutation", output_permutation.detach().clone().contiguous())

        # These small integer routing tables are structural metadata, not model
        # coefficients, and are omitted from state_dicts.
        basis = multiplication_basis(dimensions).to(device=coefficients.device)
        blades = torch.arange(q, device=coefficients.device)
        routes = torch.bitwise_xor(blades[:, None], blades[None, :])
        signs = basis[blades[:, None], blades[None, :], routes]
        inverse_output = torch.empty_like(self.output_permutation)
        inverse_output[self.output_permutation] = torch.arange(
            out_features, device=self.output_permutation.device
        )
        self.register_buffer("_blade_routes", routes, persistent=False)
        self.register_buffer("_blade_signs", signs, persistent=False)
        self.register_buffer("_inverse_output_permutation", inverse_output, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projected map while preserving all leading input dimensions."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("input must be a floating-point tensor")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"input must have final dimension {self.in_features}")

        leading_shape = x.shape[:-1]
        grouped = x.index_select(-1, self.input_permutation).reshape(
            *leading_shape, self.in_features // self.q, self.q
        )
        coefficients = self.coefficients.to(dtype=x.dtype)
        result = None
        for blade in range(self.q):
            routed = grouped.index_select(-1, self._blade_routes[blade])
            routed = routed * self._blade_signs[blade].to(dtype=x.dtype)
            contribution = torch.matmul(
                routed.transpose(-2, -1), coefficients[:, :, blade].transpose(0, 1)
            ).transpose(-2, -1)
            result = contribution if result is None else result + contribution

        grouped_output = result.reshape(*leading_shape, self.out_features)
        return grouped_output.index_select(-1, self._inverse_output_permutation)

    def reconstruct(self) -> torch.Tensor:
        """Explicitly materialize the projected dense matrix in original ordering."""
        basis = multiplication_basis(self.dimensions).to(
            device=self.coefficients.device, dtype=self.coefficients.dtype
        )
        grouped = torch.einsum("okb,bij->oikj", self.coefficients, basis).reshape(
            self.out_features, self.in_features
        )
        inverse_input = torch.empty_like(self.input_permutation)
        inverse_input[self.input_permutation] = torch.arange(
            self.in_features, device=self.input_permutation.device
        )
        return grouped.index_select(0, self._inverse_output_permutation).index_select(
            1, inverse_input
        )


def project_clifford(
    weight: torch.Tensor,
    d: int,
    seed: int | None = None,
) -> CliffordLinear:
    """Project a real matrix onto blockwise left-geometric-product operators.

    Each grouped ``q x q`` block is projected with
    ``c_b = <block, L_b> / q``.  ``seed=None`` uses contiguous identity feature
    grouping.  An integer seed deterministically shuffles input and output
    feature indices (input first, output second) before grouping.  Returned
    coefficients are always float64, independent of the input floating dtype.
    """
    dimensions = _dimension(d)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-two matrix")
    if not weight.is_floating_point():
        raise TypeError("weight must be real floating point")
    if weight.layout != torch.strided:
        raise TypeError("weight must use strided dense storage")
    if not torch.isfinite(weight).all().item():
        raise ValueError("weight must contain only finite values")

    q = 1 << dimensions
    out_features, in_features = weight.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("weight dimensions must be positive")
    if out_features % q or in_features % q:
        raise ValueError(f"weight dimensions must both be divisible by q={q}")

    if seed is None:
        input_permutation = torch.arange(in_features, device=weight.device)
        output_permutation = torch.arange(out_features, device=weight.device)
    else:
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer or None")
        try:
            seed = _index(seed)
        except TypeError as exc:
            raise ValueError("seed must be an integer or None") from exc
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        except (RuntimeError, OverflowError) as exc:
            raise ValueError("seed is outside the supported integer range") from exc
        input_permutation = torch.randperm(in_features, generator=generator).to(weight.device)
        output_permutation = torch.randperm(out_features, generator=generator).to(weight.device)

    grouped = weight.detach().to(dtype=torch.float64).index_select(
        0, output_permutation
    ).index_select(1, input_permutation).reshape(
        out_features // q, q, in_features // q, q
    )
    basis = multiplication_basis(dimensions).to(device=weight.device, dtype=torch.float64)
    coefficients = torch.einsum("oikj,bij->okb", grouped, basis) / q
    return CliffordLinear(coefficients, dimensions, input_permutation, output_permutation)
