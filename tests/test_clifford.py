"""Tests for blockwise Clifford left-geometric-product projections."""

from __future__ import annotations

import pytest
import torch

from rotor.math.clifford import CliffordLinear, multiplication_basis, project_clifford

F64 = torch.float64


def _independent_blade_product(left: int, right: int, dimensions: int) -> tuple[int, int]:
    """Return sign and result mask using the defining generator reordering rule."""
    swaps = sum(
        1
        for left_axis in range(dimensions)
        if left & (1 << left_axis)
        for right_axis in range(dimensions)
        if right & (1 << right_axis) and left_axis > right_axis
    )
    return (-1 if swaps % 2 else 1), left ^ right

def _independent_basis(dimensions: int) -> torch.Tensor:
    q = 1 << dimensions
    basis = torch.zeros(q, q, q, dtype=F64)
    for left in range(q):
        for right in range(q):
            sign, result = _independent_blade_product(left, right, dimensions)
            basis[left, result, right] = sign
    return basis


def _independent_projection(
    weight: torch.Tensor,
    dimensions: int,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
) -> torch.Tensor:
    q = 1 << dimensions
    out_features, in_features = weight.shape
    grouped = weight.index_select(0, output_permutation).index_select(1, input_permutation)
    grouped = grouped.reshape(out_features // q, q, in_features // q, q)
    return torch.einsum("oikj,bij->okb", grouped, _independent_basis(dimensions)) / q


def _direct_projected_action(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    dimensions: int,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
) -> torch.Tensor:
    """Small independent oracle, routing each product by bit-mask blades."""
    q = 1 << dimensions
    in_blocks = coefficients.shape[1]
    out_blocks = coefficients.shape[0]
    grouped_x = x.index_select(-1, input_permutation).reshape(*x.shape[:-1], in_blocks, q)
    grouped_y = torch.zeros(*x.shape[:-1], out_blocks, q, dtype=x.dtype, device=x.device)
    for out_block in range(out_blocks):
        for in_block in range(in_blocks):
            for left_blade in range(q):
                coefficient = coefficients[out_block, in_block, left_blade].to(x.dtype)
                for right_blade in range(q):
                    sign, output_blade = _independent_blade_product(
                        left_blade, right_blade, dimensions
                    )
                    grouped_y[..., out_block, output_blade] += (
                        sign * coefficient * grouped_x[..., in_block, right_blade]
                    )
    grouped_y = grouped_y.reshape(*x.shape[:-1], out_blocks * q)
    inverse_output = torch.empty_like(output_permutation)
    inverse_output[output_permutation] = torch.arange(
        output_permutation.numel(), device=output_permutation.device
    )
    return grouped_y.index_select(-1, inverse_output)


def _in_family_matrix(coefficients: torch.Tensor, dimensions: int) -> torch.Tensor:
    basis = multiplication_basis(dimensions).to(dtype=F64)
    q = 1 << dimensions
    grouped = torch.einsum("okb,bij->oikj", coefficients, basis)
    return grouped.reshape(coefficients.shape[0] * q, coefficients.shape[1] * q)


def test_blade_multiplication_matches_independent_generator_rules() -> None:
    table = multiplication_basis(3)
    assert table.dtype == torch.int8
    assert table.shape == (8, 8, 8)

    # e1*e2=e12, e2*e1=-e12, and each Euclidean generator squares to +1.
    assert table[0b001, 0b011, 0b010].item() == 1
    assert table[0b010, 0b011, 0b001].item() == -1
    for axis in range(3):
        vector = 1 << axis
        assert table[vector, 0, vector].item() == 1
        assert table[vector, 0, 0].item() == 0
        for other_axis in range(axis + 1, 3):
            other = 1 << other_axis
            assert table[vector, vector ^ other, other].item() == 1
            assert table[other, vector ^ other, vector].item() == -1

    for left in range(8):
        for right in range(8):
            sign, result = _independent_blade_product(left, right, 3)
            assert table[left, result, right].item() == sign
            assert torch.count_nonzero(table[left, :, right]).item() == 1


def test_projection_is_orthogonal_and_has_exact_scalar_count() -> None:
    dimensions = 2
    q = 1 << dimensions
    weight = torch.randn(12, 8, dtype=F64, generator=torch.Generator().manual_seed(30))

    module = project_clifford(weight, dimensions)
    assert isinstance(module, CliffordLinear)
    assert module.coefficients.shape == (3, 2, q)
    assert module.coefficients.numel() == weight.numel() // q
    assert module.input_permutation.tolist() == list(range(weight.shape[1]))
    assert module.output_permutation.tolist() == list(range(weight.shape[0]))

    basis = multiplication_basis(dimensions).to(dtype=F64)
    residual = weight - module.reconstruct()
    for out_block in range(weight.shape[0] // q):
        for in_block in range(weight.shape[1] // q):
            block = residual[
                out_block * q : (out_block + 1) * q,
                in_block * q : (in_block + 1) * q,
            ]
            correlations = torch.einsum("ij,bij->b", block, basis)
            torch.testing.assert_close(
                correlations, torch.zeros_like(correlations), rtol=0, atol=1e-12
            )


def test_projection_recovers_known_in_family_rectangular_matrix() -> None:
    dimensions = 2
    q = 1 << dimensions
    coefficients = torch.randn(
        3, 2, q, dtype=F64, generator=torch.Generator().manual_seed(31)
    )
    weight = _in_family_matrix(coefficients, dimensions)

    module = project_clifford(weight, dimensions)
    torch.testing.assert_close(module.coefficients, coefficients, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(module.reconstruct(), weight, rtol=1e-14, atol=1e-14)

    x = torch.randn(2, 3, weight.shape[1], dtype=F64)
    expected = _direct_projected_action(
        x,
        coefficients,
        dimensions,
        module.input_permutation,
        module.output_permutation,
    )
    torch.testing.assert_close(module(x), expected, rtol=1e-13, atol=1e-13)


def test_seeded_permuted_rectangular_forward_uses_direct_clifford_action() -> None:
    dimensions = 2
    weight = torch.randn(12, 8, dtype=F64, generator=torch.Generator().manual_seed(32))
    module = project_clifford(weight, dimensions, seed=42)
    assert module.coefficients.shape == (3, 2, 4)

    x = torch.randn(2, 3, 8, dtype=F64, generator=torch.Generator().manual_seed(33))
    expected = _direct_projected_action(
        x,
        module.coefficients,
        dimensions,
        module.input_permutation,
        module.output_permutation,
    )
    actual = module(x)
    expected_coefficients = _independent_projection(
        weight, dimensions, module.input_permutation, module.output_permutation
    )
    torch.testing.assert_close(module.coefficients, expected_coefficients, rtol=1e-14, atol=1e-14)
    grouped_weight = weight.index_select(0, module.output_permutation).index_select(
        1, module.input_permutation
    )
    residual = (grouped_weight - _in_family_matrix(module.coefficients, dimensions)).reshape(
        3, 4, 2, 4
    )
    basis = _independent_basis(dimensions)
    for out_block in range(residual.shape[0]):
        for in_block in range(residual.shape[2]):
            correlations = torch.einsum("ij,bij->b", residual[out_block, :, in_block, :], basis)
            torch.testing.assert_close(
                correlations, torch.zeros_like(correlations), rtol=0, atol=1e-12
            )
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual, x @ module.reconstruct().T, rtol=1e-13, atol=1e-13)

    repeated = project_clifford(weight, dimensions, seed=42)
    assert torch.equal(module.input_permutation, repeated.input_permutation)
    assert torch.equal(module.output_permutation, repeated.output_permutation)
    assert torch.equal(module.coefficients, repeated.coefficients)


@pytest.mark.parametrize(
    "dimensions", [0, -1, 7, True, 2.5, "2"]
)
def test_invalid_dimensions_are_rejected(dimensions: object) -> None:
    with pytest.raises(ValueError):
        multiplication_basis(dimensions)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "weight,dimensions,exception",
    [
        (torch.ones(8), 2, ValueError),
        (torch.ones(6, 8), 2, ValueError),
        (torch.ones(8, 6), 2, ValueError),
        (torch.empty(0, 8), 2, ValueError),
        (torch.ones(8, 8, dtype=torch.int64), 2, TypeError),
        (torch.full((4, 4), float("nan")), 2, ValueError),
    ],
)
def test_invalid_weights_are_rejected(
    weight: torch.Tensor, dimensions: int, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        project_clifford(weight, dimensions)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError):
        project_clifford(torch.eye(4, dtype=F64), 2, seed=seed)  # type: ignore[arg-type]
