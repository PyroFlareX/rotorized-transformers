"""Behavioral tests for algebraic GA-rotor SVD decomposition and execution."""

from __future__ import annotations

import math

import pytest
import torch

from rotor.math.ga import GARotorLinear, decompose_ga, rotor_count, wave_schedule

F64 = torch.float64


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _clifford_product(left: list[float], right: list[float], dimensions: int) -> list[float]:
    """Independent Euclidean geometric product on bit-mask basis blades."""
    result = [0.0] * (1 << dimensions)
    for left_mask, left_value in enumerate(left):
        if left_value == 0:
            continue
        for right_mask, right_value in enumerate(right):
            if right_value == 0:
                continue
            swaps = sum(
                1
                for i in range(dimensions)
                if left_mask & (1 << i)
                for j in range(dimensions)
                if right_mask & (1 << j) and i > j
            )
            sign = -1.0 if swaps % 2 else 1.0
            result[left_mask ^ right_mask] += sign * left_value * right_value
    return result


def _clifford_sandwich(
    vector: list[float], a: float, b: float, i: int, j: int, dimensions: int
) -> list[float]:
    """Compute ``R x reverse(R) / |R|^2`` using full multivector products."""
    rotor = [0.0] * (1 << dimensions)
    reverse = [0.0] * (1 << dimensions)
    blade = (1 << i) | (1 << j)
    rotor[0], rotor[blade] = a, b
    reverse[0], reverse[blade] = a, -b
    x = [0.0] * (1 << dimensions)
    for axis, value in enumerate(vector):
        x[1 << axis] = value
    product = _clifford_product(_clifford_product(rotor, x, dimensions), reverse, dimensions)
    norm_squared = a * a + b * b
    return [product[1 << axis] / norm_squared for axis in range(dimensions)]


@pytest.mark.parametrize("dimensions,lower", [(2, 1), (3, 1), (4, 3)])
@pytest.mark.parametrize("a,b", [(1.0, 0.0), (0.7, -1.2), (-0.3, 2.5)])
def test_angle_rotor_action_matches_independent_clifford_sandwich(
    dimensions: int, lower: int, a: float, b: float
) -> None:
    angle_count = rotor_count(dimensions, dimensions)
    right_angles = torch.zeros(angle_count, dtype=F64)
    packed_offset = next(
        offset
        for wave in wave_schedule(dimensions, dimensions)
        for _, row, offset in wave.cells(dimensions)
        if row == lower
    )
    # The oracle continues to exercise the Clifford coefficient pair, while
    # the runtime stores only its equivalent elimination angle.
    right_angles[packed_offset] = 2 * math.atan2(b, a)
    module = GARotorLinear(
        in_features=dimensions,
        out_features=dimensions,
        rank=dimensions,
        left_angles=torch.zeros(angle_count, dtype=F64),
        right_angles=right_angles,
        left_signs=torch.ones(dimensions, dtype=F64),
        right_signs=torch.ones(dimensions, dtype=F64),
        singular_values=torch.ones(dimensions, dtype=F64),
    )
    assert module.left_angles.shape == (angle_count,)
    assert module.right_angles.shape == (angle_count,)
    x = torch.tensor([[0.4, -1.1, 2.3, -0.8][:dimensions]], dtype=F64)
    expected = torch.tensor(
        [_clifford_sandwich(x[0].tolist(), a, b, lower - 1, lower, dimensions)], dtype=F64
    )
    torch.testing.assert_close(module(x), expected, rtol=1e-12, atol=1e-12)


def test_wave_schedule_is_canonical_disjoint_and_dependency_ordered() -> None:
    for rows in range(1, 11):
        for rank in range(rows + 1):
            waves = wave_schedule(rows, rank)
            flattened = []
            offset = 0
            for stage, wave in enumerate(waves):
                assert wave.stage == stage
                cells = tuple(wave.cells(rows))
                assert wave.offset == offset
                assert [j for j, _, _ in cells] == sorted(j for j, _, _ in cells)
                coordinates = [axis for _, i, _ in cells for axis in (i - 1, i)]
                assert len(coordinates) == len(set(coordinates))
                for j, i, packed in cells:
                    assert packed == wave.offset + j - wave.first_column
                    assert stage == rows - 1 - i + 2 * j
                    flattened.append((j, i, packed, stage))
                offset += wave.pair_count
            assert offset == rotor_count(rows, rank)
            assert len(flattened) == rotor_count(rows, rank)
            assert [cell[2] for cell in flattened] == list(range(offset))
            for j in range(rank):
                column = sorted((stage, i) for col, i, _, stage in flattened if col == j)
                assert [i for _, i in column] == list(range(rows - 1, j, -1))
                assert all(column[k + 1][0] == column[k][0] + 1 for k in range(len(column) - 1))
            cells = {(j, i): stage for j, i, _, stage in flattened}
            for (j, i), stage in cells.items():
                if (j + 1, i) in cells:
                    assert cells[j + 1, i] == stage + 2


@pytest.mark.parametrize("shape,rank", [
    ((2, 2), None), ((7, 3), None), ((3, 7), None), ((5, 5), None),
    ((7, 4), 2), ((4, 7), 3), ((6, 4), 0), ((6, 4), 4),
])
def test_decomposition_direct_forward_and_chunked_reconstruction(
    shape: tuple[int, int], rank: int | None
) -> None:
    out_features, in_features = shape
    generator = _generator(101 + out_features * 13 + in_features)
    weight = torch.randn(shape, dtype=F64, generator=generator)
    bias = torch.randn(out_features, dtype=F64, generator=generator)
    module = decompose_ga(weight, bias=bias, rank=rank)
    selected_rank = min(shape) if rank is None else rank
    u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    oracle = (u[:, :selected_rank] * singular_values[:selected_rank]) @ vh[:selected_rank]
    x = torch.randn(2, 3, in_features, dtype=F64, generator=generator)

    assert not list(module.parameters())
    assert module.left_angles.shape == (rotor_count(out_features, selected_rank),)
    assert module.right_angles.shape == (rotor_count(in_features, selected_rank),)
    assert module.left_angles.dtype == module.right_angles.dtype == F64
    assert not any(name.endswith("rotors") for name, _ in module.named_buffers())
    torch.testing.assert_close(module.reconstruct(chunk_size=2), oracle, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(module(x), x @ oracle.T + bias, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(module.reconstruct(chunk_size=1), module.reconstruct(chunk_size=17), rtol=0, atol=0)


def test_reconstruction_is_independent_of_bias_and_preserves_input_dtype() -> None:
    weight = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]], dtype=F64)
    bias = torch.tensor([10.0, -5.0, 1.5], dtype=F64)
    module = decompose_ga(weight, bias=bias)
    torch.testing.assert_close(module.reconstruct(), weight, rtol=1e-12, atol=1e-12)
    x = torch.randn(2, 5, 2, dtype=torch.float32)
    y = module(x)
    assert y.dtype == x.dtype
    torch.testing.assert_close(y, x @ weight.float().T + bias.float(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("shape,rank", [((5, 4), None), ((4, 6), 3), ((5, 5), None), ((5, 4), 3)])
def test_zero_repeated_and_rank_deficient_matrices(shape: tuple[int, int], rank: int | None) -> None:
    rows, columns = shape
    if (rows, columns, rank) == (5, 4, None):
        weight = torch.zeros(shape, dtype=F64)
    elif rows == columns == 5:
        weight = torch.diag(torch.tensor([3.0, 3.0, 1.0, 0.0, 0.0], dtype=F64))
    else:
        g = _generator(rows * 10 + columns)
        left = torch.randn(rows, 2, dtype=F64, generator=g)
        right = torch.randn(2, columns, dtype=F64, generator=g)
        weight = left @ right
    module = decompose_ga(weight, rank=rank)
    assert torch.isfinite(module.left_angles).all()
    assert torch.isfinite(module.right_angles).all()
    assert torch.isfinite(module.singular_values).all()
    u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    used_rank = min(rows, columns) if rank is None else rank
    expected = (u[:, :used_rank] * singular_values[:used_rank]) @ vh[:used_rank]
    torch.testing.assert_close(module.reconstruct(), expected, rtol=3e-11, atol=3e-11)


def test_elimination_stores_atan2_of_lower_and_upper() -> None:
    from rotor.math.ga import _eliminate_orthonormal_columns

    Q = torch.tensor([[0.6], [-0.8]], dtype=F64)
    angles, signs = _eliminate_orthonormal_columns(Q)
    torch.testing.assert_close(angles, torch.tensor([math.atan2(-0.8, 0.6)], dtype=F64))
    torch.testing.assert_close(signs, torch.ones(1, dtype=F64))


def test_zero_elimination_angles_are_identity_and_zero_rank_is_supported() -> None:
    zero = decompose_ga(torch.zeros((4, 3), dtype=F64))
    torch.testing.assert_close(zero.left_angles, torch.zeros_like(zero.left_angles))
    torch.testing.assert_close(zero.right_angles, torch.zeros_like(zero.right_angles))
    assert torch.count_nonzero(zero.reconstruct()) == 0
    rank_zero = decompose_ga(torch.randn((4, 3), dtype=F64), rank=0)
    assert rank_zero.reconstruct().shape == (4, 3)
    assert rank_zero.left_angles.shape == (0,)
    assert rank_zero.right_angles.shape == (0,)
    assert torch.count_nonzero(rank_zero.reconstruct()) == 0


@pytest.mark.parametrize("angle", [-math.pi + 0.2, -1.1, 0.0, 0.8, math.pi - 0.2])
def test_stored_elimination_angle_has_direct_trig_orientation(angle: float) -> None:
    # For a single two-row elimination, atan2(lower, upper) is the stored
    # angle and the direct action uses cos(angle), -sin(angle).
    module = GARotorLinear(
        in_features=2,
        out_features=2,
        rank=1,
        left_angles=torch.zeros(1, dtype=F64),
        right_angles=torch.tensor([angle], dtype=F64),
        left_signs=torch.ones(1, dtype=F64),
        right_signs=torch.ones(1, dtype=F64),
        singular_values=torch.ones(1, dtype=F64),
    )
    x = torch.tensor([[math.cos(angle), math.sin(angle)]], dtype=F64)
    torch.testing.assert_close(
        module._weight_forward(x), torch.tensor([[1.0, 0.0]], dtype=F64), rtol=1e-12, atol=1e-12
    )


def test_known_orthogonal_factors_have_expected_angle_orientation() -> None:
    angle = 0.37
    c, s = math.cos(angle), math.sin(angle)
    # This proper planar rotation is exactly the sandwich action of
    # cos(angle/2) - sin(angle/2) e_0 e_1.
    weight = torch.tensor([[c, -s], [s, c]], dtype=F64)
    module = decompose_ga(weight)
    torch.testing.assert_close(module.reconstruct(), weight, rtol=1e-12, atol=1e-12)
    x = torch.tensor([[1.0, 0.0]], dtype=F64)
    torch.testing.assert_close(module(x), torch.tensor([[c, s]], dtype=F64), rtol=1e-12, atol=1e-12)


def test_decomposition_accepts_requested_cpu_device_and_rejects_bad_shapes() -> None:
    weight = torch.randn(3, 2, dtype=torch.float32)
    module = GARotorLinear.decompose(weight, device="cpu")
    assert module.coefficient_dtype == torch.float64
    assert module.reconstruct().dtype == torch.float64
    with pytest.raises(ValueError, match="rank"):
        decompose_ga(weight, rank=4)
    with pytest.raises(ValueError, match="bias"):
        decompose_ga(weight, bias=torch.zeros(2))
