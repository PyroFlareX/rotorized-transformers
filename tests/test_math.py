"""Behavioral tests for ``rotor.math`` (float64, CPU, deterministic)."""

from __future__ import annotations

import math

import pytest
import torch

from rotor.math import (
    CascadeSpec,
    GivensStage,
    LowRankLinear,
    RotorCascade,
    RotorLinear,
    Topology,
    givens_decompose,
    givens_reconstruct,
    seeded_permutation,
    splitmix64,
    stage_matrix,
    stage_pairs,
    svd_givens_baseline,
    truncated_svd,
)

F64 = torch.float64
MASK = (1 << 64) - 1


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=_gen(seed), dtype=F64)


def _randomize(module: torch.nn.Module, seed: int = 0, scale: float = 1.0) -> None:
    g = _gen(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=F64))


def _reference_splitmix64(state: int) -> tuple[int, int]:
    """Independent re-statement of the reference algorithm (Vigna), written without the module under test."""
    state = (state + 0x9E3779B97F4A7C15) % (1 << 64)
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 % (1 << 64)
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB % (1 << 64)
    return state, z ^ (z >> 31)


# ---------------------------------------------------------------- Givens stages


@pytest.mark.parametrize("kind", ["brick", "butterfly", "seeded"])
@pytest.mark.parametrize("n", [2, 7, 16])
def test_givens_stage_matches_dense_matrix_and_transpose_inverts(kind: str, n: int) -> None:
    pairs = stage_pairs(kind, n, 1, 3)
    theta = _randn(len(pairs), seed=n)
    x = _randn(5, n, seed=n + 1)
    stage = GivensStage(pairs, n)

    G = stage_matrix(pairs, theta, n)
    y = stage.apply(x, theta)
    torch.testing.assert_close(y, x @ G.T, rtol=1e-12, atol=1e-12)

    # G is orthogonal and the transpose-stage applies G^T = G^-1
    torch.testing.assert_close(G @ G.T, torch.eye(n, dtype=F64), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(stage.apply(y, theta, transpose=True), x, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(stage.apply(x, theta, transpose=True), x @ G, rtol=1e-12, atol=1e-12)


def test_givens_stage_rotation_convention() -> None:
    stage = GivensStage(((0, 1),), 3)
    t = torch.tensor([0.7], dtype=F64)
    x = torch.tensor([[1.0, 2.0, 5.0]], dtype=F64)
    y = stage.apply(x, t)
    c, s = math.cos(0.7), math.sin(0.7)
    expected = torch.tensor([[c * 1.0 - s * 2.0, s * 1.0 + c * 2.0, 5.0]], dtype=F64)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)


def test_givens_stage_with_no_pairs_is_identity() -> None:
    stage = GivensStage((), 1)
    x = _randn(3, 1)
    assert stage.apply(x, torch.empty(0, dtype=F64)) is x


# ---------------------------------------------------------------- topology


@pytest.mark.parametrize("kind", ["brick", "butterfly", "seeded"])
@pytest.mark.parametrize("n", [2, 3, 5, 8, 12, 33])
def test_topology_pairs_are_disjoint_in_range_and_sorted(kind: str, n: int) -> None:
    for stage in range(2 * max(1, math.ceil(math.log2(n))) + 1):
        pairs = stage_pairs(kind, n, stage, 5)
        flat = [i for p in pairs for i in p]
        assert len(flat) == len(set(flat)), f"{kind} n={n} stage={stage}: pairs overlap"
        assert all(0 <= a < b < n for a, b in pairs)
        assert list(pairs) == sorted(pairs)
        if kind == "seeded":
            assert len(pairs) == n // 2


@pytest.mark.parametrize("kind", ["brick", "butterfly", "seeded"])
def test_topology_is_deterministic_across_calls(kind: str) -> None:
    first = [stage_pairs(kind, 21, s, 9) for s in range(6)]
    stage_pairs.cache_clear()
    second = [stage_pairs(kind, 21, s, 9) for s in range(6)]
    assert first == second
    t = Topology(kind, 9)
    assert [t.pairs(21, s) for s in range(6)] == first


def test_seeded_topology_differs_by_seed_and_stage() -> None:
    assert stage_pairs("seeded", 32, 0, 0) != stage_pairs("seeded", 32, 0, 1)
    assert stage_pairs("seeded", 32, 0, 0) != stage_pairs("seeded", 32, 1, 0)
    # Non-seeded topologies ignore the seed.
    assert stage_pairs("brick", 32, 0, 0) == stage_pairs("brick", 32, 0, 1)
    assert stage_pairs("butterfly", 32, 2, 0) == stage_pairs("butterfly", 32, 2, 1)


def test_topology_known_small_patterns() -> None:
    assert stage_pairs("brick", 5, 0) == ((0, 1), (2, 3))
    assert stage_pairs("brick", 5, 1) == ((1, 2), (3, 4))
    assert stage_pairs("butterfly", 8, 0) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert stage_pairs("butterfly", 8, 1) == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert stage_pairs("butterfly", 8, 2) == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert stage_pairs("brick", 1, 0) == ()


def test_topology_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        Topology("spiral")
    with pytest.raises(ValueError):
        stage_pairs("spiral", 8, 0)


def test_topology_dict_roundtrip_and_angle_count() -> None:
    t = Topology("seeded", 12)
    assert Topology.from_dict(t.to_dict()) == t
    assert t.angle_count(10, 3) == 15
    assert Topology("brick").angle_count(5, 4) == 2 * 4


@pytest.mark.parametrize("n", [2, 3, 5, 6, 7, 10, 12, 33, 100])
def test_butterfly_covers_all_coordinates_within_log2_stages(n: int) -> None:
    stages = max(1, math.ceil(math.log2(n)))
    touched = {i for s in range(stages) for p in stage_pairs("butterfly", n, s, 0) for i in p}
    assert touched == set(range(n))

    # ... and the log2 stages actually connect every coordinate to every other one (a single component).
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(stages):
        for a, b in stage_pairs("butterfly", n, s, 0):
            parent[find(a)] = find(b)
    assert len({find(i) for i in range(n)}) == 1


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
def test_butterfly_power_of_two_is_dense_after_log2_stages(n: int) -> None:
    g = _gen(0)
    P = torch.eye(n, dtype=F64)
    for s in range((n - 1).bit_length()):
        pairs = stage_pairs("butterfly", n, s, 0)
        theta = 0.3 + torch.rand(len(pairs), generator=g, dtype=F64)
        P = stage_matrix(pairs, theta, n) @ P
    assert bool((P.abs() > 1e-12).all())


def test_splitmix64_known_outputs_for_state_zero() -> None:
    state = 0
    outs = []
    for _ in range(3):
        state, out = splitmix64(state)
        outs.append(out)
    assert outs[0] == 0xE220A8397B1DCDAF
    assert outs[1] == 0x6E789E6AA1B965F4
    assert outs[2] == 0x06C45D188009454F


def test_splitmix64_matches_independent_reference_and_stays_64_bit() -> None:
    for seed in (0, 1, 0xDEADBEEF, MASK, 12345678901234567890):
        a = b = seed
        for _ in range(20):
            a, oa = splitmix64(a)
            b, ob = _reference_splitmix64(b)
            assert (a, oa) == (b, ob)
            assert 0 <= oa <= MASK and 0 <= a <= MASK


def test_seeded_permutation_is_deterministic_permutation_following_stream_definition() -> None:
    n, seed, stage = 17, 4, 2
    perm = seeded_permutation(n, seed, stage)
    assert sorted(perm) == list(range(n))
    assert perm == seeded_permutation(n, seed, stage)
    assert perm != seeded_permutation(n, seed + 1, stage)

    # Reference Fisher-Yates over the documented stream: state = seed * 0x100000001B3 + stage.
    state = (seed * 0x100000001B3 + stage) & MASK
    ref = list(range(n))
    for i in range(n - 1, 0, -1):
        state, out = _reference_splitmix64(state)
        j = out % (i + 1)
        ref[i], ref[j] = ref[j], ref[i]
    assert perm == ref


# ---------------------------------------------------------------- cascade spec


def test_cascade_spec_counts_and_roundtrip() -> None:
    spec = CascadeSpec(6, 10, 3, 4, Topology("seeded", 2), residual_rank=2)
    assert spec.width == 10
    assert spec.angles_per_rotor() == 4 * 5
    assert spec.param_breakdown() == {"angles": 3 * 20, "diagonals": 6 + 10 + 2 * 10, "residual": 2 * 16}
    assert spec.param_count() == sum(spec.param_breakdown().values())
    assert spec.dense_param_count() == 60
    assert spec.flops_per_token() == 6 * 60 + 36 + 2 * 2 * 16
    assert CascadeSpec.from_dict(spec.to_dict()) == spec


def test_cascade_spec_validation() -> None:
    with pytest.raises(ValueError):
        CascadeSpec(4, 4, 0, 2)
    with pytest.raises(ValueError):
        CascadeSpec(4, 4, 1, -1)
    with pytest.raises(ValueError):
        CascadeSpec(4, 4, 1, 2, residual_rank=-1)


# ---------------------------------------------------------------- cascade operator


@pytest.mark.parametrize("kind", ["brick", "butterfly", "seeded"])
@pytest.mark.parametrize("k", [1, 2])
def test_rotor_only_cascade_is_orthogonal_when_square(kind: str, k: int) -> None:
    n = 12
    spec = CascadeSpec(n, n, k, 5, Topology(kind, 1))
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=3)
    with torch.no_grad():
        cascade.d_in.fill_(1.0)
        cascade.d_mid.fill_(1.0)
        cascade.d_out.fill_(1.0)
    W = cascade.dense_weight().detach()
    torch.testing.assert_close(W @ W.T, torch.eye(n, dtype=F64), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(W.T @ W, torch.eye(n, dtype=F64), rtol=1e-10, atol=1e-10)


def test_fresh_cascade_with_zero_angles_is_identity_on_square() -> None:
    cascade = RotorCascade(CascadeSpec(9, 9, 2, 4, Topology("butterfly")), dtype=F64)
    torch.testing.assert_close(cascade.dense_weight(), torch.eye(9, dtype=F64), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("n_in", "n_out"), [(5, 9), (9, 5), (8, 8), (3, 10), (10, 3)], ids=["in<out", "in>out", "equal", "tall", "wide"]
)
@pytest.mark.parametrize("kind", ["brick", "butterfly", "seeded"])
@pytest.mark.parametrize("residual_rank", [0, 2])
def test_cascade_forward_equals_dense_weight_matmul(n_in: int, n_out: int, kind: str, residual_rank: int) -> None:
    spec = CascadeSpec(n_in, n_out, 3, 3, Topology(kind, 7), residual_rank=residual_rank)
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=n_in * 31 + n_out)
    x = _randn(4, 6, n_in, seed=99)

    W = cascade.dense_weight()
    assert W.shape == (n_out, n_in)
    torch.testing.assert_close(cascade(x), x @ W.T, rtol=1e-10, atol=1e-10)
    # The batched forward must agree with the vector-by-vector forward.
    torch.testing.assert_close(cascade(x[1, 2]), cascade(x)[1, 2], rtol=1e-12, atol=1e-12)


def test_cascade_residual_adds_low_rank_term() -> None:
    spec = CascadeSpec(6, 9, 2, 2, Topology("brick"), residual_rank=3)
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=5)
    no_res = CascadeSpec(6, 9, 2, 2, Topology("brick"), residual_rank=0)
    base = RotorCascade(no_res, dtype=F64)
    with torch.no_grad():
        for name in ("angles", "d_in", "d_mid", "d_out"):
            getattr(base, name).copy_(getattr(cascade, name))
    U, V = cascade.residual.U.detach(), cascade.residual.V.detach()
    torch.testing.assert_close(cascade.dense_weight(), base.dense_weight() + U @ V, rtol=1e-10, atol=1e-10)


def test_cascade_residual_starts_at_zero_output() -> None:
    cascade = RotorCascade(CascadeSpec(6, 9, 1, 2, Topology("brick"), residual_rank=3), dtype=F64)
    assert cascade.residual is not None
    x = _randn(3, 6)
    torch.testing.assert_close(cascade.residual(x), torch.zeros(3, 9, dtype=F64), rtol=0, atol=0)


@pytest.mark.parametrize(
    "spec",
    [
        CascadeSpec(5, 9, 1, 3, Topology("butterfly")),
        CascadeSpec(9, 5, 3, 4, Topology("brick"), residual_rank=2),
        CascadeSpec(8, 8, 2, 3, Topology("seeded", 3), residual_rank=1),
        CascadeSpec(7, 4, 4, 0, Topology("brick")),
    ],
)
def test_param_count_matches_counted_tensor_numel(spec: CascadeSpec) -> None:
    cascade = RotorCascade(spec, dtype=F64)
    assert sum(p.numel() for p in cascade.parameters()) == spec.param_count()
    groups = cascade.param_groups()
    assert sum(p.numel() for ps in groups.values() for p in ps) == spec.param_count()
    assert set(groups) == ({"angles", "scales", "residual"} if spec.residual_rank else {"angles", "scales"})
    assert sum(p.numel() for p in groups["angles"]) == spec.param_breakdown()["angles"]
    assert sum(p.numel() for p in groups["scales"]) == spec.param_breakdown()["diagonals"]
    if spec.residual_rank:
        assert sum(p.numel() for p in groups["residual"]) == spec.param_breakdown()["residual"]


def test_rotor_linear_matches_nn_linear_semantics_with_bias() -> None:
    spec = CascadeSpec(5, 7, 2, 3, Topology("butterfly"))
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=2)
    bias = _randn(7, seed=8)
    layer = RotorLinear(cascade, bias)
    assert (layer.in_features, layer.out_features) == (5, 7)
    x = _randn(3, 5, seed=4)
    expected = torch.nn.functional.linear(x, cascade.dense_weight(), bias)
    torch.testing.assert_close(layer(x), expected, rtol=1e-10, atol=1e-10)
    assert RotorLinear(cascade).bias is None


def test_cascade_forward_preserves_input_dtype() -> None:
    cascade = RotorCascade(CascadeSpec(6, 4, 2, 3, Topology("brick")), dtype=torch.float32)
    _randomize(cascade, seed=0)
    x = torch.randn(2, 6, generator=_gen(0)).to(torch.bfloat16)
    assert cascade(x).dtype == torch.bfloat16


def test_init_from_dense_is_deterministic_and_matches_frobenius_norm_at_zero_angles() -> None:
    W = _randn(6, 10, seed=1)
    spec = CascadeSpec(10, 6, 2, 3, Topology("butterfly"))
    a = RotorCascade(spec, dtype=F64)
    a.init_from_dense(W, angle_std=0.0)
    # all-rotation operator with equal scales a: ||W_hat||_F = a^(k+1) * sqrt(min(in,out)) when angles are zero
    # init_from_dense computes ||W||_F in float32, so only float32-level agreement is expected
    torch.testing.assert_close(a.dense_weight().norm(), W.norm(), rtol=1e-6, atol=0)

    b1, b2 = RotorCascade(spec, dtype=F64), RotorCascade(spec, dtype=F64)
    b1.init_from_dense(W, angle_std=0.05, generator=_gen(11))
    b2.init_from_dense(W, angle_std=0.05, generator=_gen(11))
    assert torch.equal(b1.angles, b2.angles)
    assert float(b1.angles.abs().max()) > 0


# ---------------------------------------------------------------- gradients


def test_gradcheck_rectangular_cascade_all_parameters_and_input() -> None:
    spec = CascadeSpec(3, 5, 2, 2, Topology("butterfly"), residual_rank=2)
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=0)
    x = _randn(2, 3, seed=1).requires_grad_(True)
    params = (cascade.angles, cascade.d_in, cascade.d_mid, cascade.d_out, cascade.residual.U, cascade.residual.V)
    for p in params:
        assert p.requires_grad

    # gradcheck perturbs the tensors in place, so evaluate through the module's own parameters.
    def fn(x_, *_params):
        return cascade(x_)

    assert torch.autograd.gradcheck(fn, (x, *params), eps=1e-6, atol=1e-6, rtol=1e-5)

    def dense_fn(*_params):
        return cascade.dense_weight()

    assert torch.autograd.gradcheck(dense_fn, params, eps=1e-6, atol=1e-6, rtol=1e-5)


def test_gradcheck_individual_parameter_groups_square_brick() -> None:
    spec = CascadeSpec(4, 4, 2, 4, Topology("brick"))
    cascade = RotorCascade(spec, dtype=F64)
    _randomize(cascade, seed=3)
    x = _randn(3, 4, seed=2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a: cascade(a), (x,), eps=1e-6, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradcheck(lambda t: cascade(x.detach()), (cascade.angles,), eps=1e-6, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- fitting capacity


def test_cascade_recovers_orthogonal_matrix_generated_by_same_spec() -> None:
    """A full-width brick cascade can represent what a cascade of the same spec produces.

    The target is built by a random cascade of the identical spec; a perturbed copy is optimized
    back onto it. This checks expressiveness + gradient quality, not that n stages fit *every* O(n).
    """
    spec = CascadeSpec(4, 4, 1, 4, Topology("brick"))
    target = RotorCascade(spec, dtype=F64)
    with torch.no_grad():
        target.angles.copy_(_randn(*target.angles.shape, seed=1))
    W = target.dense_weight().detach()
    # W is orthogonal by construction
    torch.testing.assert_close(W @ W.T, torch.eye(4, dtype=F64), rtol=1e-10, atol=1e-10)

    fit = RotorCascade(spec, dtype=F64)
    with torch.no_grad():
        fit.angles.copy_(target.angles + 0.1 * _randn(*target.angles.shape, seed=2))
    opt = torch.optim.LBFGS(
        [fit.angles], lr=1.0, max_iter=200, tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn="strong_wolfe"
    )
    denom = (W**2).sum()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = ((fit.dense_weight() - W) ** 2).sum() / denom
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    rel = float((fit.dense_weight() - W).detach().norm() / W.norm())
    assert rel < 1e-4


# ---------------------------------------------------------------- low-rank / SVD baselines


@pytest.mark.parametrize("rank", [1, 3, 5, 8])
def test_truncated_svd_matches_eckart_young(rank: int) -> None:
    W = _randn(10, 8, seed=4)
    A, B = truncated_svd(W, rank)
    assert A.shape == (10, rank) and B.shape == (rank, 8)
    S = torch.linalg.svdvals(W)
    expected = float(S[rank:].pow(2).sum().sqrt())
    assert float((W - A @ B).norm()) == pytest.approx(expected, rel=1e-9, abs=1e-12)
    # balanced split
    torch.testing.assert_close(A.norm(dim=0) ** 2, B.norm(dim=1) ** 2, rtol=1e-9, atol=1e-12)


def test_truncated_svd_clamps_rank_and_preserves_dtype() -> None:
    W = _randn(4, 6, seed=0)
    A, B = truncated_svd(W, 100)
    assert A.shape[1] == 4
    torch.testing.assert_close(A @ B, W, rtol=1e-10, atol=1e-10)
    A32, B32 = truncated_svd(W.float(), 2)
    assert A32.dtype == B32.dtype == torch.float32
    A0, B0 = truncated_svd(W, 0)
    assert A0.shape == (4, 0) and B0.shape == (0, 6)


def test_truncated_svd_is_better_than_any_other_rank_r_sample() -> None:
    W = _randn(9, 7, seed=6)
    A, B = truncated_svd(W, 2)
    best = float((W - A @ B).norm())
    g = _gen(0)
    for _ in range(10):
        A2 = torch.randn(9, 2, generator=g, dtype=F64)
        B2 = torch.linalg.lstsq(A2, W).solution
        assert float((W - A2 @ B2).norm()) >= best - 1e-12


@pytest.mark.parametrize(("m", "r"), [(6, 6), (9, 4), (12, 1), (5, 5)])
def test_givens_decompose_reconstruct_roundtrip(m: int, r: int) -> None:
    Q, _ = torch.linalg.qr(_randn(m, r, seed=m * 7 + r))
    angles, signs = givens_decompose(Q)
    assert angles.numel() == r * m - r * (r + 1) // 2
    assert signs.shape == (r,)
    assert set(signs.tolist()) <= {-1.0, 1.0}
    Q_hat = givens_reconstruct(angles, signs, m)
    assert Q_hat.shape == (m, r)
    torch.testing.assert_close(Q_hat, Q, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(Q_hat.T @ Q_hat, torch.eye(r, dtype=F64), rtol=1e-10, atol=1e-10)


def test_givens_decompose_angle_count_equals_stiefel_count_up_to_nonzero_rows() -> None:
    m, r = 9, 4
    Q, _ = torch.linalg.qr(_randn(m, r, seed=2))
    angles, _ = givens_decompose(Q)
    # The elimination sequence is one rotation per below-diagonal entry: r*m - r*(r+1)/2.
    assert angles.numel() == r * m - r * (r + 1) // 2


@pytest.mark.parametrize("rank", [1, 3, 6])
def test_svd_givens_baseline_equals_truncated_svd_and_is_cheaper_than_dense_rank_r(rank: int) -> None:
    W = _randn(14, 11, seed=rank)
    res = svd_givens_baseline(W, rank)
    A, B = truncated_svd(W, rank)
    assert res["rank"] == rank
    torch.testing.assert_close(res["W_hat"], A @ B, rtol=1e-9, atol=1e-9)

    out_f, in_f = W.shape
    stiefel = (rank * out_f - rank * (rank + 1) // 2) + (rank * in_f - rank * (rank + 1) // 2)
    assert res["params"]["angles"] == stiefel
    assert res["params"]["singular_values"] == rank
    assert res["param_count"] == stiefel + rank
    # strictly fewer parameters than dense factors A, B of the same rank (the saving is the Stiefel dimension)
    assert res["param_count"] < rank * (in_f + out_f)


def test_svd_givens_baseline_clamps_rank_and_preserves_dtype() -> None:
    W = _randn(5, 4, seed=0).float()
    res = svd_givens_baseline(W, 99)
    assert res["rank"] == 4
    assert res["W_hat"].dtype == torch.float32
    torch.testing.assert_close(res["W_hat"], W, rtol=1e-4, atol=1e-4)


def test_lowrank_linear_forward_matches_dense_truncation() -> None:
    W = _randn(9, 6, seed=3)
    bias = _randn(9, seed=4)
    x = _randn(5, 6, seed=5)
    layer = LowRankLinear.from_dense(W, 3, bias)
    A, B = truncated_svd(W, 3)
    assert (layer.in_features, layer.out_features) == (6, 9)
    torch.testing.assert_close(layer(x), x @ (A @ B).T + bias, rtol=1e-10, atol=1e-10)
    assert sum(p.numel() for p in layer.parameters()) == 3 * (6 + 9) + 9
    # full rank reproduces the dense layer
    full = LowRankLinear.from_dense(W, 6)
    torch.testing.assert_close(full(x), x @ W.T, rtol=1e-10, atol=1e-10)
    assert full.bias is None
