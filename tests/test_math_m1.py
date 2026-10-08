"""M1 mathematical-prototype gate tests for ``rotor.math`` (CPU, deterministic seeds).

Complements ``tests/test_math.py`` (which covers topology, stage/matrix agreement, orthogonality of
square rotor-only cascades, gradcheck, parameter counts); nothing here duplicates those cases.
Everything is float64 unless the test is explicitly about precision.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from rotor.math import CascadeSpec, GivensStage, RotorCascade, RotorLinear, Topology, stage_matrix

F64 = torch.float64
KINDS = ["brick", "butterfly", "seeded"]
# (in, out) shapes: square, in < out (pad), in > out (truncate)
SHAPES = [(6, 6), (4, 7), (7, 4)]
SHAPE_IDS = ["square", "in<out", "in>out"]


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = F64) -> torch.Tensor:
    return torch.randn(*shape, generator=_gen(seed), dtype=dtype)


def _set_params(cascade: RotorCascade, seed: int, angle_std: float = 1.0, diag_std: float = 0.1) -> None:
    """Random angles ~ N(0, angle_std); diagonals ~ 1 + N(0, diag_std); residual ~ N(0, 1)."""
    g = _gen(seed)
    with torch.no_grad():
        cascade.angles.copy_(angle_std * torch.randn(cascade.angles.shape, generator=g, dtype=cascade.angles.dtype))
        for d in (cascade.d_in, cascade.d_mid, cascade.d_out):
            d.copy_(1.0 + diag_std * torch.randn(d.shape, generator=g, dtype=d.dtype))
        if cascade.residual is not None:
            for p in cascade.residual.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype))


def _selection_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Pad/truncate selection: ``out x in`` with ones on the main diagonal."""
    return torch.eye(n_out, n_in, dtype=F64)


# ---------------------------------------------------------------- identity


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("n_in", "n_out"), SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("k", [1, 3])
def test_zero_angles_unit_diagonals_give_exact_identity_or_selection_matrix(
    kind: str, n_in: int, n_out: int, k: int
) -> None:
    spec = CascadeSpec(n_in, n_out, k, 4, Topology(kind, 5))
    cascade = RotorCascade(spec, dtype=F64)
    W = cascade.dense_weight()
    assert W.shape == (n_out, n_in)
    assert torch.equal(W, _selection_matrix(n_in, n_out))
    if n_in == n_out:
        assert torch.equal(W, torch.eye(n_in, dtype=F64))
    # the same holds for the forward pass on arbitrary data (exact, not approximate)
    x = _randn(5, n_in, seed=1)
    expected = torch.zeros(5, n_out, dtype=F64)
    m = min(n_in, n_out)
    expected[:, :m] = x[:, :m]
    assert torch.equal(cascade(x), expected)


# ---------------------------------------------------------------- pure scaling


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("n_in", "n_out"), SHAPES, ids=SHAPE_IDS)
def test_zero_angles_arbitrary_diagonals_give_exact_diagonal_products(kind: str, n_in: int, n_out: int) -> None:
    k = 3
    n = max(n_in, n_out)
    m = min(n_in, n_out)
    cascade = RotorCascade(CascadeSpec(n_in, n_out, k, 3, Topology(kind, 2)), dtype=F64)
    g = _gen(3)
    with torch.no_grad():
        cascade.d_in.copy_(torch.randn(n_in, generator=g, dtype=F64))
        cascade.d_mid.copy_(torch.randn(k - 1, n, generator=g, dtype=F64))
        cascade.d_out.copy_(torch.randn(n_out, generator=g, dtype=F64))
    W = cascade.dense_weight().detach()
    # D_0 * (D_1 ... D_{k-1}) * D_k on the first min(in, out) coordinates, nothing elsewhere.
    # Same multiplication order as the forward pass (D_k, D_{k-1}, ..., D_1, D_0), so the match is exact.
    expected = torch.zeros(n_out, n_in, dtype=F64)
    for j in range(m):
        v = cascade.d_in[j]
        for i in range(k - 2, -1, -1):
            v = v * cascade.d_mid[i, j]
        expected[j, j] = v * cascade.d_out[j]
    assert torch.equal(W, expected)

    # placement: entries of D_mid outside [0, min(in, out)) multiply padded/dropped coordinates only,
    # d_in[j >= out] and d_out[j >= in] never reach the operator's support
    perturbed = RotorCascade(cascade.spec, dtype=F64)
    perturbed.load_state_dict(cascade.state_dict())
    with torch.no_grad():
        perturbed.d_mid[:, m:] *= 7.0
        perturbed.d_in[m:] *= 7.0
        perturbed.d_out[m:] *= 7.0
    assert torch.equal(perturbed.dense_weight().detach(), W)
    # whereas perturbing any D_0 / D_mid / D_in entry inside the support changes exactly that diagonal entry
    for name in ("d_in", "d_out"):
        p2 = RotorCascade(cascade.spec, dtype=F64)
        p2.load_state_dict(cascade.state_dict())
        with torch.no_grad():
            getattr(p2, name)[0] *= 2.0
        diff = p2.dense_weight().detach() - W
        assert float(diff[0, 0].abs()) > 0
        diff[0, 0] = 0
        assert torch.equal(diff, torch.zeros_like(diff))
    p3 = RotorCascade(cascade.spec, dtype=F64)
    p3.load_state_dict(cascade.state_dict())
    with torch.no_grad():
        p3.d_mid[1, 0] *= 2.0
    diff = p3.dense_weight().detach() - W
    assert float(diff[0, 0].abs()) > 0
    diff[0, 0] = 0
    assert torch.equal(diff, torch.zeros_like(diff))


# ---------------------------------------------------------------- exact known rotations


def _block(t: float) -> torch.Tensor:
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s], [s, c]], dtype=F64)


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (math.pi / 2, [[0.0, -1.0], [1.0, 0.0]]),
        (math.pi, [[-1.0, 0.0], [0.0, -1.0]]),
        (math.pi / 4, [[math.sqrt(0.5), -math.sqrt(0.5)], [math.sqrt(0.5), math.sqrt(0.5)]]),
    ],
    ids=["pi/2", "pi", "pi/4"],
)
def test_single_pair_known_angles_produce_analytic_block_at_right_coordinates(theta: float, expected) -> None:
    block = torch.tensor(expected, dtype=F64)

    # stage level: pair (1, 3) inside a width-5 vector, other coordinates untouched
    n = 5
    G = stage_matrix(((1, 3),), torch.tensor([theta], dtype=F64), n)
    ref = torch.eye(n, dtype=F64)
    idx = [1, 3]
    for r in range(2):
        for c in range(2):
            ref[idx[r], idx[c]] = block[r, c]
    torch.testing.assert_close(G, ref, rtol=0, atol=1e-15)
    x = _randn(4, n, seed=2)
    stage = GivensStage(((1, 3),), n)
    torch.testing.assert_close(stage.apply(x, torch.tensor([theta], dtype=F64)), x @ ref.T, rtol=0, atol=1e-14)

    # cascade level, pair (0, 1) of a 3-wide brick (single stage, single pair; coordinate 2 passes through)
    c1 = RotorCascade(CascadeSpec(3, 3, 1, 1, Topology("brick")), dtype=F64)
    assert c1.angles.shape == (1, 1)
    with torch.no_grad():
        c1.angles.fill_(theta)
    ref1 = torch.eye(3, dtype=F64)
    ref1[:2, :2] = block
    torch.testing.assert_close(c1.dense_weight(), ref1, rtol=0, atol=1e-15)

    # cascade level, pair (1, 2): stage 0 (pair (0, 1)) at angle 0, stage 1 (pair (1, 2)) at theta
    c2 = RotorCascade(CascadeSpec(3, 3, 1, 2, Topology("brick")), dtype=F64)
    assert c2.angles.shape == (1, 2)
    with torch.no_grad():
        c2.angles.copy_(torch.tensor([[0.0, theta]], dtype=F64))
    ref2 = torch.eye(3, dtype=F64)
    ref2[1:, 1:] = block
    torch.testing.assert_close(c2.dense_weight(), ref2, rtol=0, atol=1e-15)


def test_quarter_turn_matches_textbook_matrix_exactly_up_to_cos_pi_over_2() -> None:
    # sin(pi/2) == 1.0 exactly; cos(pi/2) == 6.1e-17 -> the "0" entries are that, and nothing larger
    c = RotorCascade(CascadeSpec(2, 2, 1, 1, Topology("brick")), dtype=F64)
    with torch.no_grad():
        c.angles.fill_(math.pi / 2)
    W = c.dense_weight().detach()
    assert W[1, 0] == 1.0 and W[0, 1] == -1.0
    assert abs(float(W[0, 0])) < 1e-16 and abs(float(W[1, 1])) < 1e-16
    # acts on x = (x0, x1) as (-x1, x0)
    x = torch.tensor([[2.0, 5.0]], dtype=F64)
    torch.testing.assert_close(c(x), torch.tensor([[-5.0, 2.0]], dtype=F64), rtol=0, atol=1e-15)


def test_composition_of_two_stages_equals_product_of_analytic_matrices() -> None:
    t0, t1 = 0.7, -1.3
    # n = 3 brick: stage 0 = pair (0, 1), stage 1 = pair (1, 2); R = S_0 S_1 (stage 1 acts first)
    c = RotorCascade(CascadeSpec(3, 3, 1, 2, Topology("brick")), dtype=F64)
    with torch.no_grad():
        c.angles.copy_(torch.tensor([[t0, t1]], dtype=F64))
    S0 = torch.eye(3, dtype=F64)
    S0[:2, :2] = _block(t0)
    S1 = torch.eye(3, dtype=F64)
    S1[1:, 1:] = _block(t1)
    torch.testing.assert_close(c.dense_weight(), S0 @ S1, rtol=0, atol=1e-14)
    # and it is not the other order
    assert float((c.dense_weight().detach() - S1 @ S0).abs().max()) > 1e-3


def test_two_rotors_with_diagonals_equal_analytic_D0_R1_D1_R2_D2() -> None:
    n = 3
    spec = CascadeSpec(n, n, 2, 2, Topology("brick"))
    c = RotorCascade(spec, dtype=F64)
    g = _gen(8)
    angles = torch.randn(2, 2, generator=g, dtype=F64)
    d_in, d_mid, d_out = (torch.randn(n, generator=g, dtype=F64) for _ in range(3))
    with torch.no_grad():
        c.angles.copy_(angles)
        c.d_in.copy_(d_in)
        c.d_mid.copy_(d_mid[None])
        c.d_out.copy_(d_out)

    def rotor(row: torch.Tensor) -> torch.Tensor:
        S0 = torch.eye(n, dtype=F64)
        S0[:2, :2] = _block(float(row[0]))
        S1 = torch.eye(n, dtype=F64)
        S1[1:, 1:] = _block(float(row[1]))
        return S0 @ S1

    expected = torch.diag(d_out) @ rotor(angles[0]) @ torch.diag(d_mid) @ rotor(angles[1]) @ torch.diag(d_in)
    torch.testing.assert_close(c.dense_weight(), expected, rtol=1e-12, atol=1e-12)
    x = _randn(4, n, seed=9)
    torch.testing.assert_close(c(x), x @ expected.T, rtol=1e-12, atol=1e-12)


def test_two_rotors_same_pair_compose_by_adding_angles() -> None:
    c = RotorCascade(CascadeSpec(2, 2, 2, 1, Topology("brick")), dtype=F64)
    with torch.no_grad():
        c.angles.copy_(torch.tensor([[0.4], [0.9]], dtype=F64))
    torch.testing.assert_close(c.dense_weight(), _block(1.3), rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [2, 5, 8])
def test_transpose_stage_with_angle_t_equals_stage_with_minus_t(kind: str, n: int) -> None:
    pairs = Topology(kind, 4).pairs(n, 1)
    stage = GivensStage(pairs, n)
    theta = _randn(len(pairs), seed=n)
    x = _randn(3, 2, n, seed=n + 1)
    torch.testing.assert_close(stage.apply(x, theta, transpose=True), stage.apply(x, -theta), rtol=0, atol=1e-15)
    # dense: G(-t) == G(t)^T, and G(t)^T G(t) == I
    G = stage_matrix(pairs, theta, n)
    torch.testing.assert_close(stage_matrix(pairs, -theta, n), G.T, rtol=0, atol=1e-15)
    torch.testing.assert_close(stage.apply(stage.apply(x, theta), theta, transpose=True), x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_transposed_rotor_is_the_matrix_transpose_of_the_rotor(kind: str) -> None:
    n = 9
    c = RotorCascade(CascadeSpec(n, n, 1, 4, Topology(kind, 6)), dtype=F64)
    _set_params(c, seed=5)
    with torch.no_grad():
        c.d_in.fill_(1.0)
        c.d_out.fill_(1.0)
    R = c.dense_weight().detach()
    eye = torch.eye(n, dtype=F64)
    # rows of _rotor(eye) are R e_r -> the result is R^T; transposed rotor gives R
    torch.testing.assert_close(c._rotor(eye, 0).T, R, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(c._rotor(eye, 0, transpose=True).T, R.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(c._rotor(c._rotor(eye, 0), 0, transpose=True), eye, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- norm preservation


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("n_in", "n_out"), [(16, 16), (11, 11), (9, 14)], ids=["pow2", "odd", "pad"])
def test_rotor_only_cascade_preserves_two_norm(kind: str, n_in: int, n_out: int) -> None:
    spec = CascadeSpec(n_in, n_out, 4, 5, Topology(kind, 1))
    c = RotorCascade(spec, dtype=F64)
    with torch.no_grad():
        c.angles.copy_(2.0 * torch.randn(c.angles.shape, generator=_gen(n_in), dtype=F64))
    x = _randn(32, n_in, seed=3)
    y = c(x)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- serialization determinism


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("residual_rank", [0, 2])
def test_spec_roundtrip_and_identical_forward_after_roundtrip(kind: str, residual_rank: int) -> None:
    spec = CascadeSpec(7, 11, 3, 4, Topology(kind, 123), residual_rank=residual_rank)
    d = spec.to_dict()
    # to_dict is plain JSON data and deterministic
    text = json.dumps(d, sort_keys=True)
    assert json.dumps(spec.to_dict(), sort_keys=True) == text
    spec2 = CascadeSpec.from_dict(json.loads(text))
    assert spec2 == spec
    assert spec2.to_dict() == d
    assert spec2.topology == spec.topology and spec2.param_breakdown() == spec.param_breakdown()

    a = RotorCascade(spec, dtype=F64)
    _set_params(a, seed=17)
    b = RotorCascade(spec2, dtype=F64)
    b.load_state_dict(a.state_dict())
    x = _randn(6, 7, seed=18)
    assert torch.equal(a(x), b(x))
    assert torch.equal(a.dense_weight(), b.dense_weight())
    # a second forward of the same module is bit-identical too (no hidden state)
    assert torch.equal(a(x), a(x))
    # RotorLinear wraps the same operator
    assert torch.equal(RotorLinear(a)(x), a(x))


# ---------------------------------------------------------------- numerical precision


def _wide_cascade(dtype: torch.dtype) -> RotorCascade:
    n = 256
    spec = CascadeSpec(n, n, 8, 8, Topology("butterfly", 0))
    c = RotorCascade(spec, dtype=F64)
    _set_params(c, seed=21, angle_std=0.5, diag_std=0.1)
    return c if dtype == F64 else c.to(dtype)


def test_float32_forward_matches_float64_reference_on_wide_cascade() -> None:
    ref = _wide_cascade(F64)
    c32 = _wide_cascade(torch.float32)
    x = _randn(16, 256, seed=22)
    with torch.no_grad():
        y_ref = ref(x)
        y32 = c32(x.float())
    assert y32.dtype == torch.float32
    rel = float((y32.double() - y_ref).norm() / y_ref.norm())
    assert rel < 1e-5, f"float32 relative error {rel:.3e}"


def test_bfloat16_stored_parameters_have_bounded_error(record_property) -> None:
    ref = _wide_cascade(F64)
    stored = _wide_cascade(F64)
    with torch.no_grad():
        for p in stored.parameters():  # storage round trip: float64 master -> bfloat16 -> float64
            p.copy_(p.to(torch.bfloat16).to(F64))
    x = _randn(16, 256, seed=22)
    with torch.no_grad():
        y_ref = ref(x)
        rel = float((stored(x) - y_ref).norm() / y_ref.norm())
    record_property("bfloat16_param_roundtrip_rel_error_measured", rel)
    print(f"bfloat16 parameter round-trip relative error (256-wide, 8 rotors): {rel:.4e}")
    assert 0.0 < rel < 2e-2


# ---------------------------------------------------------------- gradient flow


@pytest.mark.parametrize(("n_in", "n_out"), [(5, 9), (9, 5)], ids=["in<out", "in>out"])
def test_every_parameter_receives_nonzero_gradient_on_rectangular_cascade_with_residual(n_in: int, n_out: int) -> None:
    spec = CascadeSpec(n_in, n_out, 3, 4, Topology("butterfly", 0), residual_rank=2)
    c = RotorCascade(spec, dtype=F64)
    _set_params(c, seed=31)
    x = _randn(8, n_in, seed=32)
    target = _randn(8, n_out, seed=33)
    loss = ((c(x) - target) ** 2).sum()
    loss.backward()
    named = dict(c.named_parameters())
    assert set(named) == {"angles", "d_in", "d_mid", "d_out", "residual.U", "residual.V"}
    for name, p in named.items():
        assert p.grad is not None, name
        assert p.grad.shape == p.shape, name
        assert bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().max()) > 0, f"{name} received an all-zero gradient"
    # param_groups cover exactly the parameters, none twice
    groups = c.param_groups()
    flat = [p for ps in groups.values() for p in ps]
    assert {id(p) for p in flat} == {id(p) for p in c.parameters()} and len(flat) == len(named)


def test_every_scalar_parameter_is_live_when_all_widths_are_used() -> None:
    # in < out: no coordinate is dropped before D_0, so every angle and diagonal entry influences the output
    spec = CascadeSpec(5, 9, 3, 4, Topology("butterfly", 0), residual_rank=2)
    c = RotorCascade(spec, dtype=F64)
    _set_params(c, seed=41)
    x = _randn(16, 5, seed=42)
    loss = (c(x) * _randn(16, 9, seed=43)).sum()
    loss.backward()
    for name, p in c.named_parameters():
        assert bool((p.grad != 0).all()), f"{name}: {(p.grad == 0).sum().item()} zero gradient entries"


# ---------------------------------------------------------------- init_from_dense


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("n", [8, 13])
def test_init_from_dense_zero_angles_square_has_frobenius_norm_of_W(kind: str, k: int, n: int) -> None:
    W = _randn(n, n, seed=n + k, dtype=torch.float32).to(F64) * 3.0
    c = RotorCascade(CascadeSpec(n, n, k, 4, Topology(kind, 2)), dtype=F64)
    c.init_from_dense(W, angle_std=0.0)
    assert float(c.angles.detach().abs().max()) == 0.0
    # init computes ||W||_F in float32 -> float32-level agreement
    torch.testing.assert_close(c.dense_weight().norm(), W.norm(), rtol=1e-6, atol=0)
    # at zero angles the operator is a pure diagonal a^(k+1) * I
    D = c.dense_weight()
    torch.testing.assert_close(D, torch.eye(n, dtype=F64) * D[0, 0], rtol=1e-12, atol=1e-12)


def test_init_from_dense_with_nonzero_angles_keeps_norm_for_orthogonal_part() -> None:
    n, k = 12, 3
    W = _randn(n, n, seed=2)
    c = RotorCascade(CascadeSpec(n, n, k, 4, Topology("butterfly")), dtype=F64)
    c.init_from_dense(W, angle_std=0.3, generator=_gen(4))
    # rotations are orthogonal and diagonals are a common scalar -> the norm is unchanged by the angles
    torch.testing.assert_close(c.dense_weight().norm(), W.norm(), rtol=1e-6, atol=0)
