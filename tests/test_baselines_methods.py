"""New baseline methods: butterfly factorization, magnitude-sparse storage, and the full equal-budget row set."""

from __future__ import annotations

import math

import pytest
import torch

from rotor.evaluation.baselines import equal_budget_baselines, lowest_rel_fro, result_rows
from rotor.math.butterfly import ButterflyMatrix, butterfly_param_count
from rotor.math.sparse import magnitude_prune, max_nnz_for_budget, sparse_storage_bytes

OUT_F, IN_F = 48, 32
SCHEDULE = {
    "phases": [{"train": ["scales"], "steps": 2}, {"train": ["angles", "scales", "residual"], "steps": 4}],
    "lrs": {"angles": 0.02, "scales": 0.02, "residual": 0.01},
}
SPEC_KWARGS = {"topology": {"kind": "butterfly", "seed": 0}, "stages_per_rotor": "auto", "residual_rank": 0}
REQUIRED_KEYS = {"params", "stored_bytes", "index_bytes", "weight_errors", "estimated_flops_per_token"}
METHODS = {"dense", "svd", "svd_givens", "magnitude_sparse", "butterfly", "rotor_weight", "rotor_residual"}


@pytest.fixture(scope="module")
def W() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(OUT_F, IN_F, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(IN_F, IN_F, generator=g, dtype=torch.float64))
    return ((U * 0.8 ** torch.arange(IN_F, dtype=torch.float64)) @ V.T).float()


@pytest.fixture(scope="module")
def X() -> torch.Tensor:
    return torch.randn(64, IN_F, generator=torch.Generator().manual_seed(1))


# ------------------------------------------------------------------------------------------------ butterfly


@pytest.mark.parametrize("n,factors", [(2, 1), (8, 3), (16, 8)])
def test_identity_initialised_butterfly_is_identity(n, factors):
    bf = ButterflyMatrix(n, factors)
    torch.testing.assert_close(bf.dense_weight(), torch.eye(n))
    x = torch.randn(5, n)
    torch.testing.assert_close(bf(x), x)
    assert bf.param_count == butterfly_param_count(n, factors) == factors * 2 * n


def test_butterfly_forward_equals_x_times_dense_weight_transposed():
    torch.manual_seed(0)
    bf = ButterflyMatrix(16, 8)
    with torch.no_grad():
        bf.blocks.add_(0.3 * torch.randn_like(bf.blocks))
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(bf(x), x @ bf.dense_weight().T, atol=1e-5, rtol=1e-5)
    # rectangular pad/truncate window: input zero-padded, output truncated
    Wr = bf.dense_weight(10, 7)
    assert Wr.shape == (10, 7)
    xs = torch.randn(4, 7)
    full = bf(torch.nn.functional.pad(xs, (0, 9)))[..., :10]
    torch.testing.assert_close(xs @ Wr.T, full, atol=1e-5, rtol=1e-5)


def test_butterfly_blocks_are_free_not_rotations_and_match_explicit_product():
    n, factors = 8, 3
    bf = ButterflyMatrix(n, factors)
    with torch.no_grad():
        bf.blocks.add_(0.4 * torch.randn_like(bf.blocks))
    M = torch.eye(n)
    for f in range(factors):
        d = 1 << (f % 3)
        F = torch.zeros(n, n)
        for k in range(n // 2):
            g, j = divmod(k, d)
            i = g * 2 * d + j
            F[i, i], F[i, i + d] = bf.blocks[f, k, 0, 0], bf.blocks[f, k, 0, 1]
            F[i + d, i], F[i + d, i + d] = bf.blocks[f, k, 1, 0], bf.blocks[f, k, 1, 1]
        M = F @ M
    torch.testing.assert_close(bf.dense_weight(), M, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(bf.dense_weight().T @ bf.dense_weight(), torch.eye(n), atol=1e-3)  # not orthogonal


def test_butterfly_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ButterflyMatrix(12, 2)
    with pytest.raises(ValueError):
        ButterflyMatrix(8, 0)
    with pytest.raises(ValueError):
        ButterflyMatrix(8, 3)(torch.zeros(4))


# ------------------------------------------------------------------------------------------------ sparse


def test_magnitude_prune_keeps_exactly_the_top_nnz_by_abs():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(7, 9, generator=g)
    nnz = 20
    values, idx = magnitude_prune(W, nnz)
    assert values.shape == idx.shape == (nnz,)
    assert torch.equal(idx, torch.sort(idx)[0]) and idx.unique().numel() == nnz
    flat = W.flatten()
    assert torch.equal(values, flat[idx])
    kept_min = flat[idx].abs().min()
    dropped = torch.ones(flat.numel(), dtype=torch.bool)
    dropped[idx] = False
    assert kept_min >= flat[dropped].abs().max()
    expected = torch.sort(flat.abs(), descending=True)[0][:nnz]
    torch.testing.assert_close(torch.sort(values.abs(), descending=True)[0], expected)
    v0, i0 = magnitude_prune(W, 0)
    assert v0.numel() == i0.numel() == 0
    with pytest.raises(ValueError):
        magnitude_prune(W, W.numel() + 1)


def test_sparse_bytes_formula_picks_the_smaller_encoding():
    rows, cols = 100, 50
    # few non-zeros: flat COO (4*nnz) beats CSR (4*nnz + 4*(rows+1))
    s = sparse_storage_bytes(10, (rows, cols), 2)
    assert s["encoding"] == "coo_flat"
    assert s["index_bytes"] == 40 and s["stored_bytes"] == 10 * 2 + 40
    # COO flat indices and CSR column indices are both 4 B per entry, so CSR only adds row pointers: COO never loses
    # when addressable; CSR is the only option when rows*cols exceeds int32.
    big = sparse_storage_bytes(5, (70_000, 70_000), 2)
    assert big["encoding"] == "csr"
    assert big["index_bytes"] == 5 * 4 + 70_001 * 4
    with pytest.raises(ValueError):
        sparse_storage_bytes(5, (70_000, 2**31), 2)
    # budget search is consistent with the formula
    budget = 1000
    nnz = max_nnz_for_budget(budget, (rows, cols), 2)
    assert sparse_storage_bytes(nnz, (rows, cols), 2)["stored_bytes"] <= budget
    assert sparse_storage_bytes(nnz + 1, (rows, cols), 2)["stored_bytes"] > budget
    assert nnz == budget // 6
    assert max_nnz_for_budget(5, (rows, cols), 2) == 0


# ------------------------------------------------------------------------------------------------ equal-budget rows


def test_all_rows_respect_budget_and_have_required_keys(W, X):
    budget = OUT_F * IN_F * 2 * 6 // 10  # 60% of dense bf16 bytes: every method (incl. one butterfly product) fits
    res = equal_budget_baselines(W, budget, "bfloat16", SPEC_KWARGS, SCHEDULE, X=X, seed=0)
    assert METHODS | {"rotor_activation"} == set(res["methods"])
    dense = res["methods"]["dense"]
    assert dense["reference"] is True and dense["stored_bytes"] == OUT_F * IN_F * 2
    assert dense["weight_errors"]["rel_fro"] == 0.0 and dense["index_bytes"] == 0
    for name, row in res["methods"].items():
        assert "skipped" not in row, (name, row)
        assert REQUIRED_KEYS <= set(row), name
        assert "activation_errors" in row, name
        assert math.isfinite(row["weight_errors"]["rel_fro"]), name
        assert row["estimated_flops_per_token"] > 0
        if name != "dense":
            assert row["stored_bytes"] <= budget, (name, row["stored_bytes"], budget)
            assert not row.get("reference")
    sparse = res["methods"]["magnitude_sparse"]
    assert sparse["index_bytes"] > 0 and sparse["stored_bytes"] == sparse["nnz"] * 2 + sparse["index_bytes"]
    assert res["methods"]["rotor_weight"]["spec"]["residual_rank"] == 0
    resid = res["methods"]["rotor_residual"]
    assert resid["spec"]["residual_rank"] == resid["residual_rank"] >= 1
    assert resid["residual_params"] <= budget // 2 // 4 + 1
    bf = res["methods"]["butterfly"]
    assert bf["width"] == 64 and bf["num_matrices"] == 1 and bf["params"] == 6 * 2 * 64
    for ctrl, rows in res["controls"].items():
        assert {"svd", "magnitude_sparse", "butterfly", "rotor_weight"} == set(rows), ctrl
        for name, row in rows.items():
            assert "skipped" in row or row["stored_bytes"] <= budget, (ctrl, name)
    # table helpers
    table = result_rows(res, "w")
    assert len(table) == len(res["methods"]) + 4 * len(res["controls"])
    assert next(r for r in table if r["method"] == "dense")["status"] == "reference"
    best = lowest_rel_fro(res)
    assert best is not None and best[0] != "dense"


def test_explicit_residual_rank_is_honoured(W):
    kw = {**SPEC_KWARGS, "residual_rank": 2}
    res = equal_budget_baselines(W, OUT_F * IN_F * 2, "bfloat16", kw, SCHEDULE, seed=0)
    assert res["methods"]["rotor_residual"]["residual_rank"] == 2
    assert res["methods"]["rotor_weight"]["spec"]["residual_rank"] == 0
    assert "rotor_activation" not in res["methods"]


def test_budget_too_small_skips_rows_without_crashing(W):
    res = equal_budget_baselines(W, 16, "bfloat16", SPEC_KWARGS, SCHEDULE, X=None, seed=0)
    assert res["methods"]["dense"]["reference"] is True
    for name in ("svd", "svd_givens", "butterfly", "rotor_weight", "rotor_residual"):
        assert "skipped" in res["methods"][name], name
        assert isinstance(res["methods"][name]["skipped"], str)
    # 16 bytes still holds two sparse entries (8 B values + 8 B indices at bf16 + int32)
    assert res["methods"]["magnitude_sparse"]["stored_bytes"] <= 16
    zero = equal_budget_baselines(W, 0, "bfloat16", SPEC_KWARGS, SCHEDULE, X=None, seed=0)
    assert "skipped" in zero["methods"]["magnitude_sparse"]
    assert lowest_rel_fro(zero) is None
    rows = result_rows(zero, "w")
    assert any(r["status"].startswith("skipped") for r in rows)


def test_budget_between_one_butterfly_and_two_uses_one_product(W):
    # n=64, log2=6: one product = 768 params = 1536 bf16 bytes; just below it butterfly is skipped
    below = equal_budget_baselines(W, 1534, "bfloat16", SPEC_KWARGS, SCHEDULE, seed=0)
    assert "skipped" in below["methods"]["butterfly"]
    at = equal_budget_baselines(W, 1536, "bfloat16", SPEC_KWARGS, SCHEDULE, seed=0)
    assert at["methods"]["butterfly"]["num_matrices"] == 1 and at["methods"]["butterfly"]["stored_bytes"] == 1536
