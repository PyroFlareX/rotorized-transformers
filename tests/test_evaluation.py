"""Evaluation: LM cross-entropy, position buckets, equal-byte baselines + controls, timing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rotor.data import random_windows, windows_fingerprint
from rotor.evaluation.baselines import equal_budget_baselines, gaussian_random, spectrum_matched_random
from rotor.evaluation.generation import compare_generations
from rotor.evaluation.lm import evaluate_lm
from rotor.evaluation.longcontext import nll_by_position
from rotor.evaluation.performance import linear_microbench, time_callable
from rotor.evaluation.prompts import _token_hash, _validate_baseline
from rotor.math.lowrank import LowRankLinear
from rotor.models.qwen3_5 import load_text_model

CONTROLS = ("spectrum_matched_random", "gaussian_random")

# ---------------------------------------------------------------------------------------------------------------------
# LM evaluation
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model(tiny_ckpt: Path):
    return load_text_model(tiny_ckpt, dtype=torch.float32)


@pytest.fixture(scope="module")
def windows() -> torch.Tensor:
    return random_windows(128, 16, 3, seed=0)


def manual_token_nll(model, windows: torch.Tensor) -> torch.Tensor:
    """(N, T-1) float64 NLL of tokens 1..T-1, computed independently of ``rotor.evaluation``."""
    with torch.no_grad():
        logits = model(windows).logits[:, :-1].double()
    logp = torch.log_softmax(logits, dim=-1)
    return -logp.gather(-1, windows[:, 1:].unsqueeze(-1)).squeeze(-1)


def test_evaluate_lm_equals_manual_cross_entropy(tiny_model, windows):
    res = evaluate_lm(tiny_model, windows, batch_size=1)
    nll = manual_token_nll(tiny_model, windows)
    assert res["tokens"] == windows.shape[0] * (windows.shape[1] - 1) == nll.numel()
    assert res["mean_nll"] == pytest.approx(float(nll.mean()), abs=1e-5)
    assert res["perplexity"] == pytest.approx(math.exp(float(nll.mean())), rel=1e-4)
    # random-init tiny model over a 128-token vocab: loss is near ln(128) and finite
    assert math.isfinite(res["mean_nll"]) and 3.0 < res["mean_nll"] < 7.0


def test_evaluate_lm_batching_does_not_change_result(tiny_model, windows):
    a = evaluate_lm(tiny_model, windows, batch_size=1)
    b = evaluate_lm(tiny_model, windows, batch_size=3)
    assert a["tokens"] == b["tokens"]
    assert a["mean_nll"] == pytest.approx(b["mean_nll"], abs=1e-5)
    assert a["windows_sha256"] == b["windows_sha256"]


def test_windows_sha256_is_stable_and_content_addressed(tiny_model, windows):
    first = evaluate_lm(tiny_model, windows)["windows_sha256"]
    assert first == evaluate_lm(tiny_model, windows.clone())["windows_sha256"]
    assert first == windows_fingerprint(windows)
    assert first == windows_fingerprint(random_windows(128, 16, 3, seed=0))
    other = random_windows(128, 16, 3, seed=1)
    assert evaluate_lm(tiny_model, other)["windows_sha256"] != first


def test_evaluate_lm_restores_training_mode_and_validates_input(tiny_model, windows):
    tiny_model.train()
    try:
        evaluate_lm(tiny_model, windows[:1])
        assert tiny_model.training
    finally:
        tiny_model.eval()
    with pytest.raises(ValueError):
        evaluate_lm(tiny_model, windows[0])  # 1-D
    with pytest.raises(ValueError):
        evaluate_lm(tiny_model, windows[:, :1])  # needs >= 2 tokens


def test_nll_by_position_buckets_partition_the_scored_tokens(tiny_model, windows):
    n, t = windows.shape
    total = evaluate_lm(tiny_model, windows)
    buckets = nll_by_position(tiny_model, windows, [0, 4, 8], device="cpu")
    assert [(b["start"], b["end"]) for b in buckets] == [(0, 4), (4, 8), (8, 16)]
    assert sum(b["tokens"] for b in buckets) == n * (t - 1) == total["tokens"]  # position 0 is never scored
    assert buckets[0]["tokens"] == n * 3 and buckets[1]["tokens"] == n * 4 and buckets[2]["tokens"] == n * 8
    weighted = sum(b["mean_nll"] * b["tokens"] for b in buckets) / total["tokens"]
    assert weighted == pytest.approx(total["mean_nll"], abs=1e-5)

    # per-bucket value equals an independent slice of the manual NLL (token t lives at column t-1)
    nll = manual_token_nll(tiny_model, windows)
    for b in buckets:
        lo, hi = max(b["start"], 1), b["end"]
        assert b["mean_nll"] == pytest.approx(float(nll[:, lo - 1 : hi - 1].mean()), abs=1e-5)


def test_nll_by_position_skips_unscored_and_out_of_range_buckets(tiny_model, windows):
    n, t = windows.shape
    buckets = nll_by_position(tiny_model, windows, [0, 1, 6, 12, 40], device="cpu")
    # [0,1) holds only the unscored token 0; [40, ..) is beyond the window
    assert [(b["start"], b["end"]) for b in buckets] == [(1, 6), (6, 12), (12, 16)]
    assert sum(b["tokens"] for b in buckets) == n * (t - 1)
    with pytest.raises(ValueError):
        nll_by_position(tiny_model, windows, [4, 4, 8], device="cpu")
    with pytest.raises(ValueError):
        nll_by_position(tiny_model, windows, [], device="cpu")


# ---------------------------------------------------------------------------------------------------------------------
# Equal-budget baselines and controls
# ---------------------------------------------------------------------------------------------------------------------

OUT_F, IN_F = 48, 32
BYTE_BUDGET = OUT_F * IN_F * 2 // 2  # 50% of the bfloat16 dense bytes
SCHEDULE = {
    "phases": [{"train": ["scales"], "steps": 2}, {"train": ["angles", "scales"], "steps": 4}],
    "lrs": {"angles": 0.02, "scales": 0.02},
    "cosine_decay": True,
}
SPEC_KWARGS = {"topology": {"kind": "butterfly", "seed": 0}, "stages_per_rotor": "auto", "residual_rank": 0}


@pytest.fixture(scope="module")
def W() -> torch.Tensor:
    """48x32 matrix with a decaying spectrum (so truncation error is informative)."""
    g = torch.Generator().manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(OUT_F, IN_F, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(IN_F, IN_F, generator=g, dtype=torch.float64))
    s = 0.8 ** torch.arange(IN_F, dtype=torch.float64)
    return ((U * s) @ V.T).float()


@pytest.fixture(scope="module")
def X() -> torch.Tensor:
    return torch.randn(64, IN_F, generator=torch.Generator().manual_seed(1))


@pytest.fixture(scope="module")
def result(W, X) -> dict:
    return equal_budget_baselines(W, BYTE_BUDGET, "bfloat16", SPEC_KWARGS, SCHEDULE, X=X, seed=0)


def test_baselines_every_method_fits_the_byte_budget(result):
    assert result["budget"]["byte_budget"] == BYTE_BUDGET
    assert result["budget"]["param_budget"] == BYTE_BUDGET // 2
    assert {"svd", "svd_givens", "rotor_weight", "rotor_activation"} <= set(result["methods"])
    ran = 0
    for name, row in result["methods"].items():
        if "skipped" in row or row.get("reference"):
            continue
        ran += 1
        assert 0 < row["stored_bytes"] <= BYTE_BUDGET, (name, row["stored_bytes"])
        assert row["estimated_flops_per_token"] > 0
        assert math.isfinite(row["weight_errors"]["rel_fro"])
    assert ran >= 3
    assert "skipped" not in result["methods"]["svd"] and "skipped" not in result["methods"]["rotor_weight"]
    assert "activation_errors" in result["methods"]["rotor_activation"]
    assert "activation_errors" in result["methods"]["svd"]
    assert any("estimate" in n.lower() for n in result["notes"])
    for ctrl in CONTROLS:
        for name, row in result["controls"][ctrl].items():
            if "skipped" not in row:
                assert row["stored_bytes"] <= BYTE_BUDGET, (ctrl, name)


def test_svd_error_equals_eckart_young_tail(W, result):
    row = result["methods"]["svd"]
    rank = row["rank"]
    assert rank == (BYTE_BUDGET // 2) // (OUT_F + IN_F)  # 768 // 80 = 9
    assert row["params"] == rank * (OUT_F + IN_F) and row["stored_bytes"] == 2 * row["params"]
    s = torch.linalg.svdvals(W.double())
    tail_fro = float(s[rank:].pow(2).sum().sqrt() / s.pow(2).sum().sqrt())
    assert row["weight_errors"]["rel_fro"] == pytest.approx(tail_fro, rel=1e-4)
    assert row["weight_errors"]["rel_spectral"] == pytest.approx(float(s[rank] / s[0]), rel=1e-4)
    # truncated SVD is the optimum over every rank-r matrix, so no rank-r method can beat it in Frobenius norm
    assert row["weight_errors"]["rel_fro"] > 0


def test_no_rank_r_method_beats_eckart_young_but_rotor_errors_are_reported(result):
    svd = result["methods"]["svd"]["weight_errors"]["rel_fro"]
    rotor = result["methods"]["rotor_weight"]["weight_errors"]["rel_fro"]
    assert math.isfinite(rotor) and rotor > 0
    # the rotor row is measured, not assumed better: just record that both are comparable numbers at equal bytes
    assert result["methods"]["rotor_weight"]["stored_bytes"] <= BYTE_BUDGET
    assert svd < 1.0


def test_controls_are_present_and_comparable(W, result):
    assert set(result["controls"]) == set(CONTROLS)
    for ctrl in CONTROLS:
        assert {"svd", "rotor_weight"} <= set(result["controls"][ctrl])
    # spectrum-matched control has W's spectrum, hence the SAME Eckart-Young error for truncated SVD
    sm = result["controls"]["spectrum_matched_random"]["svd"]
    assert sm["rank"] == result["methods"]["svd"]["rank"]
    assert sm["weight_errors"]["rel_fro"] == pytest.approx(
        result["methods"]["svd"]["weight_errors"]["rel_fro"], rel=1e-3
    )
    # an iid Gaussian matrix has a flat spectrum: truncation at the same rank loses much more than for W
    g = result["controls"]["gaussian_random"]["svd"]
    assert g["weight_errors"]["rel_fro"] > 2 * result["methods"]["svd"]["weight_errors"]["rel_fro"]


def test_spectrum_matched_random_preserves_singular_values(W):
    Wc = spectrum_matched_random(W, seed=3)
    assert Wc.shape == W.shape and Wc.dtype == W.dtype
    s, sc = torch.linalg.svdvals(W.double()), torch.linalg.svdvals(Wc.double())
    torch.testing.assert_close(sc, s, atol=1e-5, rtol=1e-4)
    assert float(Wc.double().norm()) == pytest.approx(float(W.double().norm()), rel=1e-5)
    # but the singular VECTORS are not W's
    assert (Wc - W).norm() > 0.5 * W.norm()
    # deterministic per seed, different across seeds
    assert torch.equal(Wc, spectrum_matched_random(W, seed=3))
    assert not torch.equal(Wc, spectrum_matched_random(W, seed=4))


def test_gaussian_random_matches_norm_only(W):
    Wg = gaussian_random(W, seed=2)
    assert Wg.shape == W.shape
    assert float(Wg.double().norm()) == pytest.approx(float(W.double().norm()), rel=1e-5)
    assert torch.equal(Wg, gaussian_random(W, seed=2))
    sg, s = torch.linalg.svdvals(Wg.double()), torch.linalg.svdvals(W.double())
    assert float(sg[-1] / sg[0]) > 5 * float(s[-1] / s[0])  # flat spectrum vs decaying one


def test_baselines_without_calibration_inputs_and_tiny_budget(W):
    res = equal_budget_baselines(W, BYTE_BUDGET, "bfloat16", SPEC_KWARGS, SCHEDULE, X=None, seed=0)
    assert "rotor_activation" not in res["methods"]
    assert "activation_errors" not in res["methods"]["svd"]
    # a budget below one rank-1 SVD component and one rotor is reported as skipped, not silently inflated
    tiny = equal_budget_baselines(W, 16, "bfloat16", SPEC_KWARGS, SCHEDULE, X=None, seed=0)
    assert "skipped" in tiny["methods"]["svd"] and "skipped" in tiny["methods"]["rotor_weight"]
    with pytest.raises(ValueError):
        equal_budget_baselines(W[0], BYTE_BUDGET, "bfloat16", SPEC_KWARGS, SCHEDULE)


# ---------------------------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------------------------


def test_time_callable_reports_positive_ordered_medians():
    a = torch.randn(64, 64, generator=torch.Generator().manual_seed(0))
    calls: list[int] = []

    def fn():
        calls.append(1)
        return a @ a

    res = time_callable(fn, warmup=2, iters=5)
    assert len(calls) == 2 + 5
    assert res["iters"] == 5
    assert 0 < res["p10_ms"] <= res["median_ms"] <= res["p90_ms"]


def test_time_callable_calls_sync_around_each_iteration_and_validates():
    syncs: list[int] = []
    res = time_callable(lambda: None, warmup=0, iters=3, sync=lambda: syncs.append(1))
    assert res["iters"] == 3 and res["median_ms"] >= 0
    assert len(syncs) >= 3
    with pytest.raises(ValueError):
        time_callable(lambda: None, iters=0)
    with pytest.raises(ValueError):
        time_callable(lambda: None, warmup=-1)


def test_linear_microbench_rows_cover_every_impl_and_batch(W):
    dense = torch.nn.Linear(IN_F, OUT_F, bias=False)
    with torch.no_grad():
        dense.weight.copy_(W)
    impls = {"dense": dense, "svd9": LowRankLinear.from_dense(W, rank=9)}
    rows = linear_microbench(IN_F, OUT_F, [1, 4], impls, device="cpu", dtype=torch.float32, warmup=1, iters=3)
    assert [(r["impl"], r["batch_size"]) for r in rows] == [
        ("dense", 1),
        ("svd9", 1),
        ("dense", 4),
        ("svd9", 4),
    ]
    for r in rows:
        assert r["median_ms"] > 0 and r["iters"] == 3
        assert r["params"] > 0 and r["estimated_flops"] > 0
    by = {(r["impl"], r["batch_size"]): r for r in rows}
    assert by[("dense", 1)]["params"] == OUT_F * IN_F
    assert by[("svd9", 1)]["params"] == 9 * (OUT_F + IN_F)
    assert by[("dense", 4)]["estimated_flops"] == 4 * by[("dense", 1)]["estimated_flops"]


def _baseline_row(
    settings: dict,
    input_ids: list[int],
    output_ids: list[int],
    input_text: str = "Explain a falling object:",
) -> dict:

    return {
        "prompt_id": "p1",
        "domain": "science",
        "settings": settings,
        "input_text": input_text,
        "input_ids": input_ids,
        "input_token_count": len(input_ids),
        "input_tokens_sha256": _token_hash(input_ids),
        "output_ids": output_ids,
        "output_token_count": len(output_ids),
        "output_tokens_sha256": _token_hash(output_ids),
    }


def test_prompt_baseline_validates_inputs_then_compares_prefix_and_exact(tmp_path):
    settings = {"prompt_dataset_sha256": "dataset", "max_new_tokens": 4, "dtype": "bfloat16"}
    expected = [{
        "prompt_id": "p1",
        "domain": "science",
        "input_text": "Explain a falling object:",
        "input_ids": [11, 12],
        "input_token_count": 2,
        "input_tokens_sha256": "dummy",
    }]

    expected[0]["input_tokens_sha256"] = _token_hash(expected[0]["input_ids"])
    first = _baseline_row(settings, [11, 12], [21, 22, 23, 24])
    path = tmp_path / "baseline.jsonl"
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    baseline = _validate_baseline(path, expected, settings)
    assert len(baseline) == 1
    assert compare_generations(baseline[0]["output_ids"], [21, 99, 23, 24]) == {
        "match_prefix_len": 1,
        "exact_match": False,
        "len_a": 4,
        "len_b": 4,
    }
    assert compare_generations(baseline[0]["output_ids"], [21, 22, 23, 24])["exact_match"] is True


def test_prompt_baseline_rejects_settings_and_input_mismatches(tmp_path):
    settings = {"prompt_dataset_sha256": "dataset", "max_new_tokens": 4}
    expected = [{
        "prompt_id": "p1", "domain": "science", "input_text": "Explain:",
        "input_ids": [5, 6], "input_token_count": 2,
    }]

    expected[0]["input_tokens_sha256"] = _token_hash(expected[0]["input_ids"])
    row = _baseline_row(settings, [5, 6], [7, 8], input_text="Explain:")
    path = tmp_path / "baseline.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="settings"):
        _validate_baseline(path, expected, {**settings, "max_new_tokens": 5})
    changed = [dict(expected[0], input_ids=[5, 9], input_tokens_sha256=_token_hash([5, 9]))]
    with pytest.raises(ValueError, match="input_ids"):
        _validate_baseline(path, changed, settings)
