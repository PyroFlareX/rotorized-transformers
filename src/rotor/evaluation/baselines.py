"""Equal-stored-byte comparison of structured representations against ordinary low-rank baselines.

Every method below is given the SAME stored-byte budget for one dense matrix ``W (out x in)``:

- ``dense``            the uncompressed matrix, REFERENCE row only (exempt from the budget, error 0),
- ``svd``              truncated SVD, ``rank = budget // (in + out)`` (ordinary low-rank compression),
- ``svd_givens``       truncated SVD whose orthonormal factors are stored as Givens angles,
- ``magnitude_sparse`` top-|w| entries; values + int32 indices (smaller of flat-COO / CSR),
- ``butterfly``        classic butterfly factorization (free 2x2 blocks, log2(n) factors per product),
- ``rotor_weight``     rotor/scaler cascade fit to ``W`` in weight space (no residual),
- ``rotor_residual``   rotor/scaler cascade plus a low-rank residual (~25% of the budget unless given),
- ``rotor_activation`` the same cascade structure fit on calibration inputs ``X`` (only if ``X`` given).

Two CONTROL matrices separate "the method fits W's structure" from "the method fits any matrix with
this spectrum / this norm": ``spectrum_matched_random`` (W's singular values, Haar-random singular
vectors) and ``gaussian_random`` (iid Gaussian, same Frobenius norm). ``svd``, ``magnitude_sparse``,
``butterfly`` and ``rotor_weight`` are re-run on both controls at the same budget.

Errors and parameter counts are measured/exact. ``estimated_flops_per_token`` are formula estimates.
An advantage of a low-rank method over dense is a rank-truncation effect; a rotor gain exists only if the
rotor row beats the ``svd`` (and ``svd_givens``) rows at equal bytes.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rotor.converters.budget import spec_for_budget
from rotor.converters.fit import fit_activation, fit_weight
from rotor.evaluation.reconstruction import activation_errors, weight_errors
from rotor.formats.spec import bytes_per_param
from rotor.math.butterfly import ButterflyMatrix, butterfly_param_count, next_pow2
from rotor.math.cascade import CascadeSpec
from rotor.math.decompose import stiefel_param_count, svd_givens_baseline
from rotor.math.lowrank import rank_for_budget, truncated_svd
from rotor.math.sparse import magnitude_prune, max_nnz_for_budget, sparse_storage_bytes
from rotor.math.topology import Topology
from rotor.training.layerwise import FitSchedule, optimize_cascade


def _haar_orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """``rows x cols`` (``rows >= cols``) matrix with Haar-distributed orthonormal columns, float64.

    QR of an iid Gaussian matrix, with column signs fixed so that ``diag(R) > 0`` (Mezzadri 2007).
    """
    G = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    Q, R = torch.linalg.qr(G, mode="reduced")
    d = torch.diagonal(R)
    return Q * torch.where(d < 0, -1.0, 1.0).to(Q.dtype)


def spectrum_matched_random(W: torch.Tensor, seed: int) -> torch.Tensor:
    """``Q1 diag(S) Q2^T``: same singular values as ``W``, Haar-random singular vectors (``W.dtype``)."""
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {tuple(W.shape)}")
    out_f, in_f = W.shape
    k = min(out_f, in_f)
    S = torch.linalg.svdvals(W.detach().double().cpu())
    g = torch.Generator().manual_seed(int(seed))
    Q1 = _haar_orthonormal(out_f, k, g)
    Q2 = _haar_orthonormal(in_f, k, g)
    return ((Q1 * S) @ Q2.T).to(W.dtype)


def gaussian_random(W: torch.Tensor, seed: int) -> torch.Tensor:
    """iid Gaussian matrix of ``W``'s shape rescaled to ``W``'s exact Frobenius norm (``W.dtype``)."""
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {tuple(W.shape)}")
    g = torch.Generator().manual_seed(int(seed))
    G = torch.randn(tuple(W.shape), generator=g, dtype=torch.float64)
    scale = float(W.detach().double().norm()) / float(G.norm())
    return (G * scale).to(W.dtype)


def _svd_givens_rank(out_f: int, in_f: int, param_budget: int, byte_budget: int, bpp: int) -> int:
    """Largest rank whose ``svd_givens_baseline`` storage fits.

    ``param_count(r) = stiefel(out, r) + stiefel(in, r) + r`` (angles of both factors + singular values) is
    strictly increasing on ``r <= min(in, out)``. The ``2 r`` sign bits are stored too, so
    ``params * bytes_per_param + ceil(2 r / 8)`` must not exceed ``byte_budget``.
    """

    def fits(r: int) -> bool:
        params = stiefel_param_count(out_f, r) + stiefel_param_count(in_f, r) + r
        return params <= param_budget and params * bpp + math.ceil(2 * r / 8) <= byte_budget

    lo, hi = 0, min(out_f, in_f)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _normalize_spec_kwargs(spec_kwargs: dict | None) -> dict:
    kw = dict(spec_kwargs or {})
    topo = kw.get("topology", Topology())
    kw["topology"] = Topology.from_dict(topo) if isinstance(topo, dict) else topo
    kw.setdefault("stages_per_rotor", "auto")
    return kw


def _errors(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor | None) -> dict:
    out = {"weight_errors": weight_errors(W, W_hat)}
    if X is not None:
        out["activation_errors"] = activation_errors(W, W_hat, X)
    return out


def _svd_row(W: torch.Tensor, param_budget: int, bpp: int, X: torch.Tensor | None) -> dict:
    out_f, in_f = W.shape
    rank = min(rank_for_budget(in_f, out_f, param_budget), min(in_f, out_f))
    if rank < 1:
        return {"skipped": f"budget of {param_budget} params is below one rank-1 component ({in_f + out_f} params)"}
    A, B = truncated_svd(W, rank)
    params = rank * (in_f + out_f)
    return {
        "rank": rank,
        "params": params,
        "stored_bytes": params * bpp,
        "index_bytes": 0,
        **_errors(W, A @ B, X),
        "estimated_flops_per_token": 2 * rank * (in_f + out_f),
    }


def _svd_givens_row(W: torch.Tensor, param_budget: int, byte_budget: int, bpp: int, X: torch.Tensor | None) -> dict:
    out_f, in_f = W.shape
    rank = _svd_givens_rank(out_f, in_f, param_budget, byte_budget, bpp)
    if rank < 1:
        return {"skipped": f"budget of {byte_budget} bytes is below one Givens-stored rank-1 component"}
    res = svd_givens_baseline(W, rank)
    params = int(res["param_count"])
    sign_bytes = math.ceil(res["params"]["signs_bits"] / 8)
    angles = int(res["params"]["angles"])
    return {
        "rank": res["rank"],
        "params": params,
        "sign_bytes": sign_bytes,
        "stored_bytes": params * bpp + sign_bytes,
        "index_bytes": 0,
        **_errors(W, res["W_hat"], X),
        # 6 flops per Givens rotation (4 mul + 2 add) + one scale per singular value.
        "estimated_flops_per_token": 6 * angles + res["rank"],
    }


def _rotor_row(
    W: torch.Tensor,
    spec: CascadeSpec,
    schedule: FitSchedule,
    bpp: int,
    X: torch.Tensor | None,
    objective: str,
    seed: int,
    device: str,
) -> dict:
    if objective == "activation":
        if X is None:
            raise ValueError("activation objective requires calibration inputs X")
        cascade, metrics = fit_activation(W, X, spec, schedule, seed=seed, device=device)
    elif objective == "weight":
        cascade, metrics = fit_weight(W, spec, schedule, seed=seed, device=device)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    with torch.no_grad():
        W_hat = cascade.dense_weight().detach().float().cpu()
    params = spec.param_count()
    return {
        "params": params,
        "stored_bytes": params * bpp,
        "index_bytes": 0,
        **_errors(W, W_hat, X),
        "estimated_flops_per_token": spec.flops_per_token(),
        "spec": spec.to_dict(),
        "fit": {
            "objective": objective,
            "final_loss": metrics.get("final_loss"),
            "steps": metrics.get("steps"),
        },
    }


def _dense_row(W: torch.Tensor, bpp: int, X: torch.Tensor | None) -> dict:
    out_f, in_f = W.shape
    return {
        "reference": True,
        "params": in_f * out_f,
        "stored_bytes": in_f * out_f * bpp,
        "index_bytes": 0,
        **_errors(W, W, X),
        "estimated_flops_per_token": 2 * in_f * out_f,
    }


def _sparse_row(W: torch.Tensor, byte_budget: int, bpp: int, X: torch.Tensor | None) -> dict:
    out_f, in_f = W.shape
    nnz = max_nnz_for_budget(byte_budget, (out_f, in_f), bpp)
    if nnz < 1:
        return {"skipped": f"budget of {byte_budget} bytes is below one stored sparse entry"}
    values, idx = magnitude_prune(W, nnz)
    sb = sparse_storage_bytes(nnz, (out_f, in_f), bpp)
    W_hat = torch.zeros(W.numel(), dtype=W.dtype)
    W_hat[idx] = values
    return {
        "nnz": nnz,
        "density": nnz / W.numel(),
        "encoding": sb["encoding"],
        "params": nnz,
        "stored_bytes": sb["stored_bytes"],
        "index_bytes": sb["index_bytes"],
        **_errors(W, W_hat.reshape(out_f, in_f), X),
        "estimated_flops_per_token": 2 * nnz,
    }


def _butterfly_row(
    W: torch.Tensor,
    param_budget: int,
    bpp: int,
    schedule: FitSchedule,
    X: torch.Tensor | None,
    seed: int,
) -> dict:
    out_f, in_f = W.shape
    n = next_pow2(max(out_f, in_f))
    log_n = n.bit_length() - 1
    per_matrix = butterfly_param_count(n, log_n)
    m = param_budget // per_matrix
    if m < 1:
        return {
            "skipped": f"budget of {param_budget} params is below one butterfly product "
            f"({per_matrix} params = log2({n}) factors x 2*{n})"
        }
    bf = ButterflyMatrix(n, m * log_n)
    bf.init_from_dense(W, generator=torch.Generator().manual_seed(int(seed)))
    denom = W.pow(2).sum().clamp_min(torch.finfo(torch.float32).tiny)

    def loss_fn(b: ButterflyMatrix) -> torch.Tensor:
        return (W - b.dense_weight(out_f, in_f)).pow(2).sum() / denom

    history = optimize_cascade(bf, loss_fn, schedule)  # type: ignore[arg-type]  # duck-typed param_groups()
    with torch.no_grad():
        W_hat = bf.dense_weight(out_f, in_f).detach().float().cpu()
    return {
        "params": bf.param_count,
        "stored_bytes": bf.param_count * bpp,
        "index_bytes": 0,
        "width": n,
        "num_matrices": m,
        "num_factors": bf.num_factors,
        **_errors(W, W_hat, X),
        "estimated_flops_per_token": bf.estimated_flops_per_token(),
        "fit": {"objective": "weight", "final_loss": history[-1] if history else None, "steps": len(history)},
    }


def _rotor_residual_row(
    W: torch.Tensor,
    param_budget: int,
    bpp: int,
    kw: dict,
    schedule: FitSchedule,
    X: torch.Tensor | None,
    seed: int,
    device: str,
) -> dict:
    out_f, in_f = W.shape
    rank = int(kw.get("residual_rank", 0))
    if rank <= 0:
        rank = (param_budget // 4) // (in_f + out_f)
        if rank < 1:
            return {"skipped": f"25% of {param_budget} params is below one rank-1 residual ({in_f + out_f} params)"}
    try:
        spec = spec_for_budget(in_f, out_f, param_budget, **{**kw, "residual_rank": rank})
    except ValueError as e:
        return {"skipped": f"no rotor spec with residual_rank={rank} fits {param_budget} params: {e}"}
    row = _rotor_row(W, spec, schedule, bpp, X, "weight", seed, device)
    row["residual_rank"] = rank
    row["residual_params"] = rank * (in_f + out_f)
    return row


def equal_budget_baselines(
    W: torch.Tensor,
    byte_budget: int,
    storage_dtype: str,
    spec_kwargs: dict[str, Any],
    schedule: FitSchedule | dict,
    X: torch.Tensor | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Compare dense, SVD, SVD->Givens, magnitude-sparse, butterfly, rotor (weight / +residual / activation) and
    controls at one byte budget.

    ``W``: dense weight ``(out, in)``; ``X``: optional calibration inputs ``(rows, in)``;
    ``spec_kwargs``: keyword arguments for ``spec_for_budget`` besides the sizes and budget
    (``topology`` as ``Topology`` or ``{"kind","seed"}``, ``stages_per_rotor`` int | "auto",
    ``residual_rank`` (used only by ``rotor_residual``; ``rotor_weight`` always has none), ``adapter``);
    ``schedule``: ``FitSchedule`` or its ``from_dict`` dictionary.

    Returns ``{"budget", "methods", "controls", "notes"}``. ``methods[name]`` and
    ``controls[control][name]`` are rows with ``params``, ``stored_bytes``, ``index_bytes``, ``weight_errors``,
    ``activation_errors`` (only with ``X``) and ``estimated_flops_per_token`` (a formula estimate), plus
    method-specific fields (``rank``, ``nnz``/``encoding``, ``num_matrices``, ``spec``/``fit``); a method that
    cannot fit the budget is ``{"skipped": reason}``. The ``dense`` row carries ``reference: true`` and is the
    only row allowed to exceed ``byte_budget``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (out, in); got shape {tuple(W.shape)}")
    bpp = bytes_per_param(storage_dtype)
    byte_budget = int(byte_budget)
    param_budget = byte_budget // bpp
    W = W.detach().float().cpu()
    if X is not None:
        X = X.detach().float().cpu()
        if X.ndim != 2 or X.shape[1] != W.shape[1]:
            raise ValueError(f"X must be (rows, {W.shape[1]}); got shape {tuple(X.shape)}")
    if not isinstance(schedule, FitSchedule):
        schedule = FitSchedule.from_dict(schedule)
    out_f, in_f = W.shape
    kw = _normalize_spec_kwargs(spec_kwargs)

    try:
        spec: CascadeSpec | None = spec_for_budget(in_f, out_f, param_budget, **{**kw, "residual_rank": 0})
        spec_skip = ""
    except ValueError as e:
        spec, spec_skip = None, f"no rotor spec fits {param_budget} params: {e}"

    def rotor_or_skip(mat: torch.Tensor, objective: str) -> dict:
        if spec is None:
            return {"skipped": spec_skip}
        return _rotor_row(mat, spec, schedule, bpp, X, objective, seed, device)

    reference_row = _dense_row(W, bpp, X)
    methods: dict[str, dict] = {
        "dense": reference_row,
        "svd": _svd_row(W, param_budget, bpp, X),
        "svd_givens": _svd_givens_row(W, param_budget, byte_budget, bpp, X),
        "magnitude_sparse": _sparse_row(W, byte_budget, bpp, X),
        "butterfly": _butterfly_row(W, param_budget, bpp, schedule, X, seed),
        "rotor_weight": rotor_or_skip(W, "weight"),
        "rotor_residual": _rotor_residual_row(W, param_budget, bpp, kw, schedule, X, seed, device),
    }
    if X is not None:
        methods["rotor_activation"] = rotor_or_skip(W, "activation")

    control_mats = {
        "spectrum_matched_random": spectrum_matched_random(W, seed),
        "gaussian_random": gaussian_random(W, seed),
    }
    controls: dict[str, dict] = {}
    for name, Wc in control_mats.items():
        controls[name] = {
            "svd": _svd_row(Wc, param_budget, bpp, X),
            "magnitude_sparse": _sparse_row(Wc, byte_budget, bpp, X),
            "butterfly": _butterfly_row(Wc, param_budget, bpp, schedule, X, seed),
            "rotor_weight": (
                {"skipped": spec_skip}
                if spec is None
                else _rotor_row(Wc, spec, schedule, bpp, X, "weight", seed, device)
            ),
        }

    for where, row in [(f"methods/{k}", r) for k, r in methods.items()] + [
        (f"controls/{c}/{k}", r) for c, rows in controls.items() for k, r in rows.items()
    ]:
        if "skipped" in row or row.get("reference"):
            continue
        if row["stored_bytes"] > byte_budget:  # internal invariant: every non-reference row fits the budget
            raise RuntimeError(f"{where} stores {row['stored_bytes']} bytes > budget {byte_budget}")

    notes = [
        "Equal STORED-BYTE budget is enforced: param_budget = byte_budget // bytes_per_param(storage_dtype) and "
        "every non-reference method's stored_bytes <= byte_budget. The `dense` row is the uncompressed REFERENCE "
        "(reference: true, stored at storage_dtype, error 0) and is exempt from the budget. Compute differs per "
        "method and is reported separately as estimated_flops_per_token; it is not equalized.",
        "estimated_flops_per_token are formula ESTIMATES (multiply+add per input vector), not measurements: "
        "dense 2*in*out; svd 2*r*(in+out); svd_givens 6*angles + r; magnitude_sparse 2*nnz; butterfly 6 per 2x2 block "
        "(n/2 blocks per factor); rotor CascadeSpec.flops_per_token() (6 per Givens pair, 1 per diagonal entry, "
        "2*r*(in+out) residual).",
        "weight_errors/activation_errors are measured on the float32/float64 master values; rounding the stored "
        "parameters to storage_dtype is NOT simulated (storage_dtype only sets bytes per parameter).",
        "svd is ordinary truncated-SVD compression: any gain over dense it shows is a rank-truncation effect, not a "
        "rotor gain. A structural (rotor) gain requires the rotor rows to beat svd and svd_givens at equal bytes.",
        "svd_givens stored_bytes include ceil(2*rank/8) bytes of 1-bit Givens signs on top of params*bytes_per_param.",
        "magnitude_sparse keeps the largest-|w| entries (largest nnz that fits); stored_bytes = nnz*bytes_per_param "
        "+ index_bytes where the index encoding is the smaller of int32 flat COO indices (nnz*4) and int32 CSR "
        "(nnz*4 + (rows+1)*4). It is not fit by gradient descent.",
        "butterfly is the classic factorization with FREE 2x2 blocks (4 params each, not rotations) on a width padded "
        "to the next power of two of max(in,out) (pad/truncate), as a product of m butterfly matrices of log2(n) "
        "factors; m is the largest that fits. Fit by Adam with the same FitSchedule (the block tensor is trained in "
        "every phase whose trainable kinds include 'scales', at the 'scales' learning rate). Skipped if one product "
        "exceeds the budget.",
        "rotor_weight is the full-rank rotor/scaler cascade with NO residual (residual_rank is forced to 0). "
        "rotor_residual adds a low-rank residual: spec_kwargs['residual_rank'] if > 0, otherwise the rank whose "
        "residual parameters r*(in+out) take at most ~25% of the parameter budget.",
        "index_bytes: 0 for svd, svd_givens, butterfly and rotor rows (deterministic topology; no stored indices).",
        "controls: spectrum_matched_random keeps W's singular values with Haar-random singular vectors; "
        "gaussian_random is iid Gaussian with W's Frobenius norm. svd, magnitude_sparse, butterfly and rotor_weight "
        "are re-run on each control at the same budget; control errors are measured against the control matrix "
        "itself (and X for activation_errors).",
        "rotor_activation is fit from scratch with the same schedule and structure as rotor_weight (no warm start "
        "from the weight fit).",
    ]
    return {
        "budget": {
            "byte_budget": byte_budget,
            "storage_dtype": storage_dtype,
            "bytes_per_param": bpp,
            "param_budget": param_budget,
            "in_features": in_f,
            "out_features": out_f,
            "dense_params": in_f * out_f,
            "dense_bytes": in_f * out_f * bpp,
        },
        "methods": methods,
        "controls": controls,
        "notes": notes,
    }


TABLE_COLUMNS = [
    "matrix",
    "section",
    "method",
    "status",
    "params",
    "stored_bytes",
    "index_bytes",
    "rel_fro",
    "rel_spectral",
    "act_rel_fro",
    "estimated_flops_per_token",
]


def result_rows(result: dict, matrix: str = "") -> list[dict]:
    """Flatten one ``equal_budget_baselines`` result into table rows (methods first, then controls).

    ``status`` is ``reference`` (the dense row), ``ok`` or ``skipped: <reason>``. ``estimated_flops_per_token`` is a
    formula estimate; every other numeric column is exact or measured.
    """
    rows: list[dict] = []

    def add(section: str, method: str, row: dict) -> None:
        out: dict[str, Any] = {"matrix": matrix, "section": section, "method": method}
        if "skipped" in row:
            out["status"] = f"skipped: {row['skipped']}"
        else:
            we, ae = row.get("weight_errors") or {}, row.get("activation_errors") or {}
            out.update(
                status="reference" if row.get("reference") else "ok",
                params=row.get("params"),
                stored_bytes=row.get("stored_bytes"),
                index_bytes=row.get("index_bytes"),
                rel_fro=we.get("rel_fro"),
                rel_spectral=we.get("rel_spectral"),
                act_rel_fro=ae.get("rel_fro"),
                estimated_flops_per_token=row.get("estimated_flops_per_token"),
            )
        rows.append(out)

    for method, row in result.get("methods", {}).items():
        add("method", method, row)
    for control, methods in result.get("controls", {}).items():
        for method, row in methods.items():
            add(f"control:{control}", method, row)
    return rows


def lowest_rel_fro(result: dict) -> tuple[str, float] | None:
    """``(method, rel_fro)`` of the compressed method (reference ``dense`` and controls excluded) with the lowest
    MEASURED weight-space relative Frobenius error, or ``None`` if every compressed method was skipped."""
    best: tuple[str, float] | None = None
    for method, row in result.get("methods", {}).items():
        if "skipped" in row or row.get("reference"):
            continue
        err = row["weight_errors"]["rel_fro"]
        if best is None or err < best[1]:
            best = (method, err)
    return best


def interpretation_note(findings: list[tuple[str, str, tuple[str, float] | None]]) -> str:
    """Fixed, claim-free interpretation text. ``findings``: ``(matrix label, target name, lowest_rel_fro(result))``."""
    lines = []
    for matrix, target, best in findings:
        if best is None:
            lines.append(f"- {matrix} @ {target}: every compressed method was skipped at this budget.")
        else:
            lines.append(
                f"- {matrix} @ {target}: the lowest MEASURED weight-space rel_fro among compressed methods was "
                f"`{best[0]}` ({best[1]:.6g}). Single matrix, single seed, single schedule."
            )
    lines += [
        "",
        "Gains from SVD-derived methods (`svd`, `svd_givens`) are rank-truncation effects and are NOT rotor gains. "
        "A rotor gain exists only where a `rotor_*` row beats both SVD-derived rows at equal stored bytes and also "
        "beats the same method's result on the random controls. Errors here are weight-space (and calibration-"
        "activation where an `act_rel_fro` is shown) reconstruction errors, not language-model quality. "
        "`estimated_flops_per_token` are formula estimates, not measurements; compute is not equalized across methods.",
    ]
    return "\n".join(lines)
