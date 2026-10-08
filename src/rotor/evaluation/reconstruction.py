"""Weight- and activation-space reconstruction metrics (computed in float64; measured, not estimated)."""

from __future__ import annotations

import torch


@torch.no_grad()
def weight_errors(W: torch.Tensor, W_hat: torch.Tensor) -> dict[str, float]:
    """Relative Frobenius / spectral error of ``W_hat`` vs ``W`` (both ``out x in``)."""
    W, W_hat = W.double(), W_hat.double()
    E = W - W_hat
    fro = float(W.norm())
    spec = float(torch.linalg.matrix_norm(W, ord=2))
    return {
        "rel_fro": float(E.norm()) / fro if fro else float("nan"),
        "rel_spectral": float(torch.linalg.matrix_norm(E, ord=2)) / spec if spec else float("nan"),
        "max_abs": float(E.abs().max()),
    }


@torch.no_grad()
def activation_errors(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> dict[str, float]:
    """Errors of ``X W_hat^T`` vs ``X W^T`` for calibration inputs ``X`` (``rows x in``)."""
    X = X.double()
    Y, Y_hat = X @ W.double().T, X @ W_hat.double().T
    denom = float(Y.norm())
    cos = torch.nn.functional.cosine_similarity(Y, Y_hat, dim=-1)
    return {
        "rel_fro": float((Y - Y_hat).norm()) / denom if denom else float("nan"),
        "mean_row_cosine": float(cos.mean()),
        "rows": int(X.shape[0]),
    }
