"""Next-token loss / perplexity of a causal LM on fixed token windows (measured, not estimated).

Every window is scored over all ``seq_len - 1`` next-token predictions; the sum of cross-entropies
is accumulated in float64 so that the same windows give the same number independent of batching
to within float32 logit noise. ``windows_sha256`` proves which exact token ids were scored.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from rotor.data import windows_fingerprint

# Logit rows per float64 cross-entropy call; bounds the float64 copy of (rows x vocab) logits
# (256 rows x a 250k vocab = 0.5 GB).
_ROW_CHUNK = 256


def check_windows(windows: torch.Tensor) -> None:
    if windows.ndim != 2:
        raise ValueError(f"windows must be 2-D (num_windows, seq_len); got shape {tuple(windows.shape)}")
    if windows.shape[0] < 1 or windows.shape[1] < 2:
        raise ValueError(f"need at least 1 window of >= 2 tokens; got shape {tuple(windows.shape)}")


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[None]:
    """Put ``model`` in eval mode and restore its previous training flag afterwards."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def next_token_nll(logits: torch.Tensor, targets: torch.Tensor, chunk_rows: int = _ROW_CHUNK) -> torch.Tensor:
    """Per-token cross-entropy in float64. ``logits (B, L, V)``, ``targets (B, L)`` -> ``(B, L)``."""
    b, length, vocab = logits.shape
    flat = logits.reshape(b * length, vocab)
    tgt = targets.reshape(-1).to(flat.device)
    out = torch.empty(b * length, dtype=torch.float64, device=flat.device)
    for start in range(0, b * length, chunk_rows):
        stop = start + chunk_rows
        out[start:stop] = F.cross_entropy(flat[start:stop].double(), tgt[start:stop], reduction="none")
    return out.view(b, length)


def batch_token_nll(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Float64 NLL of tokens ``batch[:, 1:]`` given their prefixes: shape ``(B, T - 1)``."""
    logits = model(batch).logits
    if logits.shape[:2] != batch.shape:
        raise ValueError(f"model returned logits {tuple(logits.shape)} for input_ids {tuple(batch.shape)}")
    return next_token_nll(logits[:, :-1], batch[:, 1:])


@torch.no_grad()
def evaluate_lm(
    model: nn.Module, windows: torch.Tensor, batch_size: int = 1, device: str | torch.device = "cpu"
) -> dict:
    """Mean next-token NLL (nats) and perplexity of ``model`` over ``windows`` (``(N, T)`` token ids).

    ``model(input_ids)`` must return an object with ``.logits`` of shape ``(B, T, V)``. The model is
    run in eval mode (previous mode restored) and is not moved; each batch is moved to ``device``.
    """
    check_windows(windows)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    total_nll = 0.0
    tokens = 0
    with eval_mode(model):
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size].to(device)
            nll = batch_token_nll(model, batch)
            total_nll += float(nll.sum())
            tokens += nll.numel()
    mean_nll = total_nll / tokens
    try:
        ppl = math.exp(mean_nll)
    except OverflowError:
        ppl = math.inf
    return {
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "tokens": tokens,
        "windows_sha256": windows_fingerprint(windows.cpu()),
    }
