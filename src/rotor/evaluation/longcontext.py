"""Quality as a function of position in the context window (measured, not estimated).

A rotor/low-rank substitution can degrade long-range behaviour differently from short-range
behaviour, so loss is reported per contiguous position bucket instead of only as a window mean.

Position convention: the position of a scored token is its index ``t`` in the window
(``1 <= t <= seq_len - 1``; token 0 has no prefix and is never scored). Bucket boundaries
``[b0, b1, b2, ...]`` define ``[b0, b1), [b1, b2), ...`` and the last bucket runs to ``seq_len``.
"""

from __future__ import annotations

import torch
from torch import nn

from rotor.evaluation.lm import batch_token_nll, check_windows, eval_mode


def _check_boundaries(boundaries: list[int]) -> list[int]:
    b = [int(x) for x in boundaries]
    if not b:
        raise ValueError("buckets must contain at least one boundary")
    if b[0] < 0 or any(hi <= lo for lo, hi in zip(b, b[1:], strict=False)):
        raise ValueError(f"bucket boundaries must be non-negative and strictly increasing; got {b}")
    return b


@torch.no_grad()
def nll_by_position(
    model: nn.Module,
    windows: torch.Tensor,
    buckets: list[int],
    device: str | torch.device = "cpu",
    batch_size: int = 1,
) -> list[dict]:
    """Mean next-token NLL (nats) per position bucket over ``windows`` (``(N, T)`` token ids).

    Returns ``[{"start", "end", "mean_nll", "tokens"}]`` for every non-empty bucket, in order, where
    ``[start, end)`` are positions of the scored (target) tokens, ``end`` is clipped to ``T`` and
    ``tokens = num_windows * (number of scored positions in the bucket)``. Buckets with no scored
    position (e.g. wholly beyond ``T``, or ``[0, 1)``) are skipped. Accumulation is float64.
    """
    check_windows(windows)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    bounds = _check_boundaries(buckets)
    num_windows, seq_len = windows.shape
    pos_sum = torch.zeros(seq_len - 1, dtype=torch.float64)  # index t-1 holds the summed NLL of token t
    with eval_mode(model):
        for start in range(0, num_windows, batch_size):
            batch = windows[start : start + batch_size].to(device)
            pos_sum += batch_token_nll(model, batch).sum(dim=0).cpu()
    out = []
    edges = [*bounds, seq_len]
    for lo, hi in zip(edges, edges[1:], strict=False):
        first, last = max(lo, 1), min(hi, seq_len)  # scored positions are 1 .. seq_len - 1
        if last <= first:
            continue
        count = (last - first) * num_windows
        out.append(
            {
                "start": lo,
                "end": last,
                "mean_nll": float(pos_sum[first - 1 : last - 1].sum()) / count,
                "tokens": count,
            }
        )
    return out
