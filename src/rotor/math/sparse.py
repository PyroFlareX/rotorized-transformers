"""Magnitude-pruned sparse matrices and their exact stored-byte accounting."""

from __future__ import annotations

import torch

INT32_MAX = 2**31 - 1
INDEX_BYTES = 4  # int32


def magnitude_prune(W: torch.Tensor, nnz: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``nnz`` largest-``|w|`` entries of ``W``.

    Returns ``(values, flat_idx)``: ``flat_idx`` (int64, ascending) are row-major flat indices into ``W`` and
    ``values = W.flatten()[flat_idx]``. Ties are broken toward the lower flat index (stable sort), so the result is
    deterministic.
    """
    numel = W.numel()
    nnz = int(nnz)
    if not 0 <= nnz <= numel:
        raise ValueError(f"nnz must be in [0, {numel}], got {nnz}")
    flat = W.detach().reshape(-1)
    order = torch.argsort(flat.abs(), descending=True, stable=True)[:nnz]
    idx, _ = torch.sort(order)
    return flat[idx], idx


def sparse_storage_bytes(nnz: int, shape: tuple[int, int], value_bytes: int) -> dict:
    """Stored bytes of ``nnz`` values in the smaller valid int32 encoding.

    - ``coo_flat``: ``nnz`` int32 row-major flat indices (valid only if ``rows * cols <= 2**31 - 1``),
    - ``csr``: ``nnz`` int32 column indices + ``(rows + 1)`` int32 row pointers (valid only if ``nnz <= 2**31 - 1``
      and ``cols <= 2**31 - 1``).

    Returns ``{"encoding", "stored_bytes", "index_bytes", "value_bytes_total"}`` for the minimum-``stored_bytes``
    valid encoding (ties go to ``coo_flat``). Raises ``ValueError`` if neither int32 encoding can address the matrix.
    """
    rows, cols = int(shape[0]), int(shape[1])
    nnz, value_bytes = int(nnz), int(value_bytes)
    if nnz < 0 or value_bytes < 1:
        raise ValueError(f"need nnz >= 0 and value_bytes >= 1, got nnz={nnz}, value_bytes={value_bytes}")
    values_total = nnz * value_bytes
    candidates: list[tuple[int, str]] = []
    if rows * cols <= INT32_MAX:
        candidates.append((nnz * INDEX_BYTES, "coo_flat"))
    if nnz <= INT32_MAX and cols <= INT32_MAX:
        candidates.append((nnz * INDEX_BYTES + (rows + 1) * INDEX_BYTES, "csr"))
    if not candidates:
        raise ValueError(f"matrix {rows}x{cols} with nnz={nnz} is not addressable with int32 indices")
    index_bytes, encoding = min(candidates, key=lambda c: c[0])
    return {
        "encoding": encoding,
        "stored_bytes": values_total + index_bytes,
        "index_bytes": index_bytes,
        "value_bytes_total": values_total,
    }


def max_nnz_for_budget(byte_budget: int, shape: tuple[int, int], value_bytes: int) -> int:
    """Largest ``nnz <= rows * cols`` whose ``sparse_storage_bytes`` fits ``byte_budget`` (0 if none)."""
    numel = int(shape[0]) * int(shape[1])
    lo, hi = 0, numel
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            fits = sparse_storage_bytes(mid, shape, value_bytes)["stored_bytes"] <= byte_budget
        except ValueError:
            fits = False
        if fits:
            lo = mid
        else:
            hi = mid - 1
    return lo
