"""Deterministic evaluation/calibration token windows.

Every comparison must use identical token sequences: windows are a pure function of
(source, tokenizer, seq_len, num_windows, seed), and ``windows_fingerprint`` hashes the exact
token ids so reports can prove two runs saw the same data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def tokenize_text_file(tokenizer, path: Path) -> torch.Tensor:
    ids = tokenizer(Path(path).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def windows_from_ids(ids: torch.Tensor, seq_len: int, num_windows: int | None = None, seed: int = 0) -> torch.Tensor:
    """Non-overlapping ``seq_len`` windows; if ``num_windows`` is set, a seeded subset (sorted order)."""
    n = ids.numel() // seq_len
    if n == 0:
        raise ValueError(f"need at least {seq_len} tokens, got {ids.numel()}")
    w = ids[: n * seq_len].view(n, seq_len)
    if num_windows is not None and num_windows < n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:num_windows].sort().values
        w = w[idx]
    return w.contiguous()


def random_windows(vocab_size: int, seq_len: int, num_windows: int, seed: int = 0) -> torch.Tensor:
    """Synthetic uniform-random windows (smoke tests / tiny fixtures only; never a quality benchmark)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (num_windows, seq_len), generator=g)


def windows_fingerprint(windows: torch.Tensor) -> str:
    return hashlib.sha256(windows.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
