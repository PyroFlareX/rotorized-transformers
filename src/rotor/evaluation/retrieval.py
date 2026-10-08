"""Synthetic passkey retrieval (SYNTHETIC probe -- not a standard benchmark and not a substitute for real tasks).

Each item is a deterministic filler passage built from a fixed small word list with one sentence
``"The pass key is NNNNN. Remember it."`` inserted at a seeded relative depth, followed by the question
``"What is the pass key? The pass key is"``. The model is scored by greedy decoding ``len(answer)`` tokens and
requiring an exact token match with the key as tokenised after the question (``" NNNNN"``).

Prompts are a pure function of ``(tokenizer, num_items, context_tokens, seed)``, so a base and a converted model
see identical inputs. Depths cycle evenly over ``(0, 1)`` and are then seed-shuffled so every depth is covered once
``num_items`` is at least the number of distinct depths requested.

Per the project plan, run long contexts only after the short-context path has matched its reference.
"""

from __future__ import annotations

import hashlib
import re

import torch
from torch import nn

from rotor.evaluation.generation import greedy_decode
from rotor.math import splitmix64

SYNTHETIC_LABEL = "synthetic passkey retrieval (generated in-code; not a standard benchmark)"

_WORDS = (
    "the grass is green and the sky is blue while a river runs past an old stone bridge near the quiet town "
    "where people walk slowly under tall trees and watch small birds fly over the fields at the end of day"
).split()
_PIECE = re.compile(r"[A-Za-z]+|\d|[^\w\s]")
_INTRO = "There is important information hidden inside a long text. Find it and remember it. "
_QUESTION = " What is the pass key? The pass key is"


def _rng(seed: int):
    """Deterministic uint64 stream (splitmix64) -- independent of torch / numpy versions."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt() -> int:
        nonlocal state
        state, out = splitmix64(state)
        return out

    return nxt


class SmokeTokenizer:
    """Word/digit-level toy tokenizer with ids hashed into ``[1, vocab_size)`` -- SMOKE TESTS ONLY.

    Used by ``evaluate_model.py`` when the checkpoint directory has no tokenizer files (e.g. the random tiny
    fixtures). Pieces are alphabetic runs, single digits, or single punctuation characters; whitespace is dropped.
    Ids are a stable SHA-256 hash of the piece, so runs are reproducible. It is not invertible and carries no
    linguistic meaning; results obtained with it say nothing about model quality.
    """

    name = "rotor-smoke-word-tokenizer"

    def __init__(self, vocab_size: int):
        if vocab_size < 8:
            raise ValueError("vocab_size must be >= 8")
        self.vocab_size = vocab_size

    def _id(self, piece: str) -> int:
        h = hashlib.sha256(piece.encode("utf-8")).digest()
        return 1 + int.from_bytes(h[:8], "little") % (self.vocab_size - 1)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        del add_special_tokens
        return {"input_ids": [self._id(p) for p in _PIECE.findall(text)]}


def _encode(tokenizer, text: str) -> list[int]:
    return [int(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _filler_ids(tokenizer, n_tokens: int, nxt) -> list[int]:
    """Exactly ``n_tokens`` token ids of word-list filler (generated in chunks, then truncated)."""
    ids: list[int] = []
    while len(ids) < n_tokens:
        sentence = " ".join(_WORDS[nxt() % len(_WORDS)] for _ in range(12)) + ". "
        ids.extend(_encode(tokenizer, sentence))
    return ids[:n_tokens]


def passkey_items(tokenizer, num_items: int, context_tokens: int, seed: int = 0) -> list[dict]:
    """Build ``num_items`` synthetic passkey prompts of exactly ``context_tokens`` tokens (prompt incl. question).

    Returns ``[{"input_ids": LongTensor(1, context_tokens), "answer_ids": list[int], "depth": float,
    "passkey": str}]``. ``depth`` is the relative position (0 = start of the filler, 1 = end) of the key sentence.
    """
    if num_items < 1:
        raise ValueError("num_items must be >= 1")
    intro, question = _encode(tokenizer, _INTRO), _encode(tokenizer, _QUESTION)
    needle_len = max(len(_encode(tokenizer, f"The pass key is {key}. Remember it. ")) for key in ("00000", "99999"))
    fixed = len(intro) + len(question) + needle_len + 4  # slack for tokenisation differences around the key
    filler_tokens = context_tokens - fixed
    if filler_tokens < 8:
        raise ValueError(
            f"context_tokens={context_tokens} too small for a passkey prompt with this tokenizer "
            f"(needs > {fixed + 8} tokens)"
        )
    nxt = _rng(seed)
    depths = [(i + 0.5) / num_items for i in range(num_items)]
    for i in range(num_items - 1, 0, -1):  # seeded Fisher-Yates shuffle
        j = nxt() % (i + 1)
        depths[i], depths[j] = depths[j], depths[i]
    items: list[dict] = []
    for depth in depths:
        key = f"{nxt() % 90000 + 10000}"
        needle = _encode(tokenizer, f"The pass key is {key}. Remember it. ")
        n_filler = context_tokens - len(intro) - len(question) - len(needle)
        if n_filler < 1:
            raise ValueError(f"context_tokens={context_tokens} too small for passkey {key} with this tokenizer")
        filler = _filler_ids(tokenizer, n_filler, nxt)
        cut = round(depth * len(filler))
        ids = intro + filler[:cut] + needle + filler[cut:] + question
        full = _encode(tokenizer, f"{_QUESTION} {key}")
        if full[: len(question)] != question:
            raise ValueError("tokenizer merges the question and the key across their boundary; cannot split answer")
        answer = full[len(question) :]
        if not answer:
            raise ValueError("passkey answer tokenises to zero tokens")
        items.append(
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "answer_ids": answer,
                "depth": depth,
                "passkey": key,
            }
        )
    return items


def passkey_accuracy(model: nn.Module, items: list[dict], device: str | torch.device = "cpu") -> dict:
    """Exact-match accuracy of greedy ``len(answer_ids)``-token continuations; ``by_depth`` lists per-item results."""
    if not items:
        raise ValueError("items is empty")
    rows: list[dict] = []
    for item in items:
        pred = greedy_decode(model, item["input_ids"], len(item["answer_ids"]), device)
        rows.append(
            {
                "depth": item["depth"],
                "correct": pred == list(item["answer_ids"]),
                "predicted_ids": pred,
                "answer_ids": list(item["answer_ids"]),
            }
        )
    rows.sort(key=lambda r: r["depth"])
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "n": len(rows),
        "context_tokens": int(items[0]["input_ids"].shape[1]),
        "by_depth": rows,
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
        "measured": True,
    }
