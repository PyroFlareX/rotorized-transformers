"""Multiple-choice task accuracy via summed continuation log-likelihood (measured).

No dataset is bundled or downloaded: the caller supplies a JSONL file the user has approved, one object per line::

    {"context": "Question or prompt text", "choices": ["first continuation", "second continuation"], "answer": 1}

``answer`` is the 0-based index of the correct choice. Each choice is tokenised separately from the context
(``add_special_tokens=False``) and appended to it, so whitespace between context and continuation must be part of the
choice string (e.g. ``" Paris"``). The score of a choice is ``sum_t log p(token_t | context, previous tokens)`` over
its continuation tokens, accumulated in float64. Two decision rules are reported on identical inputs:

* ``accuracy`` -- argmax of the summed log-likelihood.
* ``accuracy_norm`` -- argmax of the summed log-likelihood divided by the number of continuation *tokens*.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rotor.evaluation.lm import eval_mode


def load_jsonl_items(path: str | Path) -> list[dict]:
    """Read and validate multiple-choice items (``context``, ``choices``, ``answer``) from a JSONL file."""
    items: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            items.append(_validate_item(obj, f"{path}:{lineno}"))
    if not items:
        raise ValueError(f"{path}: no items found")
    return items


def _validate_item(obj: object, where: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: item must be a JSON object")
    context, choices, answer = obj.get("context"), obj.get("choices"), obj.get("answer")
    if not isinstance(context, str):
        raise ValueError(f"{where}: 'context' must be a string")
    if not isinstance(choices, list) or len(choices) < 2 or not all(isinstance(c, str) and c for c in choices):
        raise ValueError(f"{where}: 'choices' must be a list of >= 2 non-empty strings")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
        raise ValueError(f"{where}: 'answer' must be an int index into 'choices'")
    return {"context": context, "choices": list(choices), "answer": answer}


def _encode(tokenizer, text: str) -> list[int]:
    return [int(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]


@torch.no_grad()
def continuation_loglik(model: nn.Module, context_ids: list[int], cont_ids: list[int], device) -> float:
    """``sum log p(cont | context)`` in float64; ``context_ids`` and ``cont_ids`` must be non-empty."""
    ids = torch.tensor([context_ids + cont_ids], dtype=torch.long, device=device)
    logits = model(ids).logits[0, len(context_ids) - 1 : -1]
    logp = F.log_softmax(logits.double(), dim=-1)
    tgt = torch.tensor(cont_ids, dtype=torch.long, device=logp.device)
    return float(logp.gather(1, tgt[:, None]).sum())


@torch.no_grad()
def multiple_choice_accuracy(
    model: nn.Module, tokenizer, items: list[dict], device: str | torch.device = "cpu"
) -> dict:
    """Accuracy over ``items`` by summed (and token-length-normalised) continuation log-likelihood.

    Returns ``{"accuracy", "accuracy_norm", "n", "per_item"}``; each per-item record holds the per-choice
    ``loglik`` and ``loglik_norm`` lists, the predicted indices and correctness for both rules.
    """
    if not items:
        raise ValueError("items is empty")
    per_item: list[dict] = []
    with eval_mode(model):
        for k, raw in enumerate(items):
            item = _validate_item(raw, f"item {k}")
            ctx = _encode(tokenizer, item["context"])
            if not ctx:
                raise ValueError(f"item {k}: context tokenises to zero tokens")
            logliks: list[float] = []
            lengths: list[int] = []
            for choice in item["choices"]:
                cont = _encode(tokenizer, choice)
                if not cont:
                    raise ValueError(f"item {k}: choice {choice!r} tokenises to zero tokens")
                logliks.append(continuation_loglik(model, ctx, cont, device))
                lengths.append(len(cont))
            norm = [ll / n for ll, n in zip(logliks, lengths, strict=True)]
            pred = max(range(len(logliks)), key=logliks.__getitem__)
            pred_norm = max(range(len(norm)), key=norm.__getitem__)
            per_item.append(
                {
                    "pred": pred,
                    "pred_norm": pred_norm,
                    "answer": item["answer"],
                    "correct": pred == item["answer"],
                    "correct_norm": pred_norm == item["answer"],
                    "loglik": logliks,
                    "loglik_norm": norm,
                    "continuation_tokens": lengths,
                }
            )
    n = len(per_item)
    return {
        "accuracy": sum(r["correct"] for r in per_item) / n,
        "accuracy_norm": sum(r["correct_norm"] for r in per_item) / n,
        "n": n,
        "per_item": per_item,
        "measured": True,
    }
