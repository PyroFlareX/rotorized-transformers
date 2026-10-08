"""Greedy generation with the model's KV cache: latency/throughput measurement and matched-output comparison.

Decoding is deterministic (argmax, no sampling), so two models run on the same prompt can be compared token
by token with :func:`compare_generations`. All reported times and memory peaks are measured.

Definitions (measured):

* ``ttft_seconds`` -- time from the start of prefill until the first generated token is available
  (prefill forward over the whole prompt + argmax of the last position).
* ``prefill_tokens_per_second`` -- ``prompt_tokens / ttft_seconds``. It therefore includes the single
  last-position head projection and argmax of the first token.
* ``decode_tokens_per_second`` -- ``(generated_tokens - 1) / decode_seconds``, where ``decode_seconds`` covers
  every step after the first token (one single-token forward over the KV cache + argmax each). ``None`` if only
  one token is generated.

No warmup is performed: the first call in a process includes allocator/kernel start-up costs. Callers that need
steady-state numbers should call :func:`measure_generation` twice and keep the second result.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn

from rotor.environment import ResourceMonitor
from rotor.evaluation.lm import eval_mode


def _check_prompt(prompt_ids: torch.Tensor, max_new_tokens: int) -> None:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError(f"prompt_ids must have shape (1, T) with T >= 1; got {tuple(prompt_ids.shape)}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")


def _default_sync(device: str | torch.device) -> Callable[[], None] | None:
    if torch.device(device).type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return None


def _step(model: nn.Module, input_ids: torch.Tensor, past=None):
    """One forward over ``input_ids`` (prefill when ``past`` is None) -> (greedy next token ``(1, 1)``, cache)."""
    kwargs = {"logits_to_keep": 1} if past is None else {"past_key_values": past}
    out = model(input_ids=input_ids, use_cache=True, **kwargs)
    return out.logits[:, -1].argmax(dim=-1, keepdim=True), out.past_key_values


@torch.no_grad()
def greedy_decode(
    model: nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int, device: str | torch.device = "cpu"
) -> list[int]:
    """Greedy-decode ``max_new_tokens`` tokens after ``prompt_ids`` (``(1, T)``) using the KV cache (untimed)."""
    _check_prompt(prompt_ids, max_new_tokens)
    tokens: list[torch.Tensor] = []
    with eval_mode(model):
        token, past = _step(model, prompt_ids.to(device))
        tokens.append(token)
        for _ in range(max_new_tokens - 1):
            token, past = _step(model, token, past)
            tokens.append(token)
    return [int(t.item()) for t in tokens]


@torch.no_grad()
def measure_generation(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: str | torch.device = "cpu",
    sync: Callable[[], None] | None = None,
) -> dict:
    """Greedy generation with timing; see the module docstring for the metric definitions.

    ``sync`` is called before every clock read (defaults to ``torch.cuda.synchronize`` on CUDA devices, nothing
    on CPU). Returns ``prompt_tokens``, ``generated_tokens``, ``ttft_seconds``, ``prefill_tokens_per_second``,
    ``decode_seconds``, ``decode_tokens_per_second``, ``total_seconds``, ``resources`` (``ResourceMonitor``
    dict: peak RSS bytes -- a process-lifetime high-water mark -- and peak device bytes when CUDA is present),
    ``generated_ids`` and ``measured: True``.
    """
    _check_prompt(prompt_ids, max_new_tokens)
    sync = sync if sync is not None else _default_sync(device)
    prompt = prompt_ids.to(device)

    def now() -> float:
        if sync is not None:
            sync()
        return time.perf_counter()

    tokens: list[torch.Tensor] = []
    with eval_mode(model), ResourceMonitor() as mon:
        t_start = now()
        token, past = _step(model, prompt)
        t_first = now()
        tokens.append(token)
        for _ in range(max_new_tokens - 1):
            token, past = _step(model, token, past)
            tokens.append(token)
        t_end = now()
    ids = [int(t.item()) for t in tokens]
    n = len(ids)
    ttft = t_first - t_start
    decode = t_end - t_first
    return {
        "prompt_tokens": int(prompt.shape[1]),
        "generated_tokens": n,
        "ttft_seconds": ttft,
        "prefill_tokens_per_second": prompt.shape[1] / ttft if ttft > 0 else None,
        "decode_seconds": decode,
        "decode_tokens_per_second": (n - 1) / decode if n > 1 and decode > 0 else None,
        "total_seconds": t_end - t_start,
        "resources": mon.as_dict(),
        "generated_ids": ids,
        "measured": True,
    }


def compare_generations(ids_a: list[int], ids_b: list[int]) -> dict:
    """Matched-output comparison of two greedy generations: longest common prefix and exact equality."""
    prefix = 0
    for a, b in zip(ids_a, ids_b, strict=False):
        if a != b:
            break
        prefix += 1
    return {
        "match_prefix_len": prefix,
        "exact_match": len(ids_a) == len(ids_b) and prefix == len(ids_a),
        "len_a": len(ids_a),
        "len_b": len(ids_b),
    }
