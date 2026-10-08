"""Wall-clock micro-benchmarks for linear layers.

Latencies are MEASURED (``time.perf_counter``); parameter counts are exact; ``estimated_flops`` are
formula ESTIMATES (multiply+add counts per the module's own accounting), never measured FLOP/s.
Pass ``sync=torch.cuda.synchronize`` when timing CUDA so launches are included in the interval.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch
from torch import nn

from rotor.math.cascade import RotorLinear
from rotor.math.lowrank import LowRankLinear


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` in [0, 100] of an ascending list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def time_callable(
    fn: Callable[[], object], warmup: int = 3, iters: int = 20, sync: Callable[[], object] | None = None
) -> dict:
    """Time ``fn()``: ``warmup`` untimed calls, then ``iters`` timed calls (each bracketed by ``sync``).

    Returns ``{"median_ms", "p10_ms", "p90_ms", "iters"}`` (measured).
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    samples: list[float] = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": _percentile(samples, 10.0),
        "p90_ms": _percentile(samples, 90.0),
        "iters": iters,
    }


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def estimated_flops(module: nn.Module, in_features: int, out_features: int, batch_size: int) -> int | None:
    """Formula-based multiply+add estimate for one forward over ``batch_size`` rows; ``None`` if unknown.

    - dense ``nn.Linear``: ``2 * in * out * bs``
    - ``RotorLinear``: ``spec.flops_per_token() * bs`` (6 per Givens pair, 1 per diagonal entry,
      residual ``2 r (in+out)``)
    - ``LowRankLinear``: ``2 * r * (in + out) * bs``
    """
    if isinstance(module, RotorLinear):
        return module.cascade.spec.flops_per_token() * batch_size
    if isinstance(module, LowRankLinear):
        return 2 * module.A.shape[1] * (in_features + out_features) * batch_size
    if isinstance(module, nn.Linear):
        return 2 * in_features * out_features * batch_size
    return None


def _sync_for(device: torch.device) -> Callable[[], object] | None:
    return (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None


@torch.no_grad()
def linear_microbench(
    in_f: int,
    out_f: int,
    batch_sizes: list[int],
    impls: dict[str, nn.Module],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 3,
    iters: int = 20,
    seed: int = 0,
) -> list[dict]:
    """Time ``impl(x)`` with ``x = randn(bs, in_f)`` for every impl and batch size.

    Each impl is moved to ``device``/``dtype`` (in place). One row per ``(impl, batch_size)``:
    ``impl, batch_size, in_features, out_features, dtype, device, median_ms, p10_ms, p90_ms, iters,
    params`` (exact) and ``estimated_flops`` (formula estimate; ``None`` for unrecognised modules).
    Inputs are identical across impls for a given batch size (seeded generator).
    """
    dev = torch.device(device)
    sync = _sync_for(dev)
    rows: list[dict] = []
    for module in impls.values():
        module.to(device=dev, dtype=dtype).eval()
    for bs in batch_sizes:
        gen = torch.Generator().manual_seed(seed + int(bs))
        x = torch.randn(int(bs), in_f, generator=gen).to(device=dev, dtype=dtype)
        for name, module in impls.items():
            timing = time_callable(lambda m=module, x=x: m(x), warmup=warmup, iters=iters, sync=sync)
            rows.append(
                {
                    "impl": name,
                    "batch_size": int(bs),
                    "in_features": in_f,
                    "out_features": out_f,
                    "dtype": str(dtype).removeprefix("torch."),
                    "device": str(dev),
                    **timing,
                    "params": count_params(module),
                    "estimated_flops": estimated_flops(module, in_f, out_f, int(bs)),
                }
            )
    return rows
