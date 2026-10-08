#!/usr/bin/env python
"""Microbenchmark dense nn.Linear vs RotorLinear vs LowRankLinear at equal stored bytes."""

from __future__ import annotations

import argparse
import sys
import threading

import torch
from torch import nn

from rotor.environment import ResourceMonitor, capability_summary, detect_environment, memory_info
from rotor.evaluation.performance import linear_microbench
from rotor.formats import bytes_per_param
from rotor.math import LowRankLinear, RotorCascade, RotorLinear, Topology, rank_for_budget
from rotor.reporting import ExperimentRun, markdown_table

NOTE = (
    "NOTE: the PyTorch reference RotorLinear is NOT an optimized kernel (sequential gather/rotate stages); its "
    "timings say nothing about what a fused kernel could achieve. 'estimated_flops' are formula estimates, "
    "not measurements; 'median_ms' etc. are measured on this device only."
)
FAMILIES = ("ffn.gate", "ffn.up", "ffn.down")
SWAP_SAMPLE_SECONDS = 0.25


class SwapGrowthError(RuntimeError):
    """Swap usage grew during a run that must not be swap-affected."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_f", type=int, help="input features")
    p.add_argument("--out", dest="out_f", type=int, help="output features")
    p.add_argument(
        "--model",
        help="key in configs/models.yaml; shapes come from its `text` dims (no network): ffn.up/gate are "
        "intermediate x hidden, ffn.down is hidden x intermediate (needs --family)",
    )
    p.add_argument("--family", default="ffn.up", choices=FAMILIES, help="projection family (with --model)")
    p.add_argument("--byte-ratio", type=float, default=0.25)
    p.add_argument("--batch-sizes", default="1,8,64")
    p.add_argument("--topology", default="butterfly", choices=["brick", "butterfly", "seeded"])
    p.add_argument("--stages-per-rotor", default="auto")
    p.add_argument("--residual-rank", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N (cuda that is unavailable is an error)")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--refuse-swap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fail the run if swap usage grows while benchmarking (default: on for cuda devices, off for cpu)",
    )
    return p


def _shape_from_model(key: str, family: str) -> tuple[int, int]:
    """(in_features, out_features) of ``family`` from configs/models.yaml ``text`` dims (no network access)."""
    from rotor.storage import load_models, model_entry

    _, entry = model_entry(key, load_models())
    text = entry.get("text") or {}
    try:
        hidden, inter = int(text["hidden_size"]), int(text["intermediate_size"])
    except KeyError as e:
        raise SystemExit(f"configs/models.yaml entry for {key!r} has no text.{e.args[0]}") from e
    if family in ("ffn.up", "ffn.gate"):
        return hidden, inter
    if family == "ffn.down":
        return inter, hidden
    raise SystemExit(f"family {family!r} not supported (choose one of {', '.join(FAMILIES)})")


class _SwapSampler:
    """Samples swap usage from /proc/meminfo in a background thread; reports before/peak/after (measured)."""

    def __enter__(self) -> _SwapSampler:
        self.before = memory_info().get("swap_used_bytes")
        self.peak = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(SWAP_SAMPLE_SECONDS):
            self._observe()

    def _observe(self) -> None:
        cur = memory_info().get("swap_used_bytes")
        if cur is not None and (self.peak is None or cur > self.peak):
            self.peak = cur

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.after = memory_info().get("swap_used_bytes")
        self._observe()

    @property
    def growth(self) -> int | None:
        if self.before is None or self.after is None:
            return None
        return max(self.peak or 0, self.after) - self.before

    def as_dict(self) -> dict:
        return {
            "swap_used_before_bytes": self.before,
            "swap_used_peak_sampled_bytes": self.peak,
            "swap_used_after_bytes": self.after,
            "swap_growth_bytes": self.growth,
            "sample_interval_seconds": SWAP_SAMPLE_SECONDS,
            "measured": True,
        }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model and (args.in_f or args.out_f):
        raise SystemExit("give either --model KEY [--family F] or --in/--out, not both")
    if args.model:
        in_f, out_f = _shape_from_model(args.model, args.family)
    elif args.in_f and args.out_f:
        in_f, out_f = args.in_f, args.out_f
    else:
        raise SystemExit("give --in and --out, or --model KEY [--family F]")
    dtype = getattr(torch, args.dtype)
    try:
        device = torch.device(args.device)
    except RuntimeError as e:
        raise SystemExit(f"invalid --device {args.device!r}: {e}") from e
    if device.type not in ("cpu", "cuda"):
        raise SystemExit(f"unsupported --device {args.device!r}: use cpu or cuda[:N]")

    env = detect_environment()
    for w in capability_summary(env):
        print(f"capability warning: {w}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                f"error: --device {args.device} requested but torch.cuda is not available "
                f"(torch {torch.__version__}, cuda build {torch.version.cuda}, hip build "
                f"{getattr(torch.version, 'hip', None)}). Refusing to fall back to CPU: a CPU timing must "
                "never be reported as a GPU result. Use --device cpu explicitly if a CPU measurement is wanted.",
                file=sys.stderr,
            )
            return 2
        if device.index is not None and device.index >= torch.cuda.device_count():
            print(f"error: {args.device} out of range (device_count={torch.cuda.device_count()})", file=sys.stderr)
            return 2
    refuse_swap = (device.type == "cuda") if args.refuse_swap is None else args.refuse_swap

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    spr = args.stages_per_rotor if args.stages_per_rotor == "auto" else int(args.stages_per_rotor)

    from rotor.converters.budget import spec_for_budget

    # Equal stored bytes: budget is byte_ratio of the dense matrix; all params counted at the same width.
    bpp = bytes_per_param(args.dtype)
    budget_params = int(args.byte_ratio * in_f * out_f)
    spec = spec_for_budget(in_f, out_f, budget_params, Topology(args.topology, args.seed), spr, args.residual_rank)
    rank = rank_for_budget(in_f, out_f, budget_params)
    if rank < 1:
        raise SystemExit("byte ratio too small for rank-1 low-rank baseline")

    target = f"b{round(args.byte_ratio * 100):03d}"
    model_name = args.model or f"{in_f}x{out_f}"
    config = {k: v for k, v in vars(args).items()}
    config.update(
        resolved_in_features=in_f, resolved_out_features=out_f, resolved_refuse_swap=refuse_swap, resolved_target=target
    )
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    try:
        with ExperimentRun(
            "benchmark", model=model_name, method="linear-microbench", target=target, config=config, environment=env
        ) as run:
            run.log(f"shape in={in_f} out={out_f} device={dev_name} dtype={args.dtype} byte_ratio={args.byte_ratio}")
            with _SwapSampler() as swap, ResourceMonitor() as mon:
                g = torch.Generator().manual_seed(args.seed)
                W = torch.randn(out_f, in_f, generator=g) / in_f**0.5
                dense = nn.Linear(in_f, out_f, bias=False)
                with torch.no_grad():
                    dense.weight.copy_(W)
                cascade = RotorCascade(spec, dtype=torch.float32)
                cascade.init_from_dense(W, angle_std=0.01, generator=g)
                impls = {
                    "dense": dense.to(device=device, dtype=dtype),
                    "rotor": RotorLinear(cascade).to(device=device, dtype=dtype),
                    "lowrank": LowRankLinear.from_dense(W, rank).to(device=device, dtype=dtype),
                }
                rows = linear_microbench(in_f, out_f, batch_sizes, impls, device, dtype)
            stored = {
                "dense": in_f * out_f * bpp,
                "rotor": spec.param_count() * bpp,
                "lowrank": rank * (in_f + out_f) * bpp,
            }

            run.metrics["device"] = {"requested": args.device, "name": dev_name, "type": device.type}
            run.metrics["shape"] = {
                "in_features": in_f,
                "out_features": out_f,
                "family": args.family if args.model else None,
            }
            run.metrics["timings"] = rows
            run.metrics["resources"] = mon.as_dict()
            run.metrics["swap"] = swap.as_dict()
            run.metrics["stored_bytes"] = stored  # exact accounting: parameter count x storage width
            run.metrics["rotor_spec"] = spec.to_dict()
            run.metrics["lowrank_rank"] = rank
            run.estimates["flops"] = {f"{r['impl']}/batch{r['batch_size']}": r["estimated_flops"] for r in rows}

            print("stored bytes (exact, from parameter counts):")
            for k, v in stored.items():
                print(f"  {k:<8}{v:>12} B  ({v / stored['dense']:.3f} of dense)")
            print(f"rotor spec: rotors={spec.num_rotors} stages/rotor={spec.stages_per_rotor} topology={args.topology}")
            print("\nMEASURED wall time (ms):")
            print(f"{'impl':<10}{'batch':>7}{'median':>10}{'p10':>10}{'p90':>10}")
            for r in rows:
                print(
                    f"{r['impl']:<10}{r['batch_size']:>7}{r['median_ms']:>10.4f}"
                    f"{r['p10_ms']:>10.4f}{r['p90_ms']:>10.4f}"
                )
            print("\nESTIMATED flops (formula, not measured):")
            for r in rows:
                print(f"{r['impl']:<10}{r['batch_size']:>7}{r.get('estimated_flops')!s:>16}")
            print(
                f"\nMEASURED peak RSS {mon.peak_rss_bytes / 2**20:.0f} MiB (process high-water mark)"
                + (
                    f", peak device memory {mon.peak_device_bytes / 2**20:.0f} MiB"
                    if mon.peak_device_bytes is not None
                    else ""
                )
                + f"; swap used before/after: {swap.before}/{swap.after} B"
            )
            print("\n" + NOTE)

            run.add_section(
                "Measured latency (this device only)",
                markdown_table(
                    rows, ["impl", "batch_size", "dtype", "device", "median_ms", "p10_ms", "p90_ms", "iters", "params"]
                ),
            )
            run.add_section(
                "Stored bytes (exact accounting from parameter counts x storage width)",
                markdown_table(
                    [{"impl": k, "bytes": v, "fraction_of_dense": v / stored["dense"]} for k, v in stored.items()]
                ),
            )
            run.add_section(
                "Estimated multiply-add counts (formula; not measured)",
                markdown_table(
                    [{k: r[k] for k in ("impl", "batch_size", "estimated_flops")} for r in rows],
                ),
            )
            run.add_section("Caveat", NOTE)
            run.summary = (
                f"{in_f}x{out_f} dense vs rotor vs low-rank at {args.byte_ratio:.0%} of dense stored bytes on "
                f"{dev_name} ({args.dtype}); latencies measured, flop counts estimated."
            )
            if refuse_swap and (swap.growth or 0) > 0:
                raise SwapGrowthError(
                    f"swap usage grew by {swap.growth} bytes during the benchmark (before={swap.before}, "
                    f"peak={swap.peak}, after={swap.after}); timings are not trustworthy"
                )
    except SwapGrowthError as e:
        print(f"error: {e}. Failed run recorded at {run.path}. Free memory or pass --no-refuse-swap.", file=sys.stderr)
        return 3
    print(f"run record {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
