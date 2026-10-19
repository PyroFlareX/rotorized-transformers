"""Optional inference runtimes for rotor cascades.

Importing this package does not import Triton. Call :func:`enable_fused_rotors` explicitly to
replace ``RotorLinear`` cascades with the opt-in fused GPU implementation.
"""

from __future__ import annotations

from torch import nn


def enable_fused_rotors(model: nn.Module) -> int:
    """Replace every ``RotorLinear.cascade`` in ``model`` with fused inference execution.

    Triton is an optional dependency and is imported only when this function is called. The
    reference implementation remains unchanged unless the caller opts in here.

    Returns the number of cascades replaced. Fused execution requires frozen float32 cascade
    parameters on a supported CUDA/HIP device; unsupported devices fail explicitly.
    """
    try:
        from .fused import enable_fused_rotors as _enable_fused_rotors
    except ModuleNotFoundError as exc:
        if exc.name == "triton":
            raise RuntimeError("fused rotor inference requires the optional Triton package") from exc
        raise

    return _enable_fused_rotors(model)


__all__ = ["enable_fused_rotors"]
