"""Byte-budget allocation: choose the largest cascade whose STORED bytes fit a fraction of the dense bytes.

Budgets are expressed in stored bytes (``storage_dtype``), never in parameter counts, so a structured layer
and a truncated-SVD baseline can be compared at an identical byte budget.
"""

from __future__ import annotations

from rotor.formats.spec import bytes_per_param
from rotor.math.cascade import CascadeSpec
from rotor.math.topology import Topology
from rotor.models.qwen3_5 import Projection

# Checkpoint (safetensors header) dtype -> bytes per dense element.
_DENSE_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


def auto_stages(kind: str, width: int) -> int:
    """Stages per rotor for ``stages_per_rotor: auto``: ``max(1, ceil(log2(width)))`` for every topology kind."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    return max(1, (width - 1).bit_length())


def dense_bytes(p: Projection) -> int:
    """Bytes of the dense weight in the checkpoint dtype."""
    try:
        per = _DENSE_DTYPE_BYTES[p.dtype]
    except KeyError as e:
        raise ValueError(
            f"unsupported dense dtype {p.dtype!r} for {p.tensor}; expected {sorted(_DENSE_DTYPE_BYTES)}"
        ) from e
    return p.in_features * p.out_features * per


def structured_bytes(spec: CascadeSpec, storage_dtype: str) -> int:
    """Exact stored bytes of the cascade parameters (angles, diagonals, residual) in ``storage_dtype``."""
    return spec.param_count() * bytes_per_param(storage_dtype)


def spec_for_budget(
    in_features: int,
    out_features: int,
    param_budget: int,
    topology: Topology,
    stages_per_rotor: int | str,
    residual_rank: int = 0,
    adapter: str = "pad_truncate",
) -> CascadeSpec:
    """Largest ``num_rotors >= 1`` with ``param_count() <= param_budget``.

    ``param_count`` is affine in ``num_rotors``, so the answer is computed from the 1- and 2-rotor counts and
    then verified against the spec itself.
    """
    width = max(in_features, out_features)
    stages = auto_stages(topology.kind, width) if stages_per_rotor == "auto" else int(stages_per_rotor)

    def make(k: int) -> CascadeSpec:
        return CascadeSpec(
            in_features=in_features,
            out_features=out_features,
            num_rotors=k,
            stages_per_rotor=stages,
            topology=topology,
            adapter=adapter,
            residual_rank=residual_rank,
        )

    one = make(1)
    p1 = one.param_count()
    if p1 > param_budget:
        raise ValueError(
            f"a single rotor needs {p1} params but the budget is {param_budget} params "
            f"({in_features}x{out_features}, stages_per_rotor={stages}, residual_rank={residual_rank})"
        )
    step = make(2).param_count() - p1
    k = 1 + (param_budget - p1) // step if step > 0 else 1
    spec = make(k)
    while spec.param_count() > param_budget and k > 1:  # defensive: affine assumption violated
        k -= 1
        spec = make(k)
    while make(k + 1).param_count() <= param_budget:
        k += 1
        spec = make(k)
    return spec


def allocate(projections: list[Projection], policy: dict) -> dict[str, CascadeSpec]:
    """Per-projection cascade specs keyed by module path.

    ``byte_budget = floor(byte_ratio * dense_bytes)`` (dense bytes in the checkpoint dtype) and
    ``param_budget = byte_budget // bytes_per_param(storage_dtype)``.
    """
    byte_ratio = float(policy["byte_ratio"])
    if byte_ratio <= 0:
        raise ValueError(f"byte_ratio must be > 0, got {byte_ratio}")
    storage_dtype = policy.get("storage_dtype", "bfloat16")
    bpp = bytes_per_param(storage_dtype)
    topo = policy.get("topology", {"kind": "butterfly", "seed": 0})
    topology = topo if isinstance(topo, Topology) else Topology.from_dict(topo)
    stages = policy.get("stages_per_rotor", "auto")
    residual_rank = int(policy.get("residual_rank", 0))
    adapter = policy.get("adapter", "pad_truncate")

    specs: dict[str, CascadeSpec] = {}
    for p in projections:
        byte_budget = int(byte_ratio * dense_bytes(p))  # floor for positive values
        try:
            specs[p.module] = spec_for_budget(
                p.in_features, p.out_features, byte_budget // bpp, topology, stages, residual_rank, adapter
            )
        except ValueError as e:
            raise ValueError(f"{p.module}: byte_ratio={byte_ratio} is too small: {e}") from e
    return specs
