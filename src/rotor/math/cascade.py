"""Full-rank structured cascade ``W ~= D_0 R_1 D_1 R_2 ... R_k D_k`` with rectangular adapters.

Shapes follow ``torch.nn.Linear``: the dense weight is ``W`` of shape ``(out, in)`` and the layer
computes ``y = x @ W.T``. The cascade works in a square width ``n = max(in, out)``:

1. ``D_k`` scales the ``in`` input coordinates (stored length ``in``),
2. the input embeds into width ``n`` by zero-padding (``RectAdapter``),
3. ``R_k, D_{k-1}, ..., D_1, R_1`` act in width ``n`` (each ``R_i`` = ``stages_per_rotor``
   stages of disjoint Givens rotations from a deterministic ``Topology``),
4. the first ``out`` coordinates are kept, and ``D_0`` scales them (stored length ``out``).

No low-rank bottleneck is imposed: interior widths are ``n`` and the operator's rank is
limited only by ``min(in, out)``, the rank of ``W`` itself. An optional rank-``r`` residual
``U V`` (``LowRankResidual``) can be added in parallel; it is reported separately.

Parameter count (exact, see ``CascadeSpec.param_count``)::

    angles    = num_rotors * sum_s |pairs(n, s)|
    diagonals = in + out + (num_rotors - 1) * n
    residual  = r * (in + out)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from rotor.math.givens import build_stages
from rotor.math.lowrank import LowRankResidual
from rotor.math.topology import Topology

ADAPTER_KINDS = ("pad_truncate",)


@dataclass(frozen=True)
class RectAdapter:
    """Maps between ``in``/``out`` and the square working width ``n = max(in, out)``.

    ``pad_truncate``: input coordinates occupy ``[0, in)``, the rest are zero; output keeps
    ``[0, out)``. Deterministic, index-free, and the only kind implemented so far.
    """

    in_features: int
    out_features: int
    kind: str = "pad_truncate"

    def __post_init__(self) -> None:
        if self.kind not in ADAPTER_KINDS:
            raise ValueError(f"unknown adapter kind {self.kind!r}; expected one of {ADAPTER_KINDS}")

    @property
    def width(self) -> int:
        return max(self.in_features, self.out_features)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.width - self.in_features
        return x if pad == 0 else nn.functional.pad(x, (0, pad))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h if h.shape[-1] == self.out_features else h[..., : self.out_features]


@dataclass(frozen=True)
class CascadeSpec:
    """Complete, serializable structure of one rotor/scaler cascade (no parameter values)."""

    in_features: int
    out_features: int
    num_rotors: int
    stages_per_rotor: int
    topology: Topology = field(default_factory=Topology)
    adapter: str = "pad_truncate"
    residual_rank: int = 0

    def __post_init__(self) -> None:
        if self.num_rotors < 1:
            raise ValueError("num_rotors must be >= 1")
        if self.stages_per_rotor < 0:
            raise ValueError("stages_per_rotor must be >= 0")
        if self.residual_rank < 0:
            raise ValueError("residual_rank must be >= 0")

    @property
    def width(self) -> int:
        return max(self.in_features, self.out_features)

    def angles_per_rotor(self) -> int:
        return self.topology.angle_count(self.width, self.stages_per_rotor)

    def param_breakdown(self) -> dict[str, int]:
        return {
            "angles": self.num_rotors * self.angles_per_rotor(),
            "diagonals": self.in_features + self.out_features + (self.num_rotors - 1) * self.width,
            "residual": self.residual_rank * (self.in_features + self.out_features),
        }

    def param_count(self) -> int:
        return sum(self.param_breakdown().values())

    def dense_param_count(self) -> int:
        return self.in_features * self.out_features

    def flops_per_token(self) -> int:
        """Theoretical multiply+add count per input vector (estimate, not a measurement).

        A Givens pair costs 4 mul + 2 add; a diagonal entry 1 mul; residual ``2 r (in + out)``.
        """
        p = self.param_breakdown()
        return 6 * p["angles"] + p["diagonals"] + 2 * self.residual_rank * (self.in_features + self.out_features)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["topology"] = self.topology.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CascadeSpec:
        d = dict(d)
        d["topology"] = Topology.from_dict(d["topology"])
        return cls(**d)


class RotorCascade(nn.Module):
    """Learnable operator implementing ``CascadeSpec``. Acts on the last dim of its input.

    Parameters (all float32 masters by default; storage dtype is a manifest decision):

    - ``angles``: ``(num_rotors, angles_per_rotor)``; rotor ``i`` (1-based, ``R_i``) is row ``i-1``.
    - ``d_in``  (``D_k``, length ``in``), ``d_mid`` (``D_1..D_{k-1}``, ``(k-1, n)``),
      ``d_out`` (``D_0``, length ``out``).
    - ``residual``: optional ``LowRankResidual``.
    """

    def __init__(self, spec: CascadeSpec, dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        self.spec = spec
        n, k = spec.width, spec.num_rotors
        self.adapter = RectAdapter(spec.in_features, spec.out_features, spec.adapter)
        self._stages = build_stages(spec.topology, n, spec.stages_per_rotor, device)
        self._stage_slices: list[slice] = []
        off = 0
        for st in self._stages:
            self._stage_slices.append(slice(off, off + st.num_pairs))
            off += st.num_pairs
        self.grad_checkpoint = False  # recompute each rotor in backward: exact gradients, O(k) not O(k*stages) memory
        self.angles = nn.Parameter(torch.zeros(k, off, dtype=dtype, device=device))
        self.d_in = nn.Parameter(torch.ones(spec.in_features, dtype=dtype, device=device))
        self.d_mid = nn.Parameter(torch.ones(k - 1, n, dtype=dtype, device=device))
        self.d_out = nn.Parameter(torch.ones(spec.out_features, dtype=dtype, device=device))
        self.residual = (
            LowRankResidual(spec.in_features, spec.out_features, spec.residual_rank, dtype=dtype, device=device)
            if spec.residual_rank > 0
            else None
        )

    def _apply(self, fn, recurse=True):  # keep cached index buffers on the module's device
        out = super()._apply(fn, recurse)
        dev = self.angles.device
        for st in self._stages:
            st.a, st.b = st.a.to(dev), st.b.to(dev)
            st.partner, st.inv_perm = st.partner.to(dev), st.inv_perm.to(dev)
        return out

    def _rotor(self, h: torch.Tensor, i: int, transpose: bool = False) -> torch.Tensor:
        """Apply rotor ``R_{i+1}`` (row ``i``). ``R = S_1 S_2 ... S_m``: ``R h`` applies ``S_m`` first."""
        theta = self.angles[i]
        order = range(len(self._stages) - 1, -1, -1) if not transpose else range(len(self._stages))
        for s in order:
            h = self._stages[s].apply(h, theta[self._stage_slices[s]], transpose=transpose)
        return h

    def _rotor_rows(self, h: torch.Tensor, i: int) -> torch.Tensor:
        """:meth:`_rotor` on the transposed layout ``h: (n, B)`` (see ``GivensStage.apply_rows``)."""
        theta = self.angles[i]
        for s in range(len(self._stages) - 1, -1, -1):
            h = self._stages[s].apply_rows(h, theta[self._stage_slices[s]])
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(self.angles.dtype) * self.d_in
        h = self.adapter.embed(h)
        k = self.spec.num_rotors
        checkpoint = self.grad_checkpoint and torch.is_grad_enabled()
        for i in range(k - 1, -1, -1):  # R_k first, then D_{k-1}, ..., R_1
            if checkpoint:
                h = torch_checkpoint(self._rotor, h, i, use_reentrant=False)
            else:
                h = self._rotor(h, i)
            if i > 0:
                h = h * self.d_mid[i - 1]
        y = self.adapter.project(h) * self.d_out
        if self.residual is not None:
            y = y + self.residual(x.to(self.angles.dtype))
        return y.to(x.dtype)

    def dense_weight(self) -> torch.Tensor:
        """Materialize ``W_hat`` (``out x in``). Differentiable; used for weight-space fitting.

        Runs the cascade on ``H = [diag(d_in); 0]`` in the ``(n, in)`` row layout, so ``W_hat`` falls out without a
        transpose and every Givens stage is one contiguous row gather plus a fused multiply-add.
        """
        dev, dt = self.angles.device, self.angles.dtype
        n, k = self.spec.width, self.spec.num_rotors
        eye = torch.eye(self.spec.in_features, dtype=dt, device=dev)
        h = nn.functional.pad(eye * self.d_in, (0, 0, 0, n - self.spec.in_features))
        checkpoint = self.grad_checkpoint and torch.is_grad_enabled()
        for i in range(k - 1, -1, -1):
            h = torch_checkpoint(self._rotor_rows, h, i, use_reentrant=False) if checkpoint else self._rotor_rows(h, i)
            if i > 0:
                h = h * self.d_mid[i - 1].unsqueeze(1)
        w = h[: self.spec.out_features] * self.d_out.unsqueeze(1)
        if self.residual is not None:
            w = w + self.residual(eye).T
        return w

    @torch.no_grad()
    def init_from_dense(self, W: torch.Tensor, angle_std: float = 0.0, generator: torch.Generator | None = None):
        """Deterministic warm start: angles ~ N(0, angle_std), diagonals scaled to match ``||W||_F``.

        ``d_out``/``d_in``/``d_mid`` are set to a common magnitude ``a`` such that the (all-rotation)
        operator's Frobenius norm equals ``||W||_F`` (``a^(k+1) * sqrt(min(in,out)) = ||W||_F``).
        """
        if angle_std > 0:
            self.angles.normal_(0.0, angle_std, generator=generator)
        else:
            self.angles.zero_()
        k = self.spec.num_rotors
        fro = float(W.float().norm())
        r = min(self.spec.in_features, self.spec.out_features)
        a = (fro / max(r, 1) ** 0.5) ** (1.0 / (k + 1)) if fro > 0 else 1.0
        self.d_in.fill_(a)
        self.d_mid.fill_(a)
        self.d_out.fill_(a)
        if self.residual is not None:
            self.residual.reset_parameters(generator=generator)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {"angles": [self.angles], "scales": [self.d_in, self.d_mid, self.d_out]}
        if self.residual is not None:
            groups["residual"] = list(self.residual.parameters())
        return groups


class RotorLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement backed by a ``RotorCascade`` (+ optional dense bias)."""

    def __init__(self, cascade: RotorCascade, bias: torch.Tensor | None = None):
        super().__init__()
        self.cascade = cascade
        self.in_features = cascade.spec.in_features
        self.out_features = cascade.spec.out_features
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cascade(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self) -> str:
        s = self.cascade.spec
        return (
            f"in={s.in_features}, out={s.out_features}, rotors={s.num_rotors}, "
            f"stages/rotor={s.stages_per_rotor}, topology={s.topology.kind}, residual_rank={s.residual_rank}"
        )
