"""Algebraic thin-SVD factorization with ordered Euclidean GA rotors.

A Givens plane rotation stores its elimination angle ``theta``. For
``E_ij = e_i e_j`` and ``E_ij**2 = -1``, the corresponding unit rotor is
``cos(theta/2) + sin(theta/2) E_ij`` and its sandwich action is

    c = cos(theta)
    s = -sin(theta)
    (x_i, x_j) -> (c*x_i - s*x_j, s*x_i + c*x_j).

The QR eliminations use adjacent row pairs. The wave key
``(rows - 1 - i) + 2*j`` schedules the elimination of ``(column=j,row=i)``.
Within one wave, increasing ``j`` gives disjoint row pairs. For a fixed
column, bottom-up dependencies advance one wave; a rotation in column ``j``
precedes the same-pair elimination in column ``j+1`` by two waves.

This representation is deliberately separate from the fixed-butterfly
rotor-scaler format. The module stores only ordered elimination angles,
subspace signs, singular values, and an optional bias; dense SVD factors exist
only temporarily during decomposition or in the explicitly diagnostic
``reconstruct`` method's output.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from operator import index as _index

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class RotorWave:
    """One disjoint wave of adjacent-row eliminations.

    Cells are ``(column, lower_row, packed_offset)`` in increasing column
    order. Angle values are packed by increasing ``stage`` and then increasing
    column within the wave.
    """

    stage: int
    offset: int
    first_column: int
    stop_column: int

    @property
    def pair_count(self) -> int:
        return self.stop_column - self.first_column

    def cells(self, rows: int) -> Iterator[tuple[int, int, int]]:
        """Yield ``(j, i, offset)`` for row pair ``(i-1, i)`` in this wave."""
        for j in range(self.first_column, self.stop_column):
            i = rows - 1 + 2 * j - self.stage
            yield j, i, self.offset + j - self.first_column


def _dimension(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer dimension")
    try:
        result = _index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer dimension") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def rotor_count(rows: int, rank: int) -> int:
    """Number of adjacent-row elimination angles for an ``rows x rank`` factor."""
    rows = _dimension(rows, "rows")
    if isinstance(rank, bool):
        raise ValueError("rank must be an integer in [0, rows]")
    try:
        rank = _index(rank)
    except TypeError as exc:
        raise ValueError("rank must be an integer in [0, rows]") from exc
    if rank < 0 or rank > rows:
        raise ValueError("rank must be in [0, rows]")
    return rows * rank - rank * (rank + 1) // 2


def wave_schedule(rows: int, rank: int) -> tuple[RotorWave, ...]:
    """Return compact deterministic wave descriptors and packed angle offsets.

    The schedule uses ``stage = (rows - 1 - i) + 2*j`` for elimination cell
    ``(j, i)`` and adjacent row pair ``(i-1, i)``. A wave has a contiguous
    range of columns, so the descriptor count is O(rows + rank), not O(rotor
    count). ``RotorWave.cells(rows)`` expands a descriptor if needed.
    """
    count = rotor_count(rows, rank)
    if rank == 0:
        return ()
    waves: list[RotorWave] = []
    offset = 0
    for stage in range(rows + rank - 2):
        first = max(0, stage - rows + 2)
        stop = min(rank, stage // 2 + 1)
        if first < stop:
            wave = RotorWave(stage, offset, first, stop)
            waves.append(wave)
            offset += wave.pair_count
    if offset != count:  # protects the packed layout if the schedule changes
        raise RuntimeError(f"invalid GA wave schedule: packed {offset} rotors, expected {count}")
    return tuple(waves)


def _angles_from_elimination(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Record the elimination angle that zeroes ``b`` below ``a``."""
    angle = torch.atan2(b, a)
    # Make the identity case explicit rather than relying on backend-specific
    # atan2(0, 0) behavior.
    return torch.where((a == 0) & (b == 0), torch.zeros_like(angle), angle)


def _eliminate_orthonormal_columns(Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed elimination angles and diagonal signs for an ``m x r`` factor.

    ``Q`` is transient working storage.  Rotations within each wave update
    disjoint adjacent row pairs in one tensor batch, rather than executing one
    Python operation per rotor.
    """
    rows, rank = Q.shape
    waves = wave_schedule(rows, rank)
    angles = torch.empty((rotor_count(rows, rank),), dtype=Q.dtype, device=Q.device)
    A = Q.clone()
    for wave in waves:
        columns = torch.arange(wave.first_column, wave.stop_column, device=Q.device)
        lower = rows - 1 + 2 * columns - wave.stage
        upper = lower - 1
        a = A[upper, columns]
        b = A[lower, columns]
        wave_angles = _angles_from_elimination(a, b)
        angles[wave.offset : wave.offset + wave.pair_count] = wave_angles

        c = torch.cos(wave_angles)
        s = -torch.sin(wave_angles)
        old_upper = A.index_select(0, upper).clone()
        old_lower = A.index_select(0, lower).clone()
        A.index_copy_(0, upper, c[:, None] * old_upper - s[:, None] * old_lower)
        A.index_copy_(0, lower, s[:, None] * old_upper + c[:, None] * old_lower)

    signs = torch.sign(torch.diagonal(A[:rank, :rank]))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return angles, signs



class GARotorLinear(nn.Module):
    """Frozen linear map ``x -> x @ W.T + bias`` represented by an ordered GA SVD.

    Angle tensors have shape ``(rotor_count(rows, rank),)`` and store the
    elimination angle for each rotor.  The module has buffers only;
    no dense left/right factor is retained.  Leading input dimensions are
    preserved, and the result is cast back to the input's floating dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        left_angles: torch.Tensor,
        right_angles: torch.Tensor,
        left_signs: torch.Tensor,
        right_signs: torch.Tensor,
        singular_values: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        in_features = _dimension(in_features, "in_features")
        out_features = _dimension(out_features, "out_features")
        if isinstance(rank, bool):
            raise ValueError("rank must be an integer in [0, min(in_features, out_features)]")
        try:
            rank = _index(rank)
        except TypeError as exc:
            raise ValueError("rank must be an integer in [0, min(in_features, out_features)]") from exc
        if rank < 0 or rank > min(in_features, out_features):
            raise ValueError("rank must be in [0, min(in_features, out_features)]")

        expected_shapes = {
            "left_angles": (rotor_count(out_features, rank),),
            "right_angles": (rotor_count(in_features, rank),),
            "left_signs": (rank,),
            "right_signs": (rank,),
            "singular_values": (rank,),
        }
        tensors = {
            "left_angles": left_angles,
            "right_angles": right_angles,
            "left_signs": left_signs,
            "right_signs": right_signs,
            "singular_values": singular_values,
        }
        reference_dtype = left_angles.dtype
        reference_device = left_angles.device
        if reference_dtype not in (torch.float32, torch.float64):
            raise TypeError("GA rotor angles must use float32 or float64")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shapes[name]}")
            if tensor.dtype != reference_dtype or tensor.device != reference_device:
                raise ValueError("all GA factor tensors must share one dtype and device")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
                raise TypeError("bias must be a floating-point tensor or None")
            if tuple(bias.shape) != (out_features,):
                raise ValueError(f"bias has shape {tuple(bias.shape)}, expected {(out_features,)}")
            if bias.dtype != reference_dtype or bias.device != reference_device:
                raise ValueError("bias must share the GA factor dtype and device")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self._left_waves = wave_schedule(out_features, rank)
        self._right_waves = wave_schedule(in_features, rank)
        self.register_buffer("left_angles", left_angles.detach().clone().contiguous())
        self.register_buffer("right_angles", right_angles.detach().clone().contiguous())
        self.register_buffer("left_signs", left_signs.detach().clone().contiguous())
        self.register_buffer("right_signs", right_signs.detach().clone().contiguous())
        self.register_buffer("singular_values", singular_values.detach().clone().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().clone().contiguous())

    @property
    def coefficient_dtype(self) -> torch.dtype:
        return self.left_angles.dtype

    def _rotate_wave(
        self,
        h: torch.Tensor,
        angles: torch.Tensor,
        wave: RotorWave,
        rows: int,
        *,
        inverse: bool,
    ) -> None:
        """Apply one batch of disjoint adjacent-coordinate rotor actions in place."""
        columns = torch.arange(wave.first_column, wave.stop_column, device=h.device)
        lower = rows - 1 + 2 * columns - wave.stage
        upper = lower - 1
        angle = angles.narrow(0, wave.offset, wave.pair_count)
        c = torch.cos(angle)
        s = -torch.sin(angle)
        shape = (1,) * (h.ndim - 1) + (wave.pair_count,)
        c = c.view(shape)
        s = s.view(shape)
        old_upper = h.index_select(-1, upper)
        old_lower = h.index_select(-1, lower)
        if inverse:
            new_upper = c * old_upper + s * old_lower
            new_lower = -s * old_upper + c * old_lower
        else:
            new_upper = c * old_upper - s * old_lower
            new_lower = s * old_upper + c * old_lower
        h.index_copy_(-1, upper, new_upper)
        h.index_copy_(-1, lower, new_lower)

    def _weight_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the represented weight without its bias."""
        h = x.to(dtype=self.coefficient_dtype).clone()
        for wave in self._right_waves:
            self._rotate_wave(h, self.right_angles, wave, self.in_features, inverse=False)
        h = h[..., : self.rank] * self.right_signs * self.singular_values

        out_shape = (*x.shape[:-1], self.out_features)
        result = h.new_zeros(out_shape)
        if self.rank:
            result[..., : self.rank] = h * self.left_signs
        for wave in reversed(self._left_waves):
            self._rotate_wave(result, self.left_angles, wave, self.out_features, inverse=True)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("GARotorLinear input must be a floating-point tensor")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"input's final dimension must be {self.in_features}")
        if x.device != self.left_angles.device:
            raise ValueError(f"input is on {x.device}, but GA factors are on {self.left_angles.device}")
        y = self._weight_forward(x)
        if self.bias is not None:
            y = y + self.bias
        return y.to(dtype=x.dtype)

    @torch.no_grad()
    def reconstruct(self, chunk_size: int = 64) -> torch.Tensor:
        """Materialize the weight by applying bounded blocks of input basis vectors.

        Returns an ``(out_features, in_features)`` tensor in the coefficient
        dtype.  Bias is intentionally excluded.  Only the requested dense
        output is materialized; no dense SVD factor or full-size identity is
        constructed.
        """
        if isinstance(chunk_size, bool):
            raise ValueError("chunk_size must be a positive integer")
        try:
            chunk_size = _index(chunk_size)
        except TypeError as exc:
            raise ValueError("chunk_size must be a positive integer") from exc
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        weight = torch.empty(
            (self.out_features, self.in_features),
            dtype=self.coefficient_dtype,
            device=self.left_angles.device,
        )
        for start in range(0, self.in_features, chunk_size):
            stop = min(start + chunk_size, self.in_features)
            columns = torch.arange(start, stop, device=weight.device)
            basis = torch.zeros((stop - start, self.in_features), dtype=weight.dtype, device=weight.device)
            basis[torch.arange(stop - start, device=weight.device), columns] = 1
            weight[:, start:stop] = self._weight_forward(basis).T
        return weight

    @classmethod
    def decompose(
        cls,
        W: torch.Tensor,
        bias: torch.Tensor | None = None,
        rank: int | None = None,
        device: torch.device | str = "cpu",
    ) -> GARotorLinear:
        """Class-method form of :func:`decompose_ga`."""
        return decompose_ga(W, bias=bias, rank=rank, device=device)


def decompose_ga(
    W: torch.Tensor,
    bias: torch.Tensor | None = None,
    rank: int | None = None,
    device: torch.device | str = "cpu",
) -> GARotorLinear:
    """Algebraically factor a real linear weight into ordered GA SVD rotors.

    ``W`` follows ``torch.nn.Linear`` orientation ``(out_features,in_features)``.
    The thin SVD and QR elimination run in float64 on ``device``; ``rank=None``
    retains the full thin rank, while a smaller rank gives the truncated SVD.
    No training or iterative fitting is performed.
    """
    if not isinstance(W, torch.Tensor) or W.ndim != 2 or not W.is_floating_point():
        raise TypeError("W must be a two-dimensional floating-point tensor")
    out_features, in_features = W.shape
    if out_features < 1 or in_features < 1:
        raise ValueError("W dimensions must be positive")
    max_rank = min(out_features, in_features)
    if rank is None:
        rank = max_rank
    elif isinstance(rank, bool):
        raise ValueError("rank must be an integer in [0, min(W.shape)]")
    else:
        try:
            rank = _index(rank)
        except TypeError as exc:
            raise ValueError("rank must be an integer in [0, min(W.shape)]") from exc
    if rank < 0 or rank > max_rank:
        raise ValueError("rank must be in [0, min(W.shape)]")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
            raise TypeError("bias must be a floating-point tensor or None")
        if tuple(bias.shape) != (out_features,):
            raise ValueError(f"bias has shape {tuple(bias.shape)}, expected {(out_features,)}")

    target_device = torch.device(device)
    work = W.detach().to(device=target_device, dtype=torch.float64)
    angle_count_left = rotor_count(out_features, rank)
    angle_count_right = rotor_count(in_features, rank)
    if rank == 0:
        left_angles = work.new_empty((angle_count_left,))
        right_angles = work.new_empty((angle_count_right,))
        singular_values = work.new_empty((0,))
        left_signs = right_signs = work.new_empty((0,))
    elif bool(torch.count_nonzero(work) == 0):
        # The zero map has a canonical identity-factor representation; avoid
        # inheriting arbitrary left/right singular vectors from a zero SVD.
        left_angles = work.new_zeros((angle_count_left,))
        right_angles = work.new_zeros((angle_count_right,))
        left_signs = work.new_ones((rank,))
        right_signs = work.new_ones((rank,))
        singular_values = work.new_zeros((rank,))
    else:
        U, singular_values_all, Vh = torch.linalg.svd(work, full_matrices=False)
        U = U[:, :rank].contiguous()
        V = Vh[:rank, :].T.contiguous()
        singular_values = singular_values_all[:rank].contiguous()
        left_angles, left_signs = _eliminate_orthonormal_columns(U)
        right_angles, right_signs = _eliminate_orthonormal_columns(V)

    internal_bias = None if bias is None else bias.detach().to(device=target_device, dtype=torch.float64)
    return GARotorLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        left_angles=left_angles,
        right_angles=right_angles,
        left_signs=left_signs,
        right_signs=right_signs,
        singular_values=singular_values,
        bias=internal_bias,
    )
