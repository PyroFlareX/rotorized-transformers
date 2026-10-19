"""Single-kernel Triton inference for frozen rotor/scaler cascades.

One program processes one activation row at ``next_power_of_2(width)``. Compact topology maps
supply arbitrary stage partners and signs, while rotor-dependent trigonometric coefficients are
precomputed once. No dense rotation or weight matrices are constructed.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import nn

from rotor.math.cascade import RotorCascade, RotorLinear


@triton.jit
def _fused_cascade_kernel(
    X,
    D_IN,
    D_MID,
    D_OUT,
    COS_ANGLES,
    SIN_ANGLES,
    PARTNERS,
    ANGLE_CODES,
    Y,
    X_STRIDE_COL: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    INPUT_LEADING_DIMS: tl.constexpr,
    INPUT_LEADING_STRIDES: tl.constexpr,
    INPUT_ROW_DIVISORS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_ROTORS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    ANGLES_PER_ROTOR: tl.constexpr,
):
    row = tl.program_id(0)
    coord = tl.arange(0, BLOCK)
    row_offset = 0
    for dim in tl.static_range(len(INPUT_LEADING_DIMS)):
        index = (row // INPUT_ROW_DIVISORS[dim]) % INPUT_LEADING_DIMS[dim]
        row_offset += index * INPUT_LEADING_STRIDES[dim]

    x = tl.load(
        X + row_offset + coord * X_STRIDE_COL,
        mask=coord < IN_FEATURES,
        other=0.0,
    ).to(tl.float32)
    d_in = tl.load(D_IN + coord, mask=coord < IN_FEATURES, other=1.0)
    h = x * d_in

    for rotor_offset in range(NUM_ROTORS):
        rotor = NUM_ROTORS - 1 - rotor_offset
        for stage_offset in tl.static_range(NUM_STAGES):
            stage = NUM_STAGES - 1 - stage_offset
            stage_coord = stage * BLOCK + coord
            partner = tl.load(PARTNERS + stage_coord, mask=coord < WIDTH, other=coord).to(tl.int32)
            angle_code = tl.load(ANGLE_CODES + stage_coord, mask=coord < WIDTH, other=0).to(tl.int32)
            angle_mask = (coord < WIDTH) & (angle_code != 0)
            angle_id = tl.maximum(tl.abs(angle_code) - 1, 0)
            angle_ptr = rotor * ANGLES_PER_ROTOR + angle_id
            c = tl.load(COS_ANGLES + angle_ptr, mask=angle_mask, other=1.0)
            s = tl.load(SIN_ANGLES + angle_ptr, mask=angle_mask, other=0.0)
            partner_h = tl.gather(h, partner, axis=0)
            # Preserve the reference expression order per coordinate; FMA remains disabled.
            rotated_a = c * h - s * partner_h
            rotated_b = s * partner_h + c * h
            rotated = tl.where(angle_code < 0, rotated_a, rotated_b)
            h = tl.where(angle_mask, rotated, h)

        if rotor > 0:
            d_mid = tl.load(D_MID + (rotor - 1) * WIDTH + coord, mask=coord < WIDTH, other=1.0)
            h = h * d_mid

    d_out = tl.load(D_OUT + coord, mask=coord < OUT_FEATURES, other=1.0)
    y = h * d_out
    tl.store(Y + row * OUT_FEATURES + coord, y, mask=coord < OUT_FEATURES)


class FusedRotorCascade(nn.Module):
    """Inference-only fused implementation retaining the original cascade ``spec``."""

    def __init__(self, cascade: RotorCascade):
        super().__init__()
        self.spec = cascade.spec
        self.in_features = cascade.spec.in_features
        self.out_features = cascade.spec.out_features
        self.width = cascade.spec.width
        self.num_rotors = cascade.spec.num_rotors
        self.num_stages = cascade.spec.stages_per_rotor
        self.angles_per_rotor = cascade.angles.shape[1]
        self.block = triton.next_power_of_2(self.width)

        # These detached views retain the original FP32 storage without copying cascade weights.
        self.register_buffer("angles", cascade.angles.detach())
        self.register_buffer("d_in", cascade.d_in.detach())
        self.register_buffer("d_mid", cascade.d_mid.detach())
        self.register_buffer("d_out", cascade.d_out.detach())
        self.residual = cascade.residual

        device = cascade.angles.device
        # Partner and signed angle-code ranges are independent; size each packed map safely.
        partner_max = self.block - 1
        angle_code_max = self.angles_per_rotor
        partner_dtype = torch.int16 if partner_max <= 32767 else torch.int32
        angle_dtype = torch.int16 if angle_code_max <= 32767 else torch.int32
        partners = torch.arange(self.block, dtype=partner_dtype).expand(self.num_stages, self.block).clone()
        angle_codes = torch.zeros((self.num_stages, self.block), dtype=angle_dtype)
        angle_offset = 0
        for stage_index in range(self.num_stages):
            pairs = cascade.spec.topology.pairs(self.width, stage_index)
            for pair_index, (a, b) in enumerate(pairs):
                angle_id = angle_offset + pair_index
                partners[stage_index, a] = b
                partners[stage_index, b] = a
                angle_codes[stage_index, a] = -(angle_id + 1)
                angle_codes[stage_index, b] = angle_id + 1
            angle_offset += len(pairs)
        if angle_offset != self.angles_per_rotor:
            raise ValueError(
                f"cascade angle count {self.angles_per_rotor} does not match topology count {angle_offset}"
            )

        self.register_buffer("partners", partners.to(device=device))
        self.register_buffer("angle_codes", angle_codes.to(device=device))
        # Trig is evaluated once in the original FP32 master-angle dtype.
        self.register_buffer("cos_angles", torch.cos(cascade.angles.detach()))
        self.register_buffer("sin_angles", torch.sin(cascade.angles.detach()))


    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input with last dimension {self.in_features}; got shape {tuple(x.shape)}"
            )
        if x.device.type != "cuda":
            raise RuntimeError("fused rotor inference requires a CUDA/HIP tensor")
        if x.device != self.angles.device:
            raise RuntimeError(f"input is on {x.device}, but fused cascade parameters are on {self.angles.device}")

        rows = x.numel() // self.in_features
        output_shape = (*x.shape[:-1], self.out_features)
        y = torch.empty((rows, self.out_features), dtype=torch.float32, device=x.device)
        if rows:
            _fused_cascade_kernel[(rows,)](
                x,
                self.d_in,
                self.d_mid,
                self.d_out,
                self.cos_angles,
                self.sin_angles,
                self.partners,
                self.angle_codes,
                y,
                x.stride(-1),
                self.in_features,
                self.out_features,
                tuple(x.shape[:-1]),
                tuple(x.stride()[:-1]),
                tuple(math.prod(x.shape[dim + 1 : -1]) for dim in range(x.ndim - 1)),
                self.width,
                self.block,
                self.num_rotors,
                self.num_stages,
                self.angles_per_rotor,
                enable_fp_fusion=False,
            )

        output = y.reshape(output_shape)
        if self.residual is not None:
            output = output + self.residual(x.to(self.angles.dtype))
        return output.to(x.dtype)


def _check_cascade(cascade: RotorCascade) -> None:
    device = cascade.angles.device
    if device.type != "cuda":
        raise RuntimeError("fused rotor inference requires cascades placed on a CUDA/HIP device")
    if cascade.spec.width < 1 or cascade.spec.in_features < 1 or cascade.spec.out_features < 1:
        raise ValueError("fused rotor inference requires positive input, output, and working widths")
    for name, parameter in cascade.named_parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(f"fused rotor inference requires FP32 cascade parameters; {name} is {parameter.dtype}")
        if parameter.device != device:
            raise RuntimeError(f"cascade parameter {name} is on {parameter.device}, expected {device}")


def enable_fused_rotors(model: nn.Module) -> int:
    """Replace all reference ``RotorLinear`` cascades with fused GPU execution.

    All candidates are validated and prepared before model changes are committed. The replacement
    is inference-only; cascade parameters are detached/frozen and the forward path runs without
    autograd. Returns the number of replaced cascades.
    """
    candidates = [
        module
        for module in list(model.modules())
        if isinstance(module, RotorLinear) and isinstance(module.cascade, RotorCascade)
    ]
    if not candidates:
        return 0

    for layer in candidates:
        _check_cascade(layer.cascade)
    replacements = [(layer, FusedRotorCascade(layer.cascade)) for layer in candidates]

    for layer, runtime in replacements:
        if layer.bias is not None:
            layer.bias.requires_grad_(False)
        if runtime.residual is not None:
            for parameter in runtime.residual.parameters():
                parameter.requires_grad_(False)
        layer.cascade = runtime
    return len(replacements)
