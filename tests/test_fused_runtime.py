"""Fused inference preserves the reference operator, including rectangular shapes."""
from __future__ import annotations

import copy

import pytest
import torch

from rotor.math.cascade import CascadeSpec, RotorCascade, RotorLinear
from rotor.math.topology import Topology


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm and Triton")
@pytest.mark.parametrize("topology", ["brick", "butterfly", "seeded"])
@pytest.mark.parametrize("shape", [(7, 11), (16, 7)])
def test_fused_matches_reference(topology, shape):
    pytest.importorskip("triton")
    from rotor.runtime import enable_fused_rotors

    torch.manual_seed(3)
    reference = RotorLinear(RotorCascade(CascadeSpec(*shape, 3, 4, Topology(topology), residual_rank=2),
                                        device="cuda"), torch.randn(shape[1], device="cuda"))
    with torch.no_grad():
        reference.cascade.angles.normal_(0, 0.2)
        reference.cascade.d_mid.uniform_(0.8, 1.2)
    fused = copy.deepcopy(reference)
    enable_fused_rotors(fused)
    inputs = torch.randn(2, 3, shape[0], device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(fused(inputs), reference(inputs), rtol=1e-6, atol=1e-6)
