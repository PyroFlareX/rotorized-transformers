"""Algebraic conversion gates exercise real checkpoint tensors and atomic output behavior."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from rotor.converters.algebraic import convert_ga_tensors
from rotor.formats.ga import load_ga_checkpoint
from rotor.storage import StorageError


def test_algebraic_conversion_preserves_rectangular_linear_bias_and_source(tmp_storage):
    base = tmp_storage.base_path("ga-source")
    base.mkdir(parents=True)
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(9, 5, generator=generator)
    bias = torch.randn(9, generator=generator)
    save_file({"projection.weight": weight, "projection.bias": bias}, str(base / "model.safetensors"))
    source_bytes = (base / "model.safetensors").read_bytes()
    destination, report = convert_ga_tensors(
        tensor_names=["projection.weight"], name="ga-output", base_dir=base, probe_rows=7, chunk_size=3
    )
    manifest, layers = load_ga_checkpoint(destination)
    x = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    actual = layers["projection.weight"](x)
    expected = torch.nn.functional.linear(x, weight.double(), bias.double())
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert (base / "model.safetensors").read_bytes() == source_bytes
    metrics = json.loads((report / "metrics.json").read_text())["metrics"]
    assert metrics["export_load"]["projection.weight"]["direct_action"]["relative_l2"] < 1e-10
    compactness = metrics["tensors"]["projection.weight"]["compactness"]
    # Full rank: dense weight count + two rank-sized sign vectors, plus original bias.
    assert compactness["stored_scalars"] == weight.numel() + 2 * min(weight.shape) + bias.numel()
    assert compactness["source_scalars"] == weight.numel() + bias.numel()
    assert manifest["serialized"]["num_scalars"] == compactness["stored_scalars"]
    assert manifest["format"] == "ga-rotor-svd"
    with pytest.raises(StorageError, match="overwrite"):
        convert_ga_tensors(tensor_names=["projection.weight"], name="ga-output", base_dir=base)


def test_truncation_cannot_publish_an_unacceptable_approximation(tmp_storage):
    base = tmp_storage.base_path("ga-truncated-source")
    base.mkdir(parents=True)
    save_file({"projection.weight": torch.diag(torch.tensor([3.0, 2.0, 1.0]))}, str(base / "model.safetensors"))
    with pytest.raises(ValueError, match="relative error"):
        convert_ga_tensors(
            tensor_names=["projection.weight"], name="ga-rejected", base_dir=base, rank=1,
            max_relative_error=1e-3,
        )
    assert not tmp_storage.derived_path("ga-rejected").exists()


def test_explicit_truncation_reports_actual_error_instead_of_full_rank_claim(tmp_storage):
    base = tmp_storage.base_path("ga-approx-source")
    base.mkdir(parents=True)
    weight = torch.diag(torch.tensor([3.0, 0.01, 0.001]))
    save_file({"projection.weight": weight}, str(base / "model.safetensors"))
    destination, report = convert_ga_tensors(
        tensor_names=["projection.weight"], name="ga-approx", base_dir=base, rank=1,
        max_relative_error=0.02,
    )
    _, layers = load_ga_checkpoint(destination)
    expected = torch.diag(torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(layers["projection.weight"].reconstruct(), expected, atol=1e-10, rtol=1e-10)
    metrics = json.loads((report / "metrics.json").read_text())["metrics"]["tensors"]["projection.weight"]
    assert metrics["exact_full_rank"] is False
    assert metrics["weight"]["relative_l2"] == pytest.approx(
        float((weight.double() - expected).norm() / weight.double().norm()), rel=1e-10
    )
