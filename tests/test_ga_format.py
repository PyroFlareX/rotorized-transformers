"""Tests for the independent ``ga-rotor-svd`` v2 serialization format."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from rotor.formats.ga import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    load_ga_checkpoint,
    write_ga_checkpoint,
)
from rotor.math.ga import GARotorLinear, decompose_ga
from rotor.storage import StorageError, sha256_file


def _layer(seed: int = 14, *, rows: int = 5, columns: int = 3, rank: int | None = None, bias: bool = True):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    offset = torch.randn(rows, generator=generator, dtype=torch.float64) if bias else None
    return decompose_ga(weight, bias=offset, rank=rank, device="cpu"), weight, offset


def _write(tmp_storage, path: Path, dtype: str = "float64", *, layer: GARotorLinear | None = None):
    if layer is None:
        layer = _layer()[0]
    write_ga_checkpoint(
        path,
        {"model.layers.0.mlp.proj.weight": layer},
        provenance={"path": "/local/source", "revision": "local"},
        conversion_config={"method": "algebraic-ga-svd"},
        storage=tmp_storage,
        storage_dtype=dtype,
        metrics={"relative_l2": 0.0},
    )
    return layer


def _rewrite_weights_and_accounting(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write deliberately malformed tensor keys while keeping all byte accounting self-consistent."""
    weights_path = path / WEIGHTS_FILENAME
    save_file(tensors, str(weights_path), metadata={"format": FORMAT_NAME, "format_version": str(FORMAT_VERSION)})
    manifest_path = path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    serialized = manifest["serialized"]
    payload_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    data = weights_path.read_bytes()
    header_bytes = struct.unpack("<Q", data[:8])[0]
    size = weights_path.stat().st_size
    serialized.update(
        weights_file_bytes=size,
        safetensors_header_bytes=header_bytes,
        tensor_payload_bytes=payload_bytes,
        padding_bytes=size - 8 - header_bytes - payload_bytes,
        num_scalars=sum(t.numel() for t in tensors.values()),
        num_tensors=len(tensors),
        weights_sha256=sha256_file(weights_path),
    )
    serialized["checkpoint_bytes"] = size + serialized["manifest_bytes"]
    for _ in range(16):
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if serialized["manifest_bytes"] == len(encoded):
            break
        serialized["manifest_bytes"] = len(encoded)
        serialized["manifest_overhead_bytes"] = len(encoded)
        serialized["checkpoint_bytes"] = size + len(encoded)
    else:
        raise AssertionError("manifest accounting did not converge")
    manifest_path.write_bytes(encoded)


def test_roundtrip_preserves_mapping_direct_action_and_exact_accounting(tmp_storage) -> None:
    layer, weight, bias = _layer(rows=5, columns=3)
    source = "model.layers.0.mlp.proj.weight"
    destination = tmp_storage.derived_path("ga-roundtrip")
    _write(tmp_storage, destination, layer=layer)

    manifest, loaded = load_ga_checkpoint(destination)
    assert manifest["format"] == FORMAT_NAME and manifest["format_version"] == FORMAT_VERSION
    assert manifest["provenance"]["path"] == "/local/source"
    assert manifest["layers"][source]["rank"] == min(weight.shape)
    assert loaded[source].left_angles.dtype == torch.float64
    assert loaded[source].right_angles.dtype == torch.float64
    assert not list(loaded[source].parameters())

    generator = torch.Generator().manual_seed(99)
    x = torch.randn(2, 4, weight.shape[1], generator=generator, dtype=torch.float64)
    expected = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(loaded[source](x), expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(loaded[source].reconstruct(chunk_size=2), weight, atol=2e-12, rtol=2e-12)

    serialized = manifest["serialized"]
    weights_path = destination / WEIGHTS_FILENAME
    assert serialized["weights_file_bytes"] == weights_path.stat().st_size
    assert serialized["safetensors_header_bytes"] == struct.unpack("<Q", weights_path.read_bytes()[:8])[0]
    assert serialized["manifest_bytes"] == (destination / MANIFEST_FILENAME).stat().st_size
    assert serialized["manifest_overhead_bytes"] == serialized["manifest_bytes"]
    assert serialized["checkpoint_bytes"] == serialized["weights_file_bytes"] + serialized["manifest_bytes"]
    tensors = load_file(str(weights_path))
    assert serialized["num_tensors"] == len(tensors)
    assert serialized["num_scalars"] == sum(t.numel() for t in tensors.values())
    assert serialized["tensor_payload_bytes"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert serialized["weights_sha256"] == hashlib.sha256(weights_path.read_bytes()).hexdigest()
    layer_key = manifest["layers"][source]["key"]
    left_angles = tensors[f"{layer_key}.left_angles"]
    right_angles = tensors[f"{layer_key}.right_angles"]
    assert left_angles.ndim == right_angles.ndim == 1
    expected_left_angles = weight.shape[0] * layer.rank - layer.rank * (layer.rank + 1) // 2
    expected_right_angles = weight.shape[1] * layer.rank - layer.rank * (layer.rank + 1) // 2
    assert right_angles.numel() == expected_right_angles
    angle_scalars = left_angles.numel() + right_angles.numel()
    previous_rotor_coefficient_scalars = 2 * (expected_left_angles + expected_right_angles)
    assert angle_scalars * 2 == previous_rotor_coefficient_scalars


def test_float32_storage_is_explicit_and_executable(tmp_storage) -> None:
    layer, weight, bias = _layer(rows=4, columns=3)
    destination = tmp_storage.derived_path("ga-float32")
    _write(tmp_storage, destination, "float32", layer=layer)
    manifest, loaded = load_ga_checkpoint(destination)
    restored = loaded["model.layers.0.mlp.proj.weight"]
    assert restored.left_angles.dtype == torch.float32
    x = torch.randn(7, weight.shape[1], dtype=torch.float64)
    expected = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(restored(x), expected, atol=2e-5, rtol=2e-5)


def test_zero_weight_roundtrips_as_nonnegative_singular_values(tmp_storage) -> None:
    weight = torch.zeros((3, 2), dtype=torch.float64)
    layer = decompose_ga(weight, device="cpu")
    destination = tmp_storage.derived_path("ga-zero-weight")
    _write(tmp_storage, destination, layer=layer)
    manifest, loaded = load_ga_checkpoint(destination)
    restored = loaded["model.layers.0.mlp.proj.weight"]
    assert manifest["layers"]["model.layers.0.mlp.proj.weight"]["rank"] == 2
    torch.testing.assert_close(restored.reconstruct(), weight)


def test_truncated_rank_roundtrip_records_retained_rank(tmp_storage) -> None:
    layer, weight, _ = _layer(rows=5, columns=4, rank=2, bias=False)
    destination = tmp_storage.derived_path("ga-truncated")
    _write(tmp_storage, destination, layer=layer)
    manifest, loaded = load_ga_checkpoint(destination)
    restored = loaded["model.layers.0.mlp.proj.weight"]
    assert restored.rank == 2
    assert manifest["layers"]["model.layers.0.mlp.proj.weight"]["rank"] == 2
    assert float((weight - restored.reconstruct()).norm()) > 0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("left_angles", torch.zeros((1,), dtype=torch.float64), "shape"),
        ("left_angles", torch.full((9,), float("nan"), dtype=torch.float64), "non-finite"),
        ("right_signs", torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64), "exactly"),
        ("singular_values", torch.tensor([float("nan")] * 3, dtype=torch.float64), "non-finite"),
    ],
)
def test_writer_rejects_malformed_layer_tensors_before_publication(tmp_storage, field, replacement, message) -> None:
    layer, _, _ = _layer()
    setattr(layer, field, replacement)
    destination = tmp_storage.derived_path("ga-invalid")
    with pytest.raises(ValueError, match=message):
        _write(tmp_storage, destination, layer=layer)
    assert not destination.exists()


def test_writer_rejects_invalid_storage_dtype_but_accepts_zero_angles(tmp_storage) -> None:
    layer, _, _ = _layer()
    destination = tmp_storage.derived_path("ga-bad-dtype")
    with pytest.raises(ValueError, match="unsupported storage dtype"):
        _write(tmp_storage, destination, "float16", layer=layer)
    assert not destination.exists()

    layer.left_angles.zero_()
    layer.right_angles.zero_()
    zero_angle_destination = tmp_storage.derived_path("ga-zero-angles")
    _write(tmp_storage, zero_angle_destination, "float32", layer=layer)
    _, loaded = load_ga_checkpoint(zero_angle_destination)
    restored = loaded["model.layers.0.mlp.proj.weight"]
    assert torch.count_nonzero(restored.left_angles).item() == 0
    assert torch.count_nonzero(restored.right_angles).item() == 0


def test_loader_rejects_missing_and_extraneous_tensors(tmp_storage) -> None:
    destination = tmp_storage.derived_path("ga-bad-keys")
    _write(tmp_storage, destination)
    tensors = load_file(str(destination / WEIGHTS_FILENAME))
    missing_name = next(name for name in tensors if name.endswith(".bias"))
    _rewrite_weights_and_accounting(destination, {key: value for key, value in tensors.items() if key != missing_name})
    with pytest.raises(ValueError, match="missing tensors"):
        load_ga_checkpoint(destination)

    _write(tmp_storage, tmp_storage.derived_path("ga-extra-keys"))
    extra_destination = tmp_storage.derived_path("ga-extra-keys")
    extra_tensors = load_file(str(extra_destination / WEIGHTS_FILENAME))
    extra_tensors["unclaimed.tensor"] = torch.ones(1, dtype=torch.float64)
    _rewrite_weights_and_accounting(extra_destination, extra_tensors)
    with pytest.raises(ValueError, match="tensor keys differ"):
        load_ga_checkpoint(extra_destination)


def test_loader_rejects_nonfinite_angles(tmp_storage) -> None:
    destination = tmp_storage.derived_path("ga-nonfinite-angle")
    _write(tmp_storage, destination)
    tensors = load_file(str(destination / WEIGHTS_FILENAME))
    angle_name = next(name for name in tensors if name.endswith(".left_angles"))
    tensors[angle_name][0] = float("inf")
    _rewrite_weights_and_accounting(destination, tensors)
    with pytest.raises(ValueError, match="non-finite"):
        load_ga_checkpoint(destination)


def test_loader_rejects_v1_checkpoint_without_compatibility_path(tmp_storage) -> None:
    destination = tmp_storage.derived_path("ga-v1-refused")
    _write(tmp_storage, destination)
    manifest_path = destination / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    manifest_path.write_bytes(encoded)

    with pytest.raises(ValueError, match="unsupported ga-rotor-svd format_version 1"):
        load_ga_checkpoint(destination)


def test_storage_guard_no_overwrite_and_failed_verification_are_atomic(tmp_storage) -> None:
    destination = tmp_storage.derived_path("ga-protected")
    layer, _, _ = _layer()
    _write(tmp_storage, destination, layer=layer)
    original = (destination / WEIGHTS_FILENAME).read_bytes()
    with pytest.raises(StorageError, match="overwrite"):
        _write(tmp_storage, destination, layer=layer)
    assert (destination / WEIGHTS_FILENAME).read_bytes() == original

    with pytest.raises(StorageError, match="read-only"):
        _write(tmp_storage, tmp_storage.base_root / "not-writable", layer=layer)

    failed = tmp_storage.derived_path("ga-verification-failed")
    with pytest.raises(RuntimeError, match="verification failed"):
        write_ga_checkpoint(
            failed,
            {"projection.weight": layer},
            provenance={},
            conversion_config={},
            storage=tmp_storage,
            verify=lambda _: (_ for _ in ()).throw(RuntimeError("verification failed")),
        )
    assert not failed.exists()
    assert not list(failed.parent.glob(".ga-verification-failed.tmp-*"))
