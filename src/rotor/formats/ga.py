"""Versioned checkpoint I/O for algebraic ``ga-rotor-svd`` layers.

This format stores one elimination angle per plane rotation, QR signs, singular
values, and optional bias. It is independent of ``rotor-scaler`` and never
serializes dense SVD factors or reconstructed matrices.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rotor.math.ga import GARotorLinear, rotor_count
from rotor.storage import atomic_dir, sha256_file

FORMAT_NAME = "ga-rotor-svd"
FORMAT_VERSION = 2
MANIFEST_FILENAME = "ga_rotor_manifest.json"
WEIGHTS_FILENAME = "ga_rotor_weights.safetensors"
_SUPPORTED_DTYPES = {"float64": torch.float64, "float32": torch.float32}

_CONVENTIONS: dict[str, Any] = {
    "algebra": "Euclidean Cl(m,0), with E_ij=e_i e_j and E_ij^2=-1",
    "rotor_parameters": "one stored angle theta=atan2(lower,upper) per elimination rotation",
    "rotor_action": (
        "R=cos(theta/2)+sin(theta/2)*E_ij; R*x*R^-1 gives "
        "x_i'=c*x_i-s*x_j, x_j'=s*x_i+c*x_j, c=cos(theta), s=-sin(theta)"
    ),
    "factorization": "W=U*diag(singular_values)*V^T; U and V are thin orthonormal factors",
    "qr_elimination": "G*Q=E*diag(signs), hence Q=G^-1*E*diag(signs); each stored sign is +1 or -1",
    "rank_semantics": (
        "rank is the retained thin-SVD rank (1 <= rank <= min(out_features,in_features)); "
        "full rank is the default; truncation retains the leading rank singular values "
        "and matching signs/factor directions"
    ),
    "application_order": (
        "Apply right-factor V elimination waves in ascending order; truncate to rank, "
        "apply right_signs and singular_values; embed the result and apply left-factor U signs; "
        "apply inverse left-factor waves in descending order; then add optional bias"
    ),
    "canonical_wavefront_schedule": {
        "elimination": (
            "for each column j ascending, eliminate row i descending from rows-1 through j+1 "
            "using adjacent pair (i-1,i)"
        ),
        "wave": "(rows-1-i)+2*j",
        "within_wave": "columns j ascending",
        "packing": "rotation angles are packed in ascending wave number and then the stated within-wave order",
    },
}


def _safe_key(source_tensor: str) -> str:
    return "layer_" + hashlib.sha256(source_tensor.encode("utf-8")).hexdigest()


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(d) for d in tensor.shape]


def _require_json_object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dict")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from exc
    return value


def _validate_tensor(tensor: Any, name: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.dtype not in _SUPPORTED_DTYPES.values():
        raise TypeError(f"{name} must have float32 or float64 dtype, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _validate_layer(source_tensor: str, layer: GARotorLinear) -> dict[str, Any]:
    if not isinstance(source_tensor, str) or not source_tensor or "\x00" in source_tensor:
        raise ValueError("source tensor names must be non-empty strings without NUL characters")
    if not isinstance(layer, GARotorLinear):
        raise TypeError(f"layer {source_tensor!r} must be a GARotorLinear")
    if any(True for _ in layer.parameters()):
        raise ValueError(f"layer {source_tensor!r} must contain buffers only, not trainable parameters")
    dims = (layer.in_features, layer.out_features, layer.rank)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in dims):
        raise TypeError(f"layer {source_tensor!r} dimensions and rank must be integers")
    in_features, out_features, rank = dims
    if in_features < 1 or out_features < 1 or not 1 <= rank <= min(in_features, out_features):
        raise ValueError(f"layer {source_tensor!r} has invalid dimensions/rank {dims}")

    expected = {
        "left_angles": (rotor_count(out_features, rank),),
        "right_angles": (rotor_count(in_features, rank),),
        "left_signs": (rank,),
        "right_signs": (rank,),
        "singular_values": (rank,),
    }
    tensors: dict[str, torch.Tensor] = {}
    for field, shape in expected.items():
        tensors[field] = _validate_tensor(getattr(layer, field, None), f"{source_tensor}.{field}", shape)
    dtypes = {tensor.dtype for tensor in tensors.values()}
    if len(dtypes) != 1:
        raise TypeError(f"{source_tensor} tensors must all use the same floating dtype")
    bias = getattr(layer, "bias", None)
    if bias is not None:
        tensors["bias"] = _validate_tensor(bias, f"{source_tensor}.bias", (out_features,))
        if tensors["bias"].dtype not in dtypes:
            raise TypeError(f"{source_tensor}.bias must use the same floating dtype as its layer tensors")

    for field in ("left_signs", "right_signs"):
        signs = tensors[field]
        if not bool(((signs == 1) | (signs == -1)).all().item()):
            raise ValueError(f"{source_tensor}.{field} entries must be exactly +1 or -1")
    singular_values = tensors["singular_values"]
    if bool((singular_values < 0).any().item()):
        raise ValueError(f"{source_tensor}.singular_values must be nonnegative")
    if rank > 1 and bool((singular_values[:-1] < singular_values[1:]).any().item()):
        raise ValueError(f"{source_tensor}.singular_values must be in non-increasing order")
    return {
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "has_bias": bias is not None,
        "shapes": {name: _shape(tensor) for name, tensor in tensors.items()},
        "tensors": tensors,
    }


def _tensor_key(key: str, field: str) -> str:
    return f"{key}.{field}"


def _safetensors_header_bytes(path: Path) -> int:
    with path.open("rb") as stream:
        prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors file: missing header length")
    return struct.unpack("<Q", prefix)[0]


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_ga_checkpoint(
    out_dir: str | Path,
    layers: dict[str, GARotorLinear],
    *,
    provenance: dict,
    conversion_config: dict,
    storage,
    storage_dtype: str = "float64",
    metrics: dict | None = None,
    verify: Callable[[dict[str, GARotorLinear]], None] | None = None,
) -> None:
    """Atomically publish a new GA-SVD checkpoint; existing destinations are never replaced.

    When supplied, ``verify`` receives layers reloaded from the temporary serialized checkpoint.
    A verifier failure aborts publication and removes the temporary directory.
    """
    if not isinstance(layers, dict) or not layers:
        raise ValueError("layers must be a non-empty mapping of source tensor names to GARotorLinear modules")
    if any(not isinstance(name, str) for name in layers):
        raise TypeError("all source tensor names must be strings")
    if not callable(verify) and verify is not None:
        raise TypeError("verify must be a callable or None")
    dest = Path(storage.assert_writable(Path(out_dir)))
    if storage_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported storage dtype {storage_dtype!r}; expected float64 or float32")
    _require_json_object(provenance, "provenance")
    _require_json_object(conversion_config, "conversion_config")
    metrics = {} if metrics is None else _require_json_object(metrics, "metrics")
    entries: dict[str, dict[str, Any]] = {}
    payload_tensors: dict[str, torch.Tensor] = {}
    target_dtype = _SUPPORTED_DTYPES[storage_dtype]
    for source_tensor in sorted(layers):
        info = _validate_layer(source_tensor, layers[source_tensor])
        key = _safe_key(source_tensor)
        if key in (entry["key"] for entry in entries.values()):
            raise ValueError("source tensor names produced a duplicate safe key")
        entries[source_tensor] = {
            "key": key,
            "in_features": info["in_features"],
            "out_features": info["out_features"],
            "rank": info["rank"],
            "has_bias": info["has_bias"],
            "shapes": info["shapes"],
            "storage_dtype": storage_dtype,
        }
        for field, tensor in info["tensors"].items():
            # Conversion is explicit and all values are checked again after narrowing.
            stored = tensor.detach().to(device="cpu", dtype=target_dtype).contiguous()
            if not bool(torch.isfinite(stored).all().item()):
                raise ValueError(f"{source_tensor}.{field} is not finite in requested {storage_dtype} storage")
            if field in ("left_signs", "right_signs") and not bool(
                ((stored == 1) | (stored == -1)).all().item()
            ):
                raise ValueError(f"{source_tensor}.{field} has invalid signs after {storage_dtype} conversion")
            if field == "singular_values" and bool((stored < 0).any().item()):
                raise ValueError(f"{source_tensor}.singular_values is negative after {storage_dtype} conversion")
            payload_tensors[_tensor_key(key, field)] = stored

    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "conventions": _CONVENTIONS,
        "storage_dtype": storage_dtype,
        "provenance": provenance,
        "conversion_config": conversion_config,
        "metrics": metrics,
        "layers": entries,
        "serialized": {},
    }

    with atomic_dir(dest) as temp_dir:
        weights_path = temp_dir / WEIGHTS_FILENAME
        save_file(
            payload_tensors, str(weights_path), metadata={"format": FORMAT_NAME, "format_version": str(FORMAT_VERSION)}
        )
        file_bytes = weights_path.stat().st_size
        header_bytes = _safetensors_header_bytes(weights_path)
        tensor_bytes = sum(t.numel() * t.element_size() for t in payload_tensors.values())
        padding_bytes = file_bytes - 8 - header_bytes - tensor_bytes
        if padding_bytes < 0:
            raise RuntimeError("safetensors file byte accounting is inconsistent")
        manifest["serialized"] = {
            "weights_file_bytes": file_bytes,
            "safetensors_header_bytes": header_bytes,
            "tensor_payload_bytes": tensor_bytes,
            "padding_bytes": padding_bytes,
            "num_scalars": sum(t.numel() for t in payload_tensors.values()),
            "num_tensors": len(payload_tensors),
            "weights_sha256": sha256_file(weights_path),
            "manifest_bytes": 0,
            "manifest_overhead_bytes": 0,
            "checkpoint_bytes": 0,
        }
        encoded = b""
        for _ in range(16):
            encoded = _manifest_bytes(manifest)
            size = len(encoded)
            if manifest["serialized"]["manifest_bytes"] == size:
                break
            manifest["serialized"]["manifest_bytes"] = size
            manifest["serialized"]["manifest_overhead_bytes"] = size
            manifest["serialized"]["checkpoint_bytes"] = file_bytes + size
        else:
            raise RuntimeError("manifest byte accounting did not converge")
        encoded = _manifest_bytes(manifest)
        if len(encoded) != manifest["serialized"]["manifest_bytes"]:
            raise RuntimeError("manifest byte accounting is inconsistent")
        (temp_dir / MANIFEST_FILENAME).write_bytes(encoded)
        if verify is not None:
            _, reloaded = load_ga_checkpoint(temp_dir)
            verify(reloaded)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read GA checkpoint manifest in {path}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise ValueError(f"not a {FORMAT_NAME} manifest")
    version = data.get("format_version")
    if isinstance(version, bool) or version != FORMAT_VERSION:
        raise ValueError(f"unsupported {FORMAT_NAME} format_version {version!r}; supported version is {FORMAT_VERSION}")
    if data.get("conventions") != _CONVENTIONS:
        raise ValueError("GA checkpoint mathematical conventions are missing or unsupported")
    if data.get("storage_dtype") not in _SUPPORTED_DTYPES:
        raise ValueError("GA checkpoint has an unsupported storage dtype")
    _require_json_object(data.get("provenance"), "manifest provenance")
    _require_json_object(data.get("conversion_config"), "manifest conversion_config")
    _require_json_object(data.get("metrics"), "manifest metrics")
    if not isinstance(data.get("layers"), dict) or not data["layers"]:
        raise ValueError("GA checkpoint manifest must contain a non-empty layers mapping")
    if not isinstance(data.get("serialized"), dict):
        raise ValueError("GA checkpoint manifest has invalid serialized accounting")
    return data


def _validate_loaded_layer(
    source_tensor: str,
    entry: Any,
    key_names: set[str],
    tensors: dict[str, torch.Tensor],
    storage_dtype: str,
    device,
) -> GARotorLinear:
    if not isinstance(source_tensor, str) or not source_tensor or "\x00" in source_tensor:
        raise ValueError("manifest source tensor names must be non-empty strings without NUL characters")
    if not isinstance(entry, dict):
        raise ValueError(f"manifest layer entry for {source_tensor!r} must be an object")
    key = entry.get("key")
    if not isinstance(key, str) or key != _safe_key(source_tensor):
        raise ValueError(f"manifest layer {source_tensor!r} has an invalid or unstable safe key")
    dims = [entry.get("in_features"), entry.get("out_features"), entry.get("rank")]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in dims):
        raise ValueError(f"manifest layer {source_tensor!r} dimensions/rank must be integers")
    in_features, out_features, rank = dims
    if in_features < 1 or out_features < 1 or not 1 <= rank <= min(in_features, out_features):
        raise ValueError(f"manifest layer {source_tensor!r} has invalid dimensions/rank {dims}")
    has_bias = entry.get("has_bias")
    if not isinstance(has_bias, bool):
        raise ValueError(f"manifest layer {source_tensor!r} has invalid has_bias flag")
    if entry.get("storage_dtype") != storage_dtype:
        raise ValueError(f"manifest layer {source_tensor!r} storage dtype disagrees with checkpoint")
    expected_shapes = {
        "left_angles": [rotor_count(out_features, rank)],
        "right_angles": [rotor_count(in_features, rank)],
        "left_signs": [rank],
        "right_signs": [rank],
        "singular_values": [rank],
    }
    if has_bias:
        expected_shapes["bias"] = [out_features]
    if entry.get("shapes") != expected_shapes:
        raise ValueError(f"manifest layer {source_tensor!r} has inconsistent tensor shapes")
    expected_names = {_tensor_key(key, field) for field in expected_shapes}
    if not expected_names <= key_names:
        missing = sorted(expected_names - key_names)
        raise ValueError(f"checkpoint is missing tensors for {source_tensor!r}: {missing}")

    values: dict[str, torch.Tensor] = {}
    expected_dtype = _SUPPORTED_DTYPES[storage_dtype]
    for field, shape in expected_shapes.items():
        tensor_name = _tensor_key(key, field)
        tensor = tensors[tensor_name]
        if tensor.dtype != expected_dtype or list(tensor.shape) != shape:
            raise ValueError(
                f"tensor {tensor_name!r} has dtype/shape {tensor.dtype}/{list(tensor.shape)}, "
                f"expected {storage_dtype}/{shape}"
            )
        _validate_tensor(tensor, tensor_name, tuple(shape))
        values[field] = tensor.to(device=device)
    try:
        layer = GARotorLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            left_angles=values["left_angles"],
            right_angles=values["right_angles"],
            left_signs=values["left_signs"],
            right_signs=values["right_signs"],
            singular_values=values["singular_values"],
            bias=values.get("bias"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GA layer tensors for {source_tensor!r}: {exc}") from exc
    _validate_layer(source_tensor, layer)
    return layer


def load_ga_checkpoint(path: str | Path, device="cpu") -> tuple[dict[str, Any], dict[str, GARotorLinear]]:
    """Load and validate a ``ga-rotor-svd`` v2 checkpoint without materializing dense factors."""
    checkpoint = Path(path)
    manifest = _read_manifest(checkpoint)
    manifest_size = (checkpoint / MANIFEST_FILENAME).stat().st_size
    serialized = manifest["serialized"]
    if serialized.get("manifest_bytes") != manifest_size or serialized.get("manifest_overhead_bytes") != manifest_size:
        raise ValueError("GA checkpoint manifest byte accounting does not match the manifest file")
    weights_path = checkpoint / WEIGHTS_FILENAME
    try:
        weights_size = weights_path.stat().st_size
        if serialized.get("weights_file_bytes") != weights_size:
            raise ValueError("GA checkpoint weights byte accounting does not match the weights file")
        if serialized.get("weights_sha256") != sha256_file(weights_path):
            raise ValueError("GA checkpoint weights SHA-256 mismatch")
        header_bytes = _safetensors_header_bytes(weights_path)
        tensors = load_file(str(weights_path), device="cpu")
    except OSError as exc:
        raise ValueError(f"cannot read GA checkpoint weights in {checkpoint}") from exc
    payload_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    padding_bytes = weights_size - 8 - header_bytes - payload_bytes
    if (
        serialized.get("safetensors_header_bytes") != header_bytes
        or serialized.get("tensor_payload_bytes") != payload_bytes
        or serialized.get("num_scalars") != sum(t.numel() for t in tensors.values())
        or serialized.get("num_tensors") != len(tensors)
        or serialized.get("checkpoint_bytes") != weights_size + manifest_size
    ):
        raise ValueError("GA checkpoint file/header/payload accounting is inconsistent")
    if padding_bytes < 0:
        raise ValueError("GA checkpoint has impossible safetensors byte accounting")

    expected_names: set[str] = set()
    seen_keys: set[str] = set()
    layers: dict[str, GARotorLinear] = {}
    storage_dtype = manifest["storage_dtype"]
    for source_tensor, entry in manifest["layers"].items():
        if isinstance(entry, dict):
            key = entry.get("key")
            if isinstance(key, str):
                if key in seen_keys:
                    raise ValueError("manifest contains duplicate layer safe keys")
                seen_keys.add(key)
                expected_names.update(
                    _tensor_key(key, field)
                    for field in ("left_angles", "right_angles", "left_signs", "right_signs", "singular_values")
                )
                if entry.get("has_bias") is True:
                    expected_names.add(_tensor_key(key, "bias"))
        layers[source_tensor] = _validate_loaded_layer(
            source_tensor, entry, set(tensors), tensors, storage_dtype, device
        )
    actual_names = set(tensors)
    if actual_names != expected_names:
        missing, extra = sorted(expected_names - actual_names), sorted(actual_names - expected_names)
        raise ValueError(f"GA checkpoint tensor keys differ from manifest (missing={missing}, extra={extra})")
    try:
        metadata = None
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as sf:
            metadata = sf.metadata()
        if metadata != {"format": FORMAT_NAME, "format_version": str(FORMAT_VERSION)}:
            raise ValueError("GA checkpoint safetensors metadata does not match its format manifest")
    except (OSError, RuntimeError) as exc:
        raise ValueError("cannot validate GA checkpoint safetensors metadata") from exc
    return manifest, layers
