"""Algebraic tensor-to-GA conversion with measured error gates; no fitting or downloads."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from rotor.formats.ga import load_ga_checkpoint, write_ga_checkpoint
from rotor.math.ga import GARotorLinear, decompose_ga
from rotor.models.discovery import local_tensors
from rotor.reporting import ExperimentRun
from rotor.storage import StorageError, load_storage, model_entry, sha256_file


def _error(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    reference = reference.detach().cpu().double()
    actual = actual.detach().cpu().double()
    if reference.shape != actual.shape:
        raise ValueError(f"verification shape mismatch: {reference.shape} != {actual.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("verification encountered non-finite values")
    delta = actual - reference
    norm = float(reference.norm())
    residual = float(delta.norm())
    return {
        "relative_l2": residual / norm if norm else (0.0 if residual == 0 else math.inf),
        "max_absolute": float(delta.abs().max()),
        "reference_norm": norm,
        "residual_norm": residual,
    }


def verify_ga_tensor(
    weight: torch.Tensor,
    layer: GARotorLinear,
    *,
    bias: torch.Tensor | None = None,
    seed: int = 0,
    probe_rows: int = 16,
    chunk_size: int = 64,
    max_relative_error: float = 1e-10,
    exact: bool = True,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> dict:
    """Verify reconstructed coefficients and direct actions against the original, not a fitted proxy.

    Exact conversion requires both relative-norm and elementwise atol+rtol gates. Explicit rank truncation uses
    the caller's relative-norm error limit; all elementwise discrepancies are still recorded. Basis actions are
    covered by reconstruct(), whose chunks never exceed chunk_size input rows.
    """
    if probe_rows < 1 or chunk_size < 1:
        raise ValueError("probe_rows and chunk_size must be positive")
    reconstruction = layer.reconstruct(chunk_size=chunk_size).detach().cpu().double()
    source = weight.detach().cpu().double()
    generator = torch.Generator().manual_seed(seed)
    probes = torch.randn(probe_rows, source.shape[1], generator=generator, dtype=torch.float64)
    internal = layer.singular_values
    with torch.no_grad():
        actual = layer(probes.to(device=internal.device, dtype=internal.dtype)).cpu().double()
    expected = probes @ source.T
    if bias is not None:
        expected += bias.detach().cpu().double()
    metrics = {"weight": _error(source, reconstruction), "direct_action": _error(expected, actual)}
    for name, values in metrics.items():
        if values["relative_l2"] > max_relative_error:
            raise ValueError(
                f"{name} relative error {values['relative_l2']:.8g} exceeds required {max_relative_error:.8g}"
            )
    if exact:
        torch.testing.assert_close(reconstruction, source, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    metrics.update(
        {
            "passed": True,
            "exact_full_rank": exact,
            "max_relative_error": max_relative_error,
            "elementwise_gate": {"atol": atol, "rtol": rtol} if exact else None,
            "seed": seed,
            "probe_rows": probe_rows,
            "basis_chunk": chunk_size,
            "probes_sha256": hashlib.sha256(probes.numpy().tobytes()).hexdigest(),
        }
    )
    return metrics


def convert_ga_tensors(
    *,
    tensor_names: list[str],
    name: str,
    model: str | None = None,
    base_dir: Path | None = None,
    device: str = "cpu",
    storage_dtype: str = "float64",
    rank: int | None = None,
    max_relative_error: float | None = None,
    seed: int = 0,
    probe_rows: int = 16,
    chunk_size: int = 64,
) -> tuple[Path, Path]:
    """Convert explicitly selected local floating 2D tensors and refuse publication if any verification fails."""
    if storage_dtype not in {"float64", "float32"}:
        raise ValueError("GA storage_dtype must be float64 or float32")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a single new output directory name")
    if not tensor_names or len(set(tensor_names)) != len(tensor_names):
        raise ValueError("explicit distinct tensor names are required")
    if rank is not None and (isinstance(rank, bool) or rank < 1):
        raise ValueError("rank must be positive; omit it for full-rank algebraic conversion")
    tolerance = 1e-10 if storage_dtype == "float64" else 1e-5
    limit = tolerance if max_relative_error is None else max_relative_error
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("max_relative_error must be finite and positive")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested device {device} is unavailable; no CPU fallback")
    storage = load_storage()
    entry = None
    if model is not None:
        _, entry = model_entry(model)
        base = storage.base_path(base_dir or entry["local_path"])
    elif base_dir is not None:
        base = storage.base_path(base_dir)
    else:
        raise ValueError("model or base_dir is required")
    destination = storage.derived_path(name)
    if destination.exists():
        raise StorageError(f"refusing to overwrite {destination}")
    config = {
        "model": model,
        "base_dir": str(base),
        "tensor_names": tensor_names,
        "name": name,
        "device": device,
        "storage_dtype": storage_dtype,
        "rank": rank,
        "max_relative_error": limit,
        "seed": seed,
        "probe_rows": probe_rows,
        "chunk_size": chunk_size,
        "algorithm": "full-rank SVD and algebraic ordered GA plane-rotor elimination",
        "rotation_storage": "one elimination angle in radians per plane rotation (GA format v2)",
        "training": False,
    }
    with ExperimentRun(
        "conversion", model=model or base.name, method="algebraic-ga-rotors", target="unrestricted", config=config
    ) as run:
        infos = {info.name: info for info in local_tensors(base)}
        missing = set(tensor_names) - set(infos)
        if missing:
            raise ValueError(f"requested tensors not found locally: {sorted(missing)}")
        shards: set[str] = set()
        layers: dict[str, GARotorLinear] = {}
        verification = {}
        originals = {}
        for tensor_name in tensor_names:
            info = infos[tensor_name]
            if len(info.shape) != 2 or min(info.shape) < 1:
                raise ValueError(f"{tensor_name} is not a nonempty 2D tensor: {info.shape}")
            with safe_open(str(base / info.shard), framework="pt") as handle:
                weight = handle.get_tensor(tensor_name)
            if not weight.is_floating_point():
                raise ValueError(f"{tensor_name} has unsupported nonfloating dtype {weight.dtype}")
            bias = None
            bias_name = tensor_name.removesuffix(".weight") + ".bias"
            if tensor_name.endswith(".weight") and bias_name in infos:
                bias_info = infos[bias_name]
                with safe_open(str(base / bias_info.shard), framework="pt") as handle:
                    bias = handle.get_tensor(bias_name)
                shards.add(bias_info.shard)
            shards.add(info.shard)
            start = time.perf_counter()
            run.log(f"algebraically converting {tensor_name}, shape={info.shape}, device={device}; no training")
            layer = decompose_ga(weight, bias=bias, rank=rank, device=device)
            layer = layer.to(dtype=getattr(torch, storage_dtype))
            values = verify_ga_tensor(
                weight,
                layer,
                bias=bias,
                seed=seed,
                probe_rows=probe_rows,
                chunk_size=chunk_size,
                max_relative_error=limit,
                exact=layer.rank == min(info.shape),
                atol=tolerance,
                rtol=tolerance,
            )
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            values["conversion_and_verification_seconds"] = time.perf_counter() - start
            values["source_dtype"] = str(weight.dtype)
            values["source_tensor_sha256"] = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
            source_scalars = weight.numel() + (0 if bias is None else bias.numel())
            source_bytes = weight.numel() * weight.element_size()
            if bias is not None:
                source_bytes += bias.numel() * bias.element_size()
            buffers = list(layer.buffers())
            stored_scalars = sum(tensor.numel() for tensor in buffers)
            payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers)
            values["compactness"] = {
                "source_scalars": source_scalars,
                "stored_scalars": stored_scalars,
                "scalar_ratio": stored_scalars / source_scalars,
                "source_tensor_bytes": source_bytes,
                "stored_tensor_bytes": payload_bytes,
                "tensor_byte_ratio": payload_bytes / source_bytes,
            }
            verification[tensor_name] = values
            layers[tensor_name] = layer.cpu()
            originals[tensor_name] = (weight, bias)
            run.metrics["tensors"] = verification
            run.log(f"verified {tensor_name}: weight={values['weight']}, direct_action={values['direct_action']}")
        provenance = {
            "path": str(base),
            "hf_id": None if entry is None else entry.get("hf_id"),
            "revision": None if entry is None else entry.get("revision"),
            "file_sha256": {shard: sha256_file(base / shard) for shard in sorted(shards)},
        }
        if (base / "config.json").is_file():
            provenance["file_sha256"]["config.json"] = sha256_file(base / "config.json")
        roundtrip = {}

        def verify_export(loaded: dict[str, GARotorLinear]) -> None:
            for tensor_name, layer in loaded.items():
                weight, bias = originals[tensor_name]
                roundtrip[tensor_name] = verify_ga_tensor(
                    weight,
                    layer,
                    bias=bias,
                    seed=seed,
                    probe_rows=probe_rows,
                    chunk_size=chunk_size,
                    max_relative_error=limit,
                    exact=layer.rank == min(weight.shape),
                    atol=tolerance,
                    rtol=tolerance,
                )

        write_ga_checkpoint(
            destination,
            layers,
            provenance=provenance,
            conversion_config=config,
            storage=storage,
            storage_dtype=storage_dtype,
            metrics=verification,
            verify=verify_export,
        )
        manifest, _ = load_ga_checkpoint(destination)
        run.metrics["export_load"] = roundtrip
        run.metrics["checkpoint"] = str(destination)
        run.metrics["serialized"] = manifest.get("serialized", {})
        run.metrics["provenance"] = provenance
        run.summary = (
            f"Algebraically converted and verified {len(layers)} explicitly selected tensors to GA rotors; "
            "no training, dense runtime fallback, or byte-budget objective."
        )
        run.interpretation = (
            "Full-rank algebraic correctness is separate from compression or performance. "
            "Explicit rank truncation, if requested, is deterministic SVD approximation with measured error."
        )
    return destination, run.path
