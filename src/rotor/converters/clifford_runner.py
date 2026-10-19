"""No-training, blockwise Clifford projection pilot on one local checkpoint tensor.

The runner loads one exact safetensors tensor, evaluates the Clifford projection
on seeded Gaussian probes, and publishes metrics only. It never downloads weights
or writes a converted checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from datetime import timedelta
from typing import Any

import torch
from safetensors import safe_open

from rotor.evaluation.baselines import equal_budget_baselines
from rotor.math.clifford import project_clifford
from rotor.math.topology import Topology
from rotor.models.discovery import local_tensors
from rotor.reporting import ExperimentRun, make_run_id, markdown_table
from rotor.storage import Storage, load_storage, model_entry, sha256_file
from rotor.training.layerwise import FitSchedule

DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.up_proj.weight"
PROBE_SEED = 42
CONTROL_SEED = 43
_GROUPING_SEEDS = (None, 0, 1, 2)
_STRICT_ACTION_TOLERANCE = 1e-10


def _unique_run_id(run: ExperimentRun) -> None:
    """Avoid same-second report collisions without overwriting existing runs."""
    while (run.out_root / run.run_id).exists():
        run.created += timedelta(seconds=1)
        run.run_id = make_run_id(run.model, run.method, run.target, run.created)


def _fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _finite_norm(value: torch.Tensor, label: str) -> float:
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm).item():
        raise ValueError(f"{label} has a non-finite Frobenius norm")
    return float(norm)


def _frobenius_metrics(reference: torch.Tensor, approximation: torch.Tensor) -> dict[str, float]:
    if approximation.shape != reference.shape:
        raise ValueError(
            f"approximation shape {tuple(approximation.shape)} does not match reference {tuple(reference.shape)}"
        )
    if not torch.isfinite(approximation).all().item():
        raise ValueError("approximation contains non-finite values")
    reference_norm = _finite_norm(reference, "reference")
    if reference_norm == 0:
        raise ValueError("reference has zero Frobenius norm; relative error is undefined")
    error = _finite_norm(reference - approximation, "approximation error")
    return {"absolute_frobenius_error": error, "relative_frobenius_error": error / reference_norm}


def _assert_operator_reconstruction(
    operator_action: torch.Tensor, probes: torch.Tensor, reconstructed: torch.Tensor, label: str
) -> float:
    reconstructed_action = probes @ reconstructed.T
    if not torch.isfinite(operator_action).all().item() or not torch.isfinite(reconstructed_action).all().item():
        raise ValueError(f"{label} produced non-finite probe actions")
    max_abs = float((operator_action - reconstructed_action).abs().max())
    if not math.isfinite(max_abs) or max_abs > _STRICT_ACTION_TOLERANCE:
        raise AssertionError(
            f"{label} operator/reconstruction action mismatch: max_abs={max_abs:.17g}, "
            f"tolerance={_STRICT_ACTION_TOLERANCE:g}"
        )
    return max_abs


def _gaussian_control(weight: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(weight.shape, dtype=torch.float64, generator=generator)
    sample_norm = _finite_norm(sample, "Gaussian control sample")
    weight_norm = _finite_norm(weight, "source weight")
    if sample_norm == 0 or weight_norm == 0:
        raise ValueError("cannot norm-match a zero-norm source or Gaussian control")
    return sample * (weight_norm / sample_norm)


def _svd_action(
    probes: torch.Tensor,
    left: torch.Tensor,
    singular_values: torch.Tensor,
    right_transpose: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank == 0:
        return torch.zeros((probes.shape[0], left.shape[0]), dtype=torch.float64)
    projected = (probes @ right_transpose[:rank].T) * singular_values[:rank]
    return projected @ left[:, :rank].T


def _svd_rows(
    weight: torch.Tensor,
    probes: torch.Tensor,
    direct_action: torch.Tensor,
    left: torch.Tensor,
    singular_values: torch.Tensor,
    right_transpose: torch.Tensor,
    coefficient_counts: dict[int, int],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    out_features, in_features = weight.shape
    scalars_per_rank = out_features + in_features + 1
    total_squared = singular_values.square().sum()
    if not torch.isfinite(total_squared).item() or float(total_squared) <= 0:
        raise ValueError("source tensor has zero or non-finite spectral energy")
    for q, coefficient_count in coefficient_counts.items():
        rank = min(min(out_features, in_features), coefficient_count // scalars_per_rank)
        action = _svd_action(probes, left, singular_values, right_transpose, rank)
        retained_squared = singular_values[:rank].square().sum()
        residual_squared = singular_values[rank:].square().sum()
        if not torch.isfinite(retained_squared).item() or not torch.isfinite(residual_squared).item():
            raise ValueError(f"non-finite SVD metrics for q={q}")
        scalar_count = rank * scalars_per_rank
        rows[str(q)] = {
            "q": q,
            "rank": rank,
            "stored_scalar_count": scalar_count,
            "stored_bytes_fp64": scalar_count * 8,
            "coefficient_scalar_budget": coefficient_count,
            "coefficient_bytes_fp64": coefficient_count * 8,
            "unused_scalar_budget": coefficient_count - scalar_count,
            "unused_coefficient_budget_fp64_bytes": (coefficient_count - scalar_count) * 8,
            "relative_frobenius_error": math.sqrt(float(residual_squared / total_squared)),
            "retained_energy_fraction": float(retained_squared / total_squared),
            "probe_action_vs_original": _frobenius_metrics(direct_action, action),
            "interpretation": (
                "truncated SVD at the Clifford coefficient-count budget; one full SVD is reused across q values"
            ),
        }
    return rows


def _run_small_block_baselines(weight: torch.Tensor, q: int) -> dict[str, Any]:
    block = weight[:q, :q].contiguous()
    schedule = FitSchedule.from_dict(
        {
            "phases": [
                {"train": ["scales"], "steps": 2},
                {"train": ["angles", "scales", "residual"], "steps": 2},
            ],
            "lrs": {"angles": 1e-2, "scales": 1e-2, "residual": 1e-3},
        }
    )
    result = equal_budget_baselines(
        block,
        byte_budget=q * 4,
        storage_dtype="float32",
        spec_kwargs={"topology": Topology("butterfly", 0), "stages_per_rotor": "auto", "residual_rank": 0},
        schedule=schedule,
        X=None,
        seed=0,
        device="cpu",
    )
    result["scope"] = {
        "matrix": "top-left contiguous q-by-q block of the original tensor",
        "shape": [q, q],
        "storage_dtype": "float32",
        "byte_budget": q * 4,
        "budget_equivalence": (
            "same scalar-count budget as q FP64 Clifford coefficients; bounded small-block comparator subset, "
            "not an equal-FP64-byte comparison"
        ),
        "fit_schedule": {
            "phases": [[['scales'], 2], [['angles', 'scales', 'residual'], 2]],
            "total_steps": 4,
        },
        "seed": 0,
        "included_method_rows": [
            "dense", "svd", "svd_givens", "magnitude_sparse", "butterfly", "rotor_weight", "rotor_residual"
        ],
        "included_random_control_rows": ["spectrum_matched_random", "gaussian_random"],
        "notes": "All skipped/infeasible method and random-control rows from equal_budget_baselines are retained.",
    }
    return result


def _load_tensor(model: str, tensor_name: str, storage: Storage) -> tuple[torch.Tensor, dict[str, Any]]:
    _, entry = model_entry(model)
    base = storage.base_path(entry["local_path"])
    if not base.is_dir():
        raise FileNotFoundError(
            f"base checkpoint for {entry['hf_id']} (revision {entry['revision']}) not found at {base}; "
            "no download or fallback is performed"
        )
    infos = {info.name: info for info in local_tensors(base)}
    if tensor_name not in infos:
        raise KeyError(f"tensor {tensor_name!r} was not found in local safetensors files under {base}")
    info = infos[tensor_name]
    shard_path = base / info.shard
    with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(tensor_name).contiguous().to(dtype=torch.float64)
    provenance = {
        "model_key": model,
        "hf_id": entry.get("hf_id"),
        "revision": entry.get("revision"),
        "base_path": str(base.resolve()),
        "tensor": tensor_name,
        "shape": list(info.shape),
        "checkpoint_dtype": info.dtype,
        "shard": info.shard,
        "shard_sha256": sha256_file(shard_path),
        "tensor_values_fp64_sha256": _fingerprint(weight),
    }
    return weight, provenance


def run_clifford_experiment(
    model: str = DEFAULT_MODEL,
    tensor_name: str = DEFAULT_TENSOR,
    probe_rows: int = 64,
    *,
    storage: Storage | None = None,
    run: ExperimentRun | None = None,
) -> dict[str, Any]:
    """Run the no-training Clifford pilot and return report-ready measured metrics."""
    if isinstance(probe_rows, bool) or not isinstance(probe_rows, int) or probe_rows < 1:
        raise ValueError("probe_rows must be an integer >= 1")
    storage = storage or load_storage()
    weight, source = _load_tensor(model, tensor_name, storage)
    if run is not None:
        run.config["resolved_source"] = source
        run.config["matrix_shape"] = list(weight.shape)
        run.config["source_coefficient_dtype"] = "float64"
        run.config["probe_distribution"] = "iid standard Gaussian, float64"
        run.config["operator_reconstruction_action_tolerance"] = _STRICT_ACTION_TOLERANCE
    if weight.ndim != 2:
        raise ValueError(f"requested tensor must be a matrix, got shape {tuple(weight.shape)}")
    if not torch.isfinite(weight).all().item():
        raise ValueError("requested tensor contains non-finite values")
    out_features, in_features = weight.shape
    if _finite_norm(weight, "source weight") == 0:
        raise ValueError("requested tensor has zero Frobenius norm")
    for q in (4, 8):
        if out_features % q or in_features % q:
            raise ValueError(
                f"tensor shape {(out_features, in_features)} is not divisible by q={q}; both d=2 and d=3 are required"
            )

    probe_generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    probes = torch.randn((probe_rows, in_features), dtype=torch.float64, generator=probe_generator)
    probe_fingerprint = _fingerprint(probes)
    probe_info = {
        "rows": probe_rows,
        "input_features": in_features,
        "dtype": "float64",
        "distribution": "iid standard Gaussian",
        "seed": PROBE_SEED,
        "sha256": probe_fingerprint,
        "shared_by_all_projection_arms_svd_rows_and_controls": True,
    }
    direct_action = probes @ weight.T
    control = _gaussian_control(weight, CONTROL_SEED)
    control_direct_action = probes @ control.T

    source_numel = weight.numel()
    source_dtype_name = source["checkpoint_dtype"]
    source_element_bytes = (
        2 if source_dtype_name in {"BF16", "F16", "bfloat16", "float16"}
        else 4 if source_dtype_name in {"F32", "float32"}
        else None
    )
    if source_element_bytes is None:
        raise ValueError(f"unsupported checkpoint dtype {source_dtype_name!r} for byte comparison")

    arms: dict[str, dict[str, Any]] = {}
    random_control_arms: dict[str, dict[str, Any]] = {}
    coefficient_counts: dict[int, int] = {}
    for dimensions in (2, 3):
        q = 1 << dimensions
        coefficient_counts[q] = (out_features // q) * (in_features // q) * q
        for seed in _GROUPING_SEEDS:
            grouping = "contiguous" if seed is None else "seeded_permutation"
            label = f"d{dimensions}/{grouping}" if seed is None else f"d{dimensions}/{grouping}-{seed}"
            module = project_clifford(weight, dimensions, seed=seed)
            approximation = module.reconstruct()
            weight_metrics = _frobenius_metrics(weight, approximation)
            operator_action = module(probes)
            mismatch = _assert_operator_reconstruction(operator_action, probes, approximation, label)
            coefficient_count = module.coefficients.numel()
            retained_energy = float(approximation.square().sum() / weight.square().sum())
            if not math.isfinite(retained_energy):
                raise ValueError(f"{label} produced non-finite retained energy")
            arms[label] = {
                "dimensions": dimensions,
                "q": q,
                "grouping": grouping,
                "seed": seed,
                "shape": [out_features, in_features],
                "coefficient_shape": list(module.coefficients.shape),
                "coefficient_scalar_count": coefficient_count,
                "coefficient_dtype": str(module.coefficients.dtype),
                "coefficient_bytes_fp64": coefficient_count * 8,
                "permutation_index_bytes": sum(
                    t.numel() * t.element_size() for t in (module.input_permutation, module.output_permutation)
                ),
                "coefficients_plus_permutation_bytes": coefficient_count * 8 + sum(
                    t.numel() * t.element_size() for t in (module.input_permutation, module.output_permutation)
                ),
                "source_tensor_bytes_checkpoint_dtype": source_numel * source_element_bytes,
                "source_tensor_bytes_bf16": source_numel * 2,
                "equal_fp64_dense_bytes": source_numel * 8,
                "coefficient_bytes_over_source_bf16": coefficient_count * 8 / (source_numel * 2),
                "coefficient_bytes_over_equal_fp64_dense": coefficient_count * 8 / (source_numel * 8),
                "weight_approximation": weight_metrics,
                "retained_energy_fraction": retained_energy,
                "probe_fingerprint": probe_fingerprint,
                "probe_action_vs_original": _frobenius_metrics(direct_action, operator_action),
                "operator_vs_reconstructed_probe_action_max_abs_error": mismatch,
                "operator_vs_reconstructed_probe_action_tolerance": _STRICT_ACTION_TOLERANCE,
                "operator_reconstruction_assertion_passed": True,
            }

            control_module = project_clifford(control, dimensions, seed=seed)
            control_approximation = control_module.reconstruct()
            control_weight_metrics = _frobenius_metrics(control, control_approximation)
            control_action = control_module(probes)
            control_mismatch = _assert_operator_reconstruction(
                control_action, probes, control_approximation, f"{label}/gaussian_control"
            )
            control_retained_energy = float(control_approximation.square().sum() / control.square().sum())
            if not math.isfinite(control_retained_energy):
                raise ValueError(f"{label}/gaussian_control produced non-finite retained energy")
            random_control_arms[label] = {
                "dimensions": dimensions,
                "q": q,
                "grouping": grouping,
                "seed": seed,
                "control_seed": CONTROL_SEED,
                "control_matrix_frobenius_norm": _finite_norm(control, "Gaussian control"),
                "control_matrix_sha256": _fingerprint(control),
                "weight_approximation_vs_control": control_weight_metrics,
                "retained_energy_fraction": control_retained_energy,
                "probe_fingerprint": probe_fingerprint,
                "probe_action_vs_control_original": _frobenius_metrics(control_direct_action, control_action),
                "operator_vs_reconstructed_probe_action_max_abs_error": control_mismatch,
                "operator_vs_reconstructed_probe_action_tolerance": _STRICT_ACTION_TOLERANCE,
                "operator_reconstruction_assertion_passed": True,
            }
            del module, approximation, control_module, control_approximation

    # Compute one FP64 SVD; evaluate both q-specific ranks from that single decomposition.
    left, singular_values, right_transpose = torch.linalg.svd(weight, full_matrices=False)
    svd_rows = _svd_rows(
        weight, probes, direct_action, left, singular_values, right_transpose, coefficient_counts
    )
    small_block_baselines = {str(q): _run_small_block_baselines(weight, q) for q in (4, 8)}

    result: dict[str, Any] = {
        "scope": {
            "experiment": "no-training blockwise Clifford left-geometric-product projection pilot",
            "checkpoint_export": False,
            "weights_downloaded": False,
            "full_model_quality_or_performance_claim": False,
            "other_trained_baseline_comparison": (
                "broad tournament not run; only the separately identified four-step "
                "q-by-q small-block subset is included"
            ),
            "winner_claim": "none",
            "note": (
                "No model-quality or performance claim is made. The bounded small-block equal-budget subset is "
                "reported separately for context only; no broad trained-baseline tournament was run."
            ),
        },
        "source": source,
        "probes": probe_info,
        "gaussian_control": {
            "seed": CONTROL_SEED,
            "distribution": "iid standard Gaussian scaled to the source tensor's Frobenius norm",
            "shape": list(control.shape),
            "dtype": "float64",
            "sha256": _fingerprint(control),
            "frobenius_norm": _finite_norm(control, "Gaussian control"),
            "probe_action_fingerprint": _fingerprint(control_direct_action),
        },
        "clifford_arms": arms,
        "gaussian_control_arms": random_control_arms,
        "equal_coefficient_budget_svd": {
            "per_q": svd_rows,
            "coefficient_count_formula": "(out / q) * (in / q) * q",
            "svd_scalar_count_formula": "rank * (out + in + 1)",
            "rank_formula": "floor(coefficient_count / (out + in + 1)), capped at min(out, in)",
            "full_svd_computations": 1,
            "probe_fingerprint": probe_fingerprint,
        },
        "small_block_equal_budget_baselines": small_block_baselines,
    }
    if run is not None:
        run.metrics.update(result)
        run.summary = (
            f"Measured no-training Clifford projections for `{tensor_name}` ({out_features}x{in_features}) from "
            f"`{model}`; {len(arms)} projection arms, one shared FP64 probe set, one Gaussian control, and bounded "
            "small-block baseline subsets."
        )
        run.interpretation = (
            "These are tensor reconstruction and seeded random-probe action measurements only. The small-block "
            "equal-budget rows use an FP32 parameter budget equal in scalar count to q FP64 Clifford coefficients "
            "and only four optimizer steps; they are not a broad trained tournament. No model-quality or "
            "performance claim, checkpoint export, or winner claim is made."
        )
        run.add_section(
            "Clifford projection arms",
            markdown_table(
                [
                    {
                        "arm": name,
                        "q": row["q"],
                        "grouping": row["grouping"],
                        "seed": row["seed"],
                        "coefficients": row["coefficient_scalar_count"],
                        "FP64 coeff bytes": row["coefficient_bytes_fp64"],
                        "relative Frobenius error": row["weight_approximation"]["relative_frobenius_error"],
                        "retained energy": row["retained_energy_fraction"],
                        "probe action relative error": row["probe_action_vs_original"]["relative_frobenius_error"],
                    }
                    for name, row in arms.items()
                ]
            ),
        )
        run.add_section(
            "Equal coefficient-count SVD and random control",
            markdown_table(
                [
                    {
                        "q": q,
                        "SVD rank": row["rank"],
                        "SVD relative Frobenius error": row["relative_frobenius_error"],
                        "SVD probe action relative error": row["probe_action_vs_original"]["relative_frobenius_error"],
                        "Gaussian control arms": sum(arm["q"] == int(q) for arm in random_control_arms.values()),
                    }
                    for q, row in svd_rows.items()
                ]
            ),
        )
        run.add_section(
            "Bounded equal-budget small-block comparator subset",
            "The complete helper output is in `metrics.json` under `small_block_equal_budget_baselines`; it retains "
            "dense, SVD, SVD-Givens, magnitude-sparse, butterfly, rotor-weight, rotor-residual, "
            "and both random-control rows, including every infeasible/skipped row. "
            "These q-by-q FP32-budget comparisons are not equal-FP64-byte "
            "comparisons and are not a broad trained-baseline tournament.",
        )
    return result


def create_experiment_run(model: str, tensor_name: str, probe_rows: int) -> ExperimentRun:
    """Create the report object with the complete requested pilot configuration."""
    config = {
        "model": model,
        "tensor": tensor_name,
        "probe_rows": probe_rows,
        "dimensions": [2, 3],
        "grouping_seeds": [None, 0, 1, 2],
        "probe_seed": PROBE_SEED,
        "gaussian_control_seed": CONTROL_SEED,
        "projection": "blockwise Clifford left-geometric-product orthogonal projection",
        "coefficient_dtype": "float64",
        "svd": {
            "rank": "floor(coefficient_scalar_count / (out + in + 1))",
            "full_svd_computations": 1,
        },
        "small_block_equal_budget_baselines": {
            "blocks": "top-left contiguous q-by-q original tensor block for q=4 and q=8",
            "methods": [
                "dense", "svd", "svd_givens", "magnitude_sparse", "butterfly", "rotor_weight", "rotor_residual"
            ],
            "random_controls": ["spectrum_matched_random", "gaussian_random"],
            "storage_dtype": "float32",
            "byte_budget_formula": "q * 4",
            "fit_schedule": {"phases": [[['scales'], 2], [['angles', 'scales', 'residual'], 2]], "total_steps": 4},
            "fit_seed": 0,
            "scope_note": "bounded small-block comparator subset only; not the broad trained-baseline tournament",
        },
        "checkpoint_export": False,
        "no_download_or_fallback": True,
    }
    run = ExperimentRun(
        "exploration",
        model=model,
        method="clifford-block-projection",
        target=f"{tensor_name}.probes-{probe_rows}",
        config=config,
        title=f"Clifford projection pilot: {model} / {tensor_name}",
    )
    _unique_run_id(run)
    return run
