"""Behavioral gates for joint SwiGLU fitting, exact recovery, and serialized modules."""

from __future__ import annotations

import copy
import io
import shutil

import pytest
import torch
import torch.nn.functional as F

from rotor.converters.fit import fit_joint_ffn, joint_ffn_output
from rotor.formats.checkpoint import layer_accounting, load_structured_layers, write_structured
from rotor.formats.manifest import BaseProvenance, LayerEntry, Manifest
from rotor.math.cascade import CascadeSpec, Topology
from rotor.storage import atomic_dir
from rotor.training.layerwise import FitSchedule

_KEYS = ("gate", "up", "down")
_MODULES = {
    "gate": "model.layers.0.mlp.gate_proj",
    "up": "model.layers.0.mlp.up_proj",
    "down": "model.layers.0.mlp.down_proj",
}
_FAMILIES = {"gate": "ffn.gate", "up": "ffn.up", "down": "ffn.down"}


def _problem(seed: int = 12):
    generator = torch.Generator().manual_seed(seed)
    specs = {
        "gate": CascadeSpec(4, 5, 1, 2, Topology("butterfly", 0)),
        "up": CascadeSpec(4, 5, 1, 2, Topology("butterfly", 1)),
        "down": CascadeSpec(5, 4, 1, 2, Topology("butterfly", 2)),
    }
    weights = {
        "gate": torch.randn(5, 4, generator=generator) * 0.3,
        "up": torch.randn(5, 4, generator=generator) * 0.3,
        "down": torch.randn(4, 5, generator=generator) * 0.3,
    }
    biases = {
        "gate": torch.randn(5, generator=generator) * 0.1,
        "up": torch.randn(5, generator=generator) * 0.1,
        "down": torch.randn(4, generator=generator) * 0.1,
    }
    X_train = torch.randn(48, 4, generator=generator)
    heldout_X = torch.randn(40, 4, generator=generator)
    return X_train, heldout_X, weights, biases, specs


def _schedule(steps: int = 100) -> FitSchedule:
    return FitSchedule(
        phases=[(("angles", "scales"), steps)],
        lrs={"angles": 0.02, "scales": 0.02},
        cosine_decay=True,
    )


def _student_output(layers, x: torch.Tensor) -> torch.Tensor:
    return layers["down"](F.silu(layers["gate"](x)) * layers["up"](x))


def _normalized_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prediction - target).square().sum() / target.square().sum()).detach())


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prediction - target).norm() / target.norm()).detach())


def _assert_tree_equal(actual, expected, path: str = "state") -> None:
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected), path
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_tree_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            _assert_tree_equal(got, want, f"{path}[{i}]")
    else:
        assert actual == expected, path


def test_joint_fit_improves_composed_swiglu_error_on_disjoint_heldout_rows() -> None:
    X_train, heldout_X, weights, biases, specs = _problem()
    teacher = joint_ffn_output(heldout_X, weights, biases)
    gate = F.linear(heldout_X, weights["gate"], biases["gate"])
    up = F.linear(heldout_X, weights["up"], biases["up"])
    explicit_teacher = F.linear(F.silu(gate) * up, weights["down"], biases["down"])
    torch.testing.assert_close(teacher, explicit_teacher, rtol=0, atol=0)
    zero_step = FitSchedule(phases=[(("angles", "scales"), 0)])
    initial, _ = fit_joint_ffn(
        X_train,
        weights,
        specs,
        zero_step,
        biases=biases,
        heldout_X=heldout_X,
        seed=27,
        init_angle_std=0.03,
    )
    initial_error = _normalized_squared_error(_student_output(initial, heldout_X), teacher)

    fitted, metrics = fit_joint_ffn(
        X_train,
        weights,
        specs,
        _schedule(120),
        biases=biases,
        heldout_X=heldout_X,
        seed=27,
        init_angle_std=0.03,
        init=initial,
        combined_byte_budget=sum((specs[key].param_count() + biases[key].numel()) * 2 for key in _KEYS),
    )
    expected_per_projection_bytes = {key: (specs[key].param_count() + biases[key].numel()) * 2 for key in _KEYS}
    assert metrics["per_projection_bytes"] == expected_per_projection_bytes
    assert metrics["combined_bytes"] == sum(expected_per_projection_bytes.values())
    assert metrics["combined_dense_bytes"] == sum(
        weights[key].numel() * weights[key].element_size() + biases[key].numel() * biases[key].element_size()
        for key in _KEYS
    )
    final_error = _normalized_squared_error(_student_output(fitted, heldout_X), teacher)
    assert not any(torch.equal(train_row, heldout_row) for train_row in X_train for heldout_row in heldout_X)
    assert final_error < initial_error
    assert metrics["heldout_output_relative_error"] == pytest.approx(final_error, rel=1e-6)
    train_teacher = joint_ffn_output(X_train, weights, biases)
    train_error = _normalized_squared_error(_student_output(fitted, X_train), train_teacher)
    assert metrics["train_output_relative_error"] == pytest.approx(train_error, rel=1e-6)
    with pytest.raises(ValueError, match="exceeding combined_byte_budget"):
        fit_joint_ffn(
            X_train,
            weights,
            specs,
            FitSchedule(phases=[]),
            biases=biases,
            storage_dtype="bfloat16",
            combined_byte_budget=sum(expected_per_projection_bytes.values()) - 1,
        )


def test_joint_fit_rejects_overlapping_train_and_heldout_rows() -> None:
    X_train, _, weights, biases, specs = _problem(seed=99)
    heldout_X = torch.cat((torch.randn(3, 4, generator=torch.Generator().manual_seed(5)), X_train[:1]), dim=0)
    with pytest.raises(ValueError, match="fit and held-out rows must be disjoint"):
        fit_joint_ffn(
            X_train,
            weights,
            specs,
            FitSchedule(phases=[]),
            biases=biases,
            heldout_X=heldout_X,
        )


def test_joint_fit_resume_matches_uninterrupted_modules_and_optimizer_exactly() -> None:
    X_train, heldout_X, weights, biases, specs = _problem(seed=23)
    schedule = FitSchedule(
        phases=[(("scales",), 3), (("angles", "scales"), 6)],
        lrs={"angles": 0.015, "scales": 0.015},
        cosine_decay=True,
    )
    common = {
        "biases": biases,
        "heldout_X": heldout_X,
        "seed": 6,
        "init_angle_std": 0.04,
    }
    uninterrupted_final = {}

    def retain_final(state):
        if state["phase_index"] == 1 and state["next_step"] == 6:
            uninterrupted_final["state"] = copy.deepcopy(state)

    uninterrupted, _ = fit_joint_ffn(X_train, weights, specs, schedule, **common, progress_callback=retain_final)
    assert "state" in uninterrupted_final

    interrupted_state = {}

    class Interrupted(Exception):
        pass

    def interrupt_after_two_steps_in_second_phase(state):
        if state["phase_index"] == 1 and state["next_step"] == 2:
            interrupted_state["state"] = copy.deepcopy(state)
            buffer = io.BytesIO()
            torch.save(interrupted_state["state"], buffer)
            buffer.seek(0)
            interrupted_state["state"] = torch.load(buffer, weights_only=True)
            raise Interrupted

    with pytest.raises(Interrupted):
        fit_joint_ffn(
            X_train,
            weights,
            specs,
            schedule,
            **common,
            progress_callback=interrupt_after_two_steps_in_second_phase,
        )

    resumed_final = {}

    def retain_resumed_final(state):
        if state["phase_index"] == 1 and state["next_step"] == 6:
            resumed_final["state"] = copy.deepcopy(state)

    resumed, _ = fit_joint_ffn(
        X_train,
        weights,
        specs,
        schedule,
        **common,
        resume_state=interrupted_state["state"],
        progress_callback=retain_resumed_final,
    )
    assert "state" in resumed_final
    _assert_tree_equal(resumed_final["state"], uninterrupted_final["state"])
    for key in _KEYS:
        _assert_tree_equal(resumed[key].state_dict(), uninterrupted[key].state_dict(), key)


def _manifest_for(layers, specs, storage_dtype: str) -> Manifest:
    entries = []
    for key in _KEYS:
        module = _MODULES[key]
        layer = layers[module]
        spec = specs[key]
        structured_bytes, learned = layer_accounting(module, layer, storage_dtype)
        entries.append(
            LayerEntry(
                module=module,
                source_tensor=f"model.language_model.layers.0.mlp.{key}_proj.weight",
                family=_FAMILIES[key],
                spec=spec,
                storage_dtype=storage_dtype,
                has_bias=layer.bias is not None,
                dense_params=spec.dense_param_count(),
                structured_params=spec.param_count(),
                dense_bytes=spec.dense_param_count() * 4,
                structured_bytes=structured_bytes,
                learned_scalars=learned,
                dense_dtype="F32",
            )
        )
    return Manifest(
        base=BaseProvenance(
            hf_id="unit/joint-ffn",
            revision="test",
            path="/unused/test-base",
            file_sha256={},
            tied_word_embeddings=None,
            architectures=[],
        ),
        layers=entries,
        conversion_config={"name": "joint-ffn-test", "budget": {"storage_dtype": storage_dtype}},
        environment={},
    )


def test_joint_fit_structured_export_load_preserves_composed_output_with_bfloat16_rounding(
    tmp_storage,
) -> None:
    X_train, _, weights, biases, specs = _problem(seed=41)
    modules, _ = fit_joint_ffn(
        X_train,
        weights,
        specs,
        FitSchedule(phases=[(("angles", "scales"), 0)]),
        biases=biases,
        seed=14,
        init_angle_std=0.02,
    )
    assert all(parameter.dtype == torch.float32 for module in modules.values() for parameter in module.parameters())
    layers = {_MODULES[key]: modules[key] for key in _KEYS}
    manifest = _manifest_for(layers, specs, "bfloat16")
    destination = tmp_storage.derived_path("joint-ffn-test")
    with atomic_dir(destination) as staging:
        write_structured(staging, manifest, layers, base_dir=None)

    loaded_manifest, loaded = load_structured_layers(destination, dtype=torch.float32)
    assert loaded_manifest.to_json() == manifest.to_json()
    restored = {key: loaded[_MODULES[key]] for key in _KEYS}
    generator = torch.Generator().manual_seed(71)
    probe = torch.randn(32, 4, generator=generator)
    reference_output = _student_output(modules, probe)
    restored_output = _student_output(restored, probe)

    for key in _KEYS:
        for name, parameter in modules[key].state_dict().items():
            stored = restored[key].state_dict()[name]
            assert torch.equal(stored, parameter.to(torch.bfloat16).to(torch.float32)), f"{key}.{name}"
    assert _relative_error(restored_output, reference_output) < 0.1


def test_joint_runner_publishes_tiny_checkpoint_and_report(tmp_path, tiny_ckpt, tmp_storage, monkeypatch) -> None:
    from rotor.converters.joint_ffn_runner import load_joint_config, run_joint_ffn

    base = tmp_storage.base_root / "tiny-joint-ffn"
    base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tiny_ckpt, base)
    train_text = "\n".join(
        f"TRAINING {index} alpha {index * 17} beta {index * 31} gamma" for index in range(120)
    )
    heldout_text = "\n".join(
        f"HELDOUT {index} delta {index * 19} epsilon {index * 37} zeta" for index in range(120)
    )
    train_path = tmp_path / "train.txt"
    heldout_path = tmp_path / "heldout.txt"
    train_path.write_text(train_text, encoding="utf-8")
    heldout_path.write_text(heldout_text, encoding="utf-8")
    monkeypatch.setattr(
        "rotor.converters.joint_ffn_runner.model_entry",
        lambda _key: ("joint-test", {"hf_id": "tiny/joint-test", "revision": "tiny", "local_path": str(base)}),
    )
    import yaml

    config_data = {
        "name": "joint-runner-smoke",
        "model": "joint-test",
        "layer": 0,
        "budget": {"byte_ratio": 1.0, "storage_dtype": "float32"},
        "cascade": {"topology": {"kind": "brick", "seed": 0}, "stages_per_rotor": 1},
        "fit": {
            "seed": 3,
            "device": "cpu",
            "init_angle_std": 0.01,
            "schedule": {"phases": [{"train": ["angles", "scales"], "steps": 0}]},
        },
        "calibration": {"path": str(train_path), "seq_len": 8, "num_windows": 4, "seed": 0, "max_rows": 32},
        "heldout": {"path": str(heldout_path), "seq_len": 8, "num_windows": 4, "seed": 1, "max_rows": 32},
    }
    config_path = tmp_path / "joint.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config_data = load_joint_config(config_path)
    output = run_joint_ffn(
        config_data,
        storage=tmp_storage,
        log=lambda _: None,
        snapshot_interval_seconds=3600,
        report_root=tmp_path / "reports",
    )
    manifest, layers = load_structured_layers(output, dtype=torch.float32)
    assert len(manifest.layers) == 3
    assert len(layers) == 3
    assert (tmp_path / "reports").is_dir()
