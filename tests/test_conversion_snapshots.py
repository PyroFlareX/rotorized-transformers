"""Recovery checkpoints preserve optimizer progress and published conversion results."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from rotor.converters.pipeline import ConversionConfig, convert
from rotor.converters.snapshots import load_snapshot, publish_snapshot
from rotor.formats import WEIGHTS_FILENAME
from rotor.math.cascade import CascadeSpec, RotorCascade
from rotor.math.topology import Topology
from rotor.training.layerwise import FitSchedule, optimize_cascade


@pytest.mark.parametrize("stop_after", [2, 3, 4, 8])
def test_optimizer_resume_matches_uninterrupted(stop_after):
    torch.manual_seed(7)
    initial = RotorCascade(CascadeSpec(4, 4, 2, 2, Topology("brick")))
    target = torch.randn(4, 4)
    schedule = FitSchedule(phases=[(("scales",), 3), (("angles", "scales"), 5)], cosine_decay=True)

    def loss(cascade):
        return (cascade.dense_weight() - target).square().mean()

    reference = copy.deepcopy(initial)
    expected_history = optimize_cascade(reference, loss, schedule)
    interrupted = copy.deepcopy(initial)
    captured = {}

    class Interrupted(Exception):
        pass

    def capture(state):
        if len(state["history"]) == stop_after:
            captured.update(copy.deepcopy(state))
            raise Interrupted

    with pytest.raises(Interrupted):
        optimize_cascade(interrupted, loss, schedule, progress_callback=capture)
    resumed = copy.deepcopy(initial)
    actual_history = optimize_cascade(resumed, loss, schedule, resume_state=captured)
    assert actual_history == expected_history
    for name, parameter in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], parameter, rtol=0, atol=0)


def test_conversion_resume_skips_completed_and_preserves_weights(tmp_storage, tiny_ckpt):
    config = ConversionConfig(
        name="recovery", model="qwen3.5-0.8b", layers=[0], families=["ffn.up", "ffn.down"],
        fit={"device": "cpu", "schedule": {"phases": [
            {"train": ["scales"], "steps": 2}, {"train": ["angles", "scales"], "steps": 3},
        ]}},
    )
    saved = []

    class Interrupted(Exception):
        pass

    def interrupt(message):
        if message.startswith("saved conversion snapshot "):
            path = Path(message.removeprefix("saved conversion snapshot "))
            saved.append(path)
            state = load_snapshot(path)
            if state["next_projection_index"] == 1 and state["active"] is not None:
                raise Interrupted

    with pytest.raises(Interrupted):
        convert(config, storage=tmp_storage, base_dir=tiny_ckpt, report=False,
                snapshot_interval_seconds=1e-9, log=interrupt)
    root = saved[-1].parent
    assert len(list(root.glob("snapshot-*"))) == 3
    state = load_snapshot(saved[-1])
    assert set(state["layers"]) == {state["identity"]["projections"][0]["module"]}
    changed = copy.deepcopy(config)
    changed.fit["seed"] = 42
    with pytest.raises(ValueError, match="identity"):
        convert(changed, storage=tmp_storage, base_dir=tiny_ckpt, report=False, resume_from=saved[-1])
    resumed = convert(config, storage=tmp_storage, base_dir=tiny_ckpt, report=False,
                      resume_from=saved[-1], log=lambda _: None)
    reference_config = copy.deepcopy(config)
    reference_config.name = "reference"
    reference = convert(reference_config, storage=tmp_storage, base_dir=tiny_ckpt,
                        report=False, log=lambda _: None)
    expected, actual = load_file(reference / WEIGHTS_FILENAME), load_file(resumed / WEIGHTS_FILENAME)
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)


def test_failed_snapshot_publish_preserves_previous(tmp_path, monkeypatch):
    state = {"schema_version": 1, "identity": {}, "next_projection_index": 0,
             "layers": {}, "active": None, "fit_seconds": {}, "notes": []}
    first = publish_snapshot(tmp_path, state, 3)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    def fail(*args, **kwargs):
        raise OSError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="write failed"):
            publish_snapshot(tmp_path, state, 3)
    assert load_snapshot(first) == state
    for _ in range(4):
        publish_snapshot(tmp_path, state, 3)
    assert not first.exists()
    assert len(list(tmp_path.glob("snapshot-*"))) == 3
    assert unrelated.is_dir()
