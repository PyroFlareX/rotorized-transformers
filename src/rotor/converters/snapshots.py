"""Atomic, weights-only conversion progress snapshots."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import torch

from rotor.storage import atomic_dir

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_STATE_FILENAME = "state.pt"
_OWNER_FILENAME = "owner.json"
_OWNER_MARKER = json.dumps(
    {"format": "rotor-conversion-snapshot", "schema_version": SNAPSHOT_SCHEMA_VERSION}, sort_keys=True
).encode("utf-8") + b"\n"
_SNAPSHOT_NAME = re.compile(r"^snapshot-(\d+)-([0-9a-f]{32})$")
_REQUIRED_KEYS = {
    "schema_version",
    "identity",
    "next_projection_index",
    "layers",
    "active",
    "fit_seconds",
    "notes",
}


def cpu_copy(value):
    """Recursively copy tensor and container values accepted by weights-only torch.load."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {cpu_copy(key): cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"snapshot contains unsupported value type {type(value).__name__}")


def validate_snapshot_state(state: object) -> dict:
    if not isinstance(state, dict) or set(state) != _REQUIRED_KEYS:
        raise ValueError("invalid conversion snapshot state fields")
    version = state.get("schema_version")
    if isinstance(version, bool) or version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported conversion snapshot schema {version!r}")
    if not isinstance(state.get("identity"), dict):
        raise ValueError("invalid conversion snapshot identity")
    next_index = state.get("next_projection_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int) or next_index < 0:
        raise ValueError("invalid conversion snapshot projection index")
    layers, fit_seconds = state.get("layers"), state.get("fit_seconds")
    if not isinstance(layers, dict) or not isinstance(fit_seconds, dict):
        raise ValueError("invalid conversion snapshot completed layer data")
    for module, record in layers.items():
        if not isinstance(module, str) or not module or not isinstance(record, dict):
            raise ValueError("invalid conversion snapshot layer record")
        if set(record) != {"entry", "state_dict"} or not isinstance(record["entry"], dict):
            raise ValueError("invalid conversion snapshot layer record fields")
        state_dict = record["state_dict"]
        if not isinstance(state_dict, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()
        ):
            raise ValueError("invalid conversion snapshot layer state")
    if not all(
        isinstance(module, str)
        and isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and seconds >= 0
        for module, seconds in fit_seconds.items()
    ):
        raise ValueError("invalid conversion snapshot fit times")
    active = state.get("active")
    if active is not None:
        if not isinstance(active, dict) or set(active) != {"module", "fit_state", "elapsed_seconds"}:
            raise ValueError("invalid conversion snapshot active fit data")
        if not isinstance(active["module"], str) or not active["module"]:
            raise ValueError("invalid conversion snapshot active module")
        if (
            not isinstance(active["elapsed_seconds"], (int, float))
            or isinstance(active["elapsed_seconds"], bool)
            or active["elapsed_seconds"] < 0
        ):
            raise ValueError("invalid conversion snapshot active elapsed time")
        fit_state = active["fit_state"]
        fit_keys = {"phase_index", "next_step", "history", "cascade_state", "optimizer_state"}
        if not isinstance(fit_state, dict) or set(fit_state) != fit_keys:
            raise ValueError("invalid conversion snapshot optimizer progress")
        if any(
            isinstance(fit_state[k], bool) or not isinstance(fit_state[k], int) or fit_state[k] < 0
            for k in ("phase_index", "next_step")
        ):
            raise ValueError("invalid conversion snapshot optimizer position")
        if not isinstance(fit_state["history"], list) or not all(
            isinstance(loss, (int, float)) and not isinstance(loss, bool) for loss in fit_state["history"]
        ):
            raise ValueError("invalid conversion snapshot loss history")
        cascade_state = fit_state["cascade_state"]
        if not isinstance(cascade_state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in cascade_state.items()
        ):
            raise ValueError("invalid conversion snapshot cascade state")
        optimizer_state = fit_state["optimizer_state"]
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {"state", "param_groups"}:
            raise ValueError("invalid conversion snapshot optimizer state")
        if not isinstance(optimizer_state["state"], dict) or not isinstance(optimizer_state["param_groups"], list):
            raise ValueError("invalid conversion snapshot optimizer state structure")
    if not isinstance(state.get("notes"), list) or not all(isinstance(note, str) for note in state["notes"]):
        raise ValueError("invalid conversion snapshot notes")
    return state


def _is_owned_snapshot(path: Path) -> bool:
    if _SNAPSHOT_NAME.fullmatch(path.name) is None or path.is_symlink() or not path.is_dir():
        return False
    marker = path / _OWNER_FILENAME
    state = path / SNAPSHOT_STATE_FILENAME
    return (
        not marker.is_symlink()
        and marker.is_file()
        and marker.read_bytes() == _OWNER_MARKER
        and not state.is_symlink()
        and state.is_file()
        and state.stat().st_size > 0
    )


def load_snapshot(path: str | Path) -> dict:
    """Load and validate a published snapshot using PyTorch's restricted unpickler."""
    path = Path(path)
    if not _is_owned_snapshot(path):
        raise ValueError(f"not a valid owned conversion snapshot directory: {path}")
    state = torch.load(path / SNAPSHOT_STATE_FILENAME, map_location="cpu", weights_only=True)
    return validate_snapshot_state(state)


def _valid_owned_snapshot(path: Path, identity: dict) -> bool:
    try:
        if not _is_owned_snapshot(path):
            return False
        state = torch.load(path / SNAPSHOT_STATE_FILENAME, map_location="cpu", weights_only=True)
        validate_snapshot_state(state)
        return state["identity"] == identity
    except Exception:
        return False


def _prune(root: Path, keep: int, identity: dict) -> None:
    if not root.is_dir() or root.is_symlink():
        return
    owned = [child for child in root.iterdir() if _valid_owned_snapshot(child, identity)]
    owned.sort(key=lambda p: int(_SNAPSHOT_NAME.fullmatch(p.name).group(1)), reverse=True)
    for old in owned[keep:]:
        shutil.rmtree(old)

def prune_snapshots(root: str | Path, keep: int, identity: dict) -> None:
    """Prune only valid snapshots owned by the given conversion identity."""
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("snapshot_keep must be an integer >= 1")
    _prune(Path(root), keep, identity)


def publish_snapshot(root: str | Path, state: dict, keep: int) -> Path:
    """Atomically publish one snapshot and rotate this conversion's valid owned snapshots."""
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("snapshot_keep must be an integer >= 1")
    state = validate_snapshot_state(cpu_copy(state))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        dest = root / f"snapshot-{time.time_ns()}-{uuid.uuid4().hex}"
        if not dest.exists():
            break
    with atomic_dir(dest) as tmp:
        state_path = tmp / SNAPSHOT_STATE_FILENAME
        with state_path.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        marker_path = tmp / _OWNER_FILENAME
        with marker_path.open("wb") as f:
            f.write(_OWNER_MARKER)
            f.flush()
            os.fsync(f.fileno())
    _prune(root, keep, state["identity"])
    return dest
