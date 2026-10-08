"""Shared pytest fixtures. Everything here is CPU-only and never touches the real NAS."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

# Keep test output quiet; must be set before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _isolated_reports_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every ``ExperimentRun`` without an explicit ``out_root`` publishes into a tmp dir, never the repo."""
    root = tmp_path_factory.mktemp("reports")
    monkeypatch.setenv("ROTOR_REPORTS_DIR", str(root))
    return root


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deterministic random tiny Qwen3.5 checkpoint in the real on-disk layout (shared, read-only)."""
    from rotor.models.tiny import make_tiny_checkpoint

    return make_tiny_checkpoint(tmp_path_factory.mktemp("tiny_ckpt"), seed=0)


@pytest.fixture
def tmp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A ``Storage`` rooted in ``tmp_path/nas``; ``ROTOR_STORAGE_CONFIG`` points at the generated yaml."""
    from rotor.storage import load_storage

    nas = tmp_path / "nas"
    nas.mkdir()
    cfg = {
        "version": 1,
        "nas_root": str(nas),
        "base_root": str(nas / "base"),
        "derived_root": str(nas / "derived"),
        "rotor_derived_root": str(nas / "derived" / "rotor-scaler"),
        "preexisting_read_only": [str(nas / "legacy")],
    }
    cfg_path = tmp_path / "storage.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ROTOR_STORAGE_CONFIG", str(cfg_path))
    return load_storage()
