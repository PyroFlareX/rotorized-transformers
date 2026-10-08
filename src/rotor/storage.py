"""Centralized storage configuration and safe-write helpers.

All filesystem locations for checkpoints come from ``configs/storage.yaml`` (optionally overlaid by
``configs/storage.local.yaml`` or the file named by ``ROTOR_STORAGE_CONFIG``). Scripts must call
``load_storage()`` instead of hardcoding paths.

Safety invariants enforced here:

- Derived outputs must resolve inside ``nas_root`` (``assert_under_nas``).
- Base checkpoints and ``preexisting_read_only`` paths are never written (``assert_writable``).
- Writes go to a temporary sibling and are published with an atomic rename (``atomic_dir`` /
  ``atomic_write_bytes``); an existing destination is never overwritten.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class StorageError(RuntimeError):
    pass


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass(frozen=True)
class Storage:
    nas_root: Path
    base_root: Path
    derived_root: Path
    rotor_derived_root: Path
    preexisting_read_only: tuple[Path, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: dict) -> Storage:
        return cls(
            nas_root=Path(d["nas_root"]),
            base_root=Path(d["base_root"]),
            derived_root=Path(d["derived_root"]),
            rotor_derived_root=Path(d["rotor_derived_root"]),
            preexisting_read_only=tuple(Path(p) for p in d.get("preexisting_read_only", [])),
        )

    def validate(self) -> None:
        for name in ("base_root", "derived_root", "rotor_derived_root"):
            p = getattr(self, name)
            if not _is_within(p, self.nas_root):
                raise StorageError(f"{name}={p} is outside nas_root={self.nas_root}")

    def nas_available(self) -> bool:
        return self.nas_root.is_dir()

    def assert_under_nas(self, path: Path) -> Path:
        if not _is_within(path, self.nas_root):
            raise StorageError(f"refusing to use {path}: model artifacts must live under {self.nas_root}")
        return path

    def assert_writable(self, path: Path) -> Path:
        """Refuse destinations inside base checkpoints or pre-existing read-only model dirs."""
        self.assert_under_nas(path)
        if _is_within(path, self.base_root):
            raise StorageError(f"refusing to write {path}: base checkpoints are read-only")
        for ro in self.preexisting_read_only:
            if _is_within(path, ro):
                raise StorageError(f"refusing to write {path}: {ro} is pre-existing read-only content")
        return path

    def base_path(self, local_path: str | Path) -> Path:
        p = Path(local_path)
        return self.assert_under_nas(p if p.is_absolute() else self.base_root / p)

    def derived_path(self, *parts: str) -> Path:
        return self.assert_writable(self.rotor_derived_root.joinpath(*parts))


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(os.path.abspath(root))
        return True
    except ValueError:
        return False


def load_storage(path: str | Path | None = None) -> Storage:
    """Load storage config: explicit path > ``$ROTOR_STORAGE_CONFIG`` > repo default (+ local overlay)."""
    explicit = path or os.environ.get("ROTOR_STORAGE_CONFIG")
    if explicit:
        data = yaml.safe_load(Path(explicit).read_text())
    else:
        data = yaml.safe_load((CONFIG_DIR / "storage.yaml").read_text())
        local = CONFIG_DIR / "storage.local.yaml"
        if local.exists():
            data = _merge(data, yaml.safe_load(local.read_text()) or {})
    st = Storage.from_dict(data)
    st.validate()
    return st


def load_models(path: str | Path | None = None) -> dict:
    data = yaml.safe_load(Path(path or CONFIG_DIR / "models.yaml").read_text())
    local = CONFIG_DIR / "models.local.yaml"
    if path is None and local.exists():
        data = _merge(data, yaml.safe_load(local.read_text()) or {})
    return data


def model_entry(key_or_hf_id: str, models: dict | None = None) -> tuple[str, dict]:
    models = models or load_models()
    for key, entry in models["models"].items():
        if key_or_hf_id in (key, entry["hf_id"]):
            return key, entry
    raise KeyError(f"model {key_or_hf_id!r} not in configs/models.yaml")


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def atomic_dir(dest: Path) -> Iterator[Path]:
    """Yield a temp dir next to ``dest``; rename to ``dest`` on success, remove on failure.

    Raises if ``dest`` already exists: outputs are never overwritten.
    """
    dest = Path(dest)
    if dest.exists():
        raise StorageError(f"refusing to overwrite existing output {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=dest.parent))
    try:
        yield tmp
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest = Path(dest)
    if dest.exists():
        raise StorageError(f"refusing to overwrite existing file {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.tmp-", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
