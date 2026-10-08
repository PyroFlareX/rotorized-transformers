"""Run records: every measurement written by a script carries config, environment, and timestamps."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from rotor.formats.manifest import environment_record
from rotor.storage import REPO_ROOT, atomic_write_bytes

REPORTS_DIR = REPO_ROOT / "reports" / "experiments"


def write_run_record(kind: str, name: str, payload: dict, out_dir: Path | None = None) -> Path:
    """Write ``<out_dir>/<UTC timestamp>-<kind>-<name>.json`` atomically. Returns the path.

    ``payload`` must contain the measured results plus the exact config used; ``environment``
    (versions, hardware, git commit, timestamp) is added here.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    path = Path(out_dir or REPORTS_DIR) / f"{stamp}-{kind}-{safe}.json"
    record = {"kind": kind, "name": name, "environment": environment_record(), **payload}
    atomic_write_bytes(path, (json.dumps(record, indent=2, default=str) + "\n").encode())
    return path
