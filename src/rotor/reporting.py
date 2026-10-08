"""Experiment run directories: ``reports/experiments/<run_id>/`` with report.md, metrics.json, config.yaml,
environment.json, figures/, logs/.

Usage::

    with ExperimentRun("convert", model="qwen3.5-0.8b", method="rotor-butterfly-weight", target="b025",
                       config=resolved_config) as run:
        run.log("fitting ...")
        run.metrics["storage"] = {...}            # measured values
        run.estimates["flops_per_token"] = ...    # formula-derived values (kept separate, labeled estimates)
        run.add_section("Baselines", markdown_table)
        run.interpretation = "..."
    # -> published atomically as reports/experiments/<run_id>/

On an exception the run is still published with ``status: failed`` and the traceback (failed runs are reported,
never silently dropped), then the exception propagates. ``run_id`` = ``<model>__<method>__<target>__<UTC stamp>``.
"""

from __future__ import annotations

import json
import re
import string
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rotor.environment import capability_summary, detect_environment
from rotor.storage import REPO_ROOT, StorageError, atomic_dir

EXPERIMENTS_DIR = REPO_ROOT / "reports" / "experiments"
TEMPLATE_PATH = REPO_ROOT / "reports" / "templates" / "experiment_report.md"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: str) -> str:
    return _SAFE.sub("-", str(s)).strip("-") or "na"


def make_run_id(model: str, method: str, target: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(model)}__{_slug(method)}__{_slug(target)}__{stamp}"


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        elif isinstance(v, list | tuple) and len(v) > 8:
            out[key] = f"[{len(v)} items]"
        else:
            out[key] = v
    return out


def markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_"
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.6g}"
        return "" if v is None else str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _kv_table(d: dict, limit: int = 200) -> str:
    flat = flatten(d)
    rows = [{"key": k, "value": v} for k, v in list(flat.items())[:limit]]
    extra = f"\n\n_{len(flat) - limit} more keys in metrics.json_" if len(flat) > limit else ""
    return markdown_table(rows, ["key", "value"]) + extra


def _environment_summary(env: dict) -> str:
    cpu, mem, torch = env.get("cpu", {}), env.get("memory", {}), env.get("torch", {})
    gpus = ", ".join(f"{g.get('name') or g.get('device_id')} ({g.get('driver')})" for g in env.get("gpus_pci", []))
    gib = 2**30
    rows = [
        {"item": "host", "value": env.get("hostname")},
        {"item": "cpu", "value": f"{cpu.get('model')} ({cpu.get('logical_cpus')} threads)"},
        {"item": "ram", "value": f"{(mem.get('ram_total_bytes') or 0) / gib:.1f} GiB"},
        {"item": "swap used", "value": f"{(mem.get('swap_used_bytes') or 0) / gib:.2f} GiB"},
        {"item": "gpus (pci)", "value": gpus or "none"},
        {"item": "torch", "value": f"{torch.get('version')} cuda={torch.get('cuda_available')}"},
        {"item": "git", "value": f"{env.get('git', {}).get('commit')} dirty={env.get('git', {}).get('dirty')}"},
    ]
    return markdown_table(rows, ["item", "value"])


class ExperimentRun:
    """Collects one experiment's outputs and publishes them as a run directory (see module docstring)."""

    def __init__(
        self,
        kind: str,
        *,
        model: str,
        method: str,
        target: str,
        config: dict,
        title: str | None = None,
        out_root: Path | None = None,
        environment: dict | None = None,
    ):
        self.kind = kind
        self.model, self.method, self.target = model, method, target
        self.config = config
        self.title = title or f"{kind}: {model} / {method} / {target}"
        self.created = datetime.now(UTC)
        self.run_id = make_run_id(model, method, target, self.created)
        self.out_root = Path(out_root or EXPERIMENTS_DIR)
        self.environment = environment if environment is not None else detect_environment()
        self.metrics: dict = {}
        self.estimates: dict = {}
        self.sections: list[tuple[str, str]] = []
        self.figures: dict[str, bytes] = {}
        self.log_lines: list[str] = []
        self.log_references: list[str] = []
        self.summary = ""
        self.interpretation = ""
        self.status = "running"
        self.failure: str | None = None
        self.path: Path | None = None

    # --- collection -------------------------------------------------------------------------------------------
    def log(self, message: str, echo: bool = True) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}"
        self.log_lines.append(line)
        if echo:
            print(message, flush=True)

    def add_section(self, title: str, markdown: str) -> None:
        self.sections.append((title, markdown))

    def add_figure(self, filename: str, data: bytes) -> str:
        """Store figure bytes (e.g. PNG/SVG from matplotlib ``savefig`` to a BytesIO); returns relative path."""
        name = _slug(filename)
        self.figures[name] = data
        return f"figures/{name}"

    def add_log_reference(self, ref: str) -> None:
        self.log_references.append(ref)

    # --- publishing -------------------------------------------------------------------------------------------
    def render_report(self) -> str:
        template = string.Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
        sections = "\n\n".join(f"## {t}\n\n{body}" for t, body in self.sections) or "_(no additional sections)_"
        warnings = capability_summary(self.environment)
        logs = ["- `logs/run.log`"] + [f"- {r}" for r in self.log_references]
        return template.safe_substitute(
            title=self.title,
            run_id=self.run_id,
            status=self.status,
            created=self.created.isoformat(timespec="seconds"),
            kind=self.kind,
            model=self.model,
            method=self.method,
            target=self.target,
            summary=self.summary or "_(none provided)_",
            sections=sections,
            metrics_table=_kv_table(self.metrics) if self.metrics else "_(no measured metrics)_",
            estimates_table=_kv_table(self.estimates) if self.estimates else "_(none)_",
            environment_summary=_environment_summary(self.environment),
            warnings="\n".join(f"- {w}" for w in warnings) or "- none",
            failure=f"```\n{self.failure}\n```" if self.failure else "None.",
            interpretation=self.interpretation or "_(not yet interpreted)_",
            logs="\n".join(logs),
            figures="\n".join(f"![{n}](figures/{n})" for n in self.figures) or "_(none)_",
        )

    def finalize(self, status: str = "completed") -> Path:
        if self.path is not None:
            raise StorageError(f"run {self.run_id} already published at {self.path}")
        self.status = status
        dest = self.out_root / self.run_id
        payload = {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "created_utc": self.created.isoformat(),
            "finished_utc": datetime.now(UTC).isoformat(),
            "model": self.model,
            "method": self.method,
            "target": self.target,
            "metrics": self.metrics,
            "estimates": self.estimates,
            "failure": self.failure,
        }
        with atomic_dir(dest) as tmp:
            (tmp / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
            (tmp / "config.yaml").write_text(yaml.safe_dump(_plain(self.config), sort_keys=False))
            (tmp / "environment.json").write_text(json.dumps(self.environment, indent=2, default=str) + "\n")
            (tmp / "logs").mkdir()
            (tmp / "logs" / "run.log").write_text("\n".join(self.log_lines) + ("\n" if self.log_lines else ""))
            if self.figures:
                (tmp / "figures").mkdir()
                for name, data in self.figures.items():
                    (tmp / "figures" / name).write_bytes(data)
            (tmp / "report.md").write_text(self.render_report())
        self.path = dest
        return dest

    def __enter__(self) -> ExperimentRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finalize("completed")
        else:
            self.failure = "".join(traceback.format_exception(exc_type, exc, tb))
            self.log(f"FAILED: {exc_type.__name__}: {exc}", echo=False)
            self.finalize("failed")
        return False


def _plain(obj):
    """YAML-safe deep copy (Paths/tuples/dataclass-ish objects become plain types)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        return _plain(obj.to_dict())
    return str(obj)
