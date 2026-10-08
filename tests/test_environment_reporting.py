"""rotor.environment and rotor.reporting: probes are serialisable, runs publish complete directories atomically."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from rotor.environment import ResourceMonitor, capability_summary, detect_environment
from rotor.reporting import ExperimentRun, markdown_table
from rotor.storage import StorageError

REQUIRED_ENV_KEYS = {
    "cpu", "memory", "gpus_pci", "nvidia", "vulkan", "torch", "libraries", "tools", "git", "storage",
}  # fmt: skip


@pytest.fixture(scope="module")
def env() -> dict:
    return detect_environment()


def test_detect_environment_keys_and_json(env):
    assert REQUIRED_ENV_KEYS <= set(env)
    json.dumps(env)  # must not raise: every probe value is a plain JSON type
    assert env["cpu"]["logical_cpus"] >= 1
    assert isinstance(env["cpu"]["isa"], dict)
    assert isinstance(capability_summary(env), list)


def test_detect_environment_without_storage():
    assert "storage" not in detect_environment(include_storage=False)


def test_resource_monitor_measures():
    with ResourceMonitor() as mon:
        time.sleep(0.02)
        _ = bytearray(1 << 20)
    d = mon.as_dict()
    assert d["wall_seconds"] >= 0.02
    assert d["cpu_seconds"] >= 0
    assert d["peak_rss_bytes"] > 0
    assert d["measured"] is True
    json.dumps(d)


def _run(tmp_path: Path, **kw) -> ExperimentRun:
    defaults = {"model": "m", "method": "meth", "target": "b025", "config": {"a": 1, "p": tmp_path}}
    return ExperimentRun("unit", out_root=tmp_path / "out", environment={"hostname": "t"}, **{**defaults, **kw})


def test_run_publishes_all_files(tmp_path):
    with _run(tmp_path) as run:
        run.log("hello", echo=False)
        run.metrics["x"] = 1.5
        run.estimates["flops"] = 123
        run.add_figure("f.png", b"\x89PNG")
    assert run.path is not None and run.path.parent == tmp_path / "out"
    for name in ("report.md", "metrics.json", "config.yaml", "environment.json", "logs/run.log", "figures/f.png"):
        assert (run.path / name).is_file(), name
    metrics = json.loads((run.path / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["metrics"] == {"x": 1.5}
    assert metrics["estimates"] == {"flops": 123}
    assert yaml.safe_load((run.path / "config.yaml").read_text())["a"] == 1
    assert json.loads((run.path / "environment.json").read_text()) == {"hostname": "t"}
    assert "hello" in (run.path / "logs/run.log").read_text()
    assert run.run_id.startswith("m__meth__b025__")


def test_failed_run_publishes_traceback_and_reraises(tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with _run(tmp_path) as run:
            run.metrics["partial"] = 1
            raise ValueError("boom")
    assert run.path is not None
    payload = json.loads((run.path / "metrics.json").read_text())
    assert payload["status"] == "failed"
    assert "Traceback" in payload["failure"] and "ValueError: boom" in payload["failure"]
    assert payload["metrics"] == {"partial": 1}
    report = (run.path / "report.md").read_text()
    assert "**failed**" in report and "ValueError: boom" in report


def test_runs_never_overwrite(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    second.created = first.created  # force an identical run id to hit the collision path
    second.run_id = first.run_id
    first.finalize()
    with pytest.raises(StorageError):
        second.finalize()
    kept = json.loads((first.path / "metrics.json").read_text())
    assert kept["run_id"] == first.run_id and kept["status"] == "completed"
    assert len(list((tmp_path / "out").iterdir())) == 1  # no stray temp dirs either
    with pytest.raises(StorageError):
        first.finalize()  # an already-published run cannot be published again


def test_distinct_runs_get_distinct_dirs(tmp_path):
    a = _run(tmp_path, method="a")
    b = _run(tmp_path, method="b")
    a.finalize()
    b.finalize()
    assert a.path != b.path
    assert len(list((tmp_path / "out").iterdir())) == 2


def test_markdown_table_escapes_pipes():
    table = markdown_table([{"k": "a|b", "v": 1.0}, {"k": "c", "v": None}], ["k", "v"])
    lines = table.splitlines()
    assert lines[0] == "| k | v |"
    assert "a\\|b" in lines[2]
    assert lines[3].endswith("|  |")  # None renders as an empty cell
    for line in lines[2:]:
        assert line.replace("\\|", "").count("|") == 3  # no stray column separators
    assert markdown_table([]) == "_(no rows)_"


def test_report_separates_measured_and_estimates(tmp_path):
    with _run(tmp_path) as run:
        run.metrics["latency_ms"] = 3.25
        run.estimates["flops_per_token"] = 987654
        run.add_section("Custom", "body text")
    report = (run.path / "report.md").read_text()
    measured_at = report.index("## Measured metrics")
    estimates_at = report.index("## Theoretical estimates")
    assert measured_at < estimates_at
    measured, estimates = report[measured_at:estimates_at], report[estimates_at:]
    assert "latency_ms" in measured and "flops_per_token" not in measured
    assert "flops_per_token" in estimates.split("## Environment")[0] and "latency_ms" not in estimates
    assert "## Custom" in report and "body text" in report
