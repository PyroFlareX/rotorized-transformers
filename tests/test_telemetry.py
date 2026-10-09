"""Process/host runtime telemetry JSONL and summary contracts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rotor.telemetry import RuntimeTelemetry


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_runtime_telemetry_writes_samples_and_phase_markers(tmp_path):
    path = tmp_path / "run.jsonl"
    with RuntimeTelemetry(path, interval_seconds=0.02) as telemetry:
        telemetry.set_phase("generation", request_id="prompt-001")
        time.sleep(0.05)
    summary = telemetry.as_dict()

    records = _records(path)
    samples = [record for record in records if record["record_type"] == "sample"]
    assert len(samples) >= 3  # entry, at least one periodic sample, exit
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["timestamp_utc"].endswith("Z") for record in records)
    assert all(record["elapsed_seconds"] >= 0 for record in records)
    assert any(
        record["record_type"] == "phase"
        and record["phase"] == "generation"
        and record["request_id"] == "prompt-001"
        for record in records
    )
    sample = samples[-1]
    assert sample["phase"] == "generation"
    assert sample["request_id"] == "prompt-001"
    assert sample["process_cpu_time_seconds"] >= 0
    assert sample["process_rss_bytes"] is None or sample["process_rss_bytes"] > 0
    assert set(sample["host_memory"]) == {
        "available_ram_bytes", "available_swap_bytes", "total_swap_bytes", "used_swap_bytes"
    }
    assert isinstance(sample["devices"], list)
    assert summary["sample_count"] == len(samples)
    assert summary["duration_seconds"] >= 0.05
    assert summary["current_process_rss_bytes"] == sample["process_rss_bytes"]
    assert summary["peak_sampled_process_rss_bytes"] is None or summary["peak_sampled_process_rss_bytes"] > 0
    assert summary["sampling_interval_seconds"] == 0.02
    assert summary["phase"] == "generation"
    assert summary["request_id"] == "prompt-001"
    json.dumps(summary)


def test_runtime_telemetry_preserves_original_exception(tmp_path):
    path = tmp_path / "failed.jsonl"
    with pytest.raises(ValueError, match="inference failed"):
        with RuntimeTelemetry(path, interval_seconds=0.02) as telemetry:
            telemetry.set_phase("generation", request_id="prompt-err")
            raise ValueError("inference failed")
    records = _records(path)
    assert any(record["record_type"] == "sample" for record in records)
    assert records[-1]["record_type"] == "run_exception"
    assert records[-1]["exception_type"] == "ValueError"
    assert records[-1]["exception_message"] == "inference failed"
    assert "inference failed" in records[-1]["traceback"]


def test_runtime_telemetry_never_overwrites_a_listing(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("preserve me\n")
    with pytest.raises(FileExistsError):
        with RuntimeTelemetry(path):
            pass
    assert path.read_text() == "preserve me\n"


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_runtime_telemetry_requires_positive_finite_interval(tmp_path, interval):
    with pytest.raises(ValueError, match="interval_seconds"):
        RuntimeTelemetry(tmp_path / "invalid.jsonl", interval_seconds=interval)
