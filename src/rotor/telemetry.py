"""Low-rate, process-scoped runtime telemetry written as flush-on-sample JSONL.

Process and host measurements use procfs; AMD devices use amdgpu sysfs. Optional CUDA allocator metrics are sampled
only when an already-imported PyTorch CUDA build reports CUDA available. Sampling never launches a subprocess.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import traceback as traceback_module
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _process_rss_bytes() -> int | None:
    """Return current resident pages from procfs, not lifetime peak RSS."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _process_io() -> dict[str, int | None]:
    values: dict[str, int | None] = {"read_bytes": None, "write_bytes": None}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key == "read_bytes":
                values["read_bytes"] = int(raw.strip())
            elif key == "write_bytes":
                values["write_bytes"] = int(raw.strip())
    except (OSError, ValueError):
        pass
    return values


def _host_memory() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(raw.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    swap_free = values.get("SwapFree")
    swap_total = values.get("SwapTotal")
    return {
        "available_ram_bytes": values.get("MemAvailable"),
        "available_swap_bytes": swap_free,
        "total_swap_bytes": swap_total,
        "used_swap_bytes": swap_total - swap_free if swap_total is not None and swap_free is not None else None,
    }


def _sysfs_amd_devices() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    try:
        cards = sorted(Path("/sys/class/drm").glob("card[0-9]*"))
    except OSError:
        return devices
    for card in cards:
        if not re.fullmatch(r"card\d+", card.name):
            continue
        dev = card / "device"
        if _read_text(dev / "vendor") != "0x1002":
            continue
        resolved = dev.resolve()
        match = re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$", str(resolved))
        driver = (dev / "driver").resolve().name if (dev / "driver").exists() else None
        temperatures: dict[str, float] = {}
        power_w: float | None = None
        try:
            hwmon_dirs = sorted((dev / "hwmon").glob("hwmon*"))
        except OSError:
            hwmon_dirs = []
        for hwmon in hwmon_dirs:
            try:
                temp_inputs = sorted(hwmon.glob("temp*_input"))
            except OSError:
                temp_inputs = []
            for path in temp_inputs:
                raw = _read_int(path)
                if raw is not None:
                    stem = path.name.removesuffix("_input")
                    label = _read_text(hwmon / f"{stem}_label") or stem
                    temperatures[label] = raw / 1000.0
            if power_w is None:
                for filename in ("power1_average", "power1_input"):
                    raw_power = _read_int(hwmon / filename)
                    if raw_power is not None:
                        power_w = raw_power / 1_000_000.0
                        break
        devices.append(
            {
                "backend": "amd_sysfs",
                "card": card.name,
                "pci_address": match.group(0) if match else None,
                "name": _read_text(dev / "product_name") or _read_text(dev / "product"),
                "driver": driver,
                "gpu_busy_percent": _read_int(dev / "gpu_busy_percent"),
                "vram_used_bytes": _read_int(dev / "mem_info_vram_used"),
                "vram_total_bytes": _read_int(dev / "mem_info_vram_total"),
                "gtt_used_bytes": _read_int(dev / "mem_info_gtt_used"),
                "gtt_total_bytes": _read_int(dev / "mem_info_gtt_total"),
                "temperature_c": temperatures or None,
                "power_w": power_w,
            }
        )
    return devices


def _cuda_devices() -> list[dict[str, Any]]:
    """Read allocator counters without importing torch just to sample."""
    torch = sys.modules.get("torch")
    if torch is None:
        return []
    try:
        if torch.version.cuda is None or not torch.cuda.is_available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            try:
                devices.append(
                    {
                        "backend": "cuda",
                        "device_index": index,
                        "name": torch.cuda.get_device_name(index),
                        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    }
                )
            except Exception:  # noqa: BLE001 - device may disappear while a run is active
                devices.append(
                    {
                        "backend": "cuda",
                        "device_index": index,
                        "name": None,
                        "allocated_bytes": None,
                        "reserved_bytes": None,
                    }
                )
        return devices
    except Exception:  # noqa: BLE001 - unavailable/broken CUDA is represented by no CUDA samples
        return []


def _logical_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


class RuntimeTelemetry:
    """Asynchronously sample process, host, and available device resources to an exclusive JSONL file.

    Emits an initial sample, periodic samples on a daemon thread, and a final sample on exit. Each line is flushed
    immediately. Telemetry failures are retained in :meth:`as_dict` and do not replace errors from measured work.
    """

    def __init__(self, path: Path, interval_seconds: float = 1.0):
        try:
            interval = float(interval_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("interval_seconds must be a finite number greater than zero") from exc
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval_seconds must be a finite number greater than zero")
        try:
            self.path = Path(path)
        except TypeError as exc:
            raise TypeError("path must be path-like") from exc
        self.interval_seconds = interval
        self._file = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._entered = False
        self._closed = False
        self._start_monotonic = 0.0
        self._start_cpu = 0.0
        self._last_sample_elapsed: float | None = None
        self._last_sample_cpu: float | None = None
        self._last_sample: dict[str, Any] | None = None
        self._sample_count = 0
        self._peak_rss_bytes: int | None = None
        self._peak_cpu_utilization: float | None = None
        self._peak_device_memory: dict[tuple[str, str], dict[str, Any]] = {}
        self._phase = "unattributed"
        self._request_id: str | None = None
        self._sampling_error: str | None = None
        self._cpu_count = _logical_cpu_count()
        self._end_elapsed: float | None = None
        self._end_cpu_seconds: float | None = None

    def __enter__(self) -> RuntimeTelemetry:
        if self._entered:
            raise RuntimeError("RuntimeTelemetry instances are single-use")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._entered = True
        self._start_monotonic = time.perf_counter()
        self._start_cpu = time.process_time()
        try:
            self._emit_sample()
        except Exception as sample_exc:  # noqa: BLE001 - telemetry failures must not prevent measured work
            self._record_sampling_error(sample_exc)
        thread = threading.Thread(target=self._sample_loop, name="rotor-runtime-telemetry", daemon=True)
        try:
            thread.start()
            self._thread = thread
        except Exception as thread_exc:  # noqa: BLE001 - report it at exit without preventing measured work
            self._record_sampling_error(thread_exc)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._entered or self._closed:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._emit_sample()
        except Exception as sample_exc:  # noqa: BLE001 - don't replace a measured-work failure
            self._record_sampling_error(sample_exc)
        self._end_elapsed = max(0.0, time.perf_counter() - self._start_monotonic)
        self._end_cpu_seconds = max(0.0, time.process_time() - self._start_cpu)
        if exc is not None:
            self._write_exception_record(exc_type, exc, traceback)
        try:
            self._file.flush()
        except Exception as file_exc:  # noqa: BLE001 - retain error without masking measured work
            self._record_sampling_error(file_exc)
        try:
            self._file.close()
        except Exception as file_exc:  # noqa: BLE001 - retain error without masking measured work
            self._record_sampling_error(file_exc)
        finally:
            self._closed = True
        if exc is None and self._sampling_error is not None:
            raise RuntimeError(f"RuntimeTelemetry sampling failed: {self._sampling_error}")
        return False

    def set_phase(self, phase: str, request_id: str | None = None) -> None:
        """Attribute subsequent samples to a named phase and optional request; emit a phase marker."""
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        if request_id is not None and not isinstance(request_id, str):
            raise TypeError("request_id must be a string or None")
        with self._lock:
            if not self._entered or self._closed:
                raise RuntimeError("set_phase() requires an active RuntimeTelemetry context")
            if phase == self._phase and request_id == self._request_id:
                return
            self._phase, self._request_id = phase, request_id
            try:
                self._write_record(self._base_record("phase"))
            except Exception as phase_exc:  # noqa: BLE001 - phase logging must not interrupt measured work
                self._record_sampling_error(phase_exc)

    def as_dict(self) -> dict[str, Any]:
        """Return the measured summary, usually called after context exit."""
        with self._lock:
            sample = self._last_sample or {}
            elapsed = self._end_elapsed
            cpu_seconds = self._end_cpu_seconds
            if elapsed is None:
                elapsed = max(0.0, time.perf_counter() - self._start_monotonic) if self._entered else 0.0
            if cpu_seconds is None:
                cpu_seconds = max(0.0, time.process_time() - self._start_cpu) if self._entered else 0.0
            return {
                "schema_version": _SCHEMA_VERSION,
                "measured": True,
                "samples_path": str(self.path),
                "sampling_interval_seconds": self.interval_seconds,
                "sample_count": self._sample_count,
                "duration_seconds": elapsed,
                "cpu_logical_count": self._cpu_count,
                "process_cpu_time_seconds": cpu_seconds,
                "process_cpu_utilization_percent": self._cpu_utilization(cpu_seconds, elapsed),
                "current_process_rss_bytes": sample.get("process_rss_bytes"),
                "peak_sampled_process_rss_bytes": self._peak_rss_bytes,
                "peak_sampled_cpu_utilization_percent": self._peak_cpu_utilization,
                "devices": sample.get("devices", []),
                "peak_sampled_device_memory": list(self._peak_device_memory.values()),
                "host_available_ram_bytes": sample.get("host_memory", {}).get("available_ram_bytes"),
                "host_available_swap_bytes": sample.get("host_memory", {}).get("available_swap_bytes"),
                "host_total_swap_bytes": sample.get("host_memory", {}).get("total_swap_bytes"),
                "process_read_bytes": sample.get("process_io", {}).get("read_bytes"),
                "process_write_bytes": sample.get("process_io", {}).get("write_bytes"),
                "phase": self._phase,
                "request_id": self._request_id,
                "sampling_error": self._sampling_error,
            }

    def _sample_loop(self) -> None:
        next_sample = self._start_monotonic + self.interval_seconds
        while not self._stop.is_set():
            if self._stop.wait(max(0.0, next_sample - time.perf_counter())):
                break
            try:
                self._emit_sample()
            except Exception as sample_exc:  # noqa: BLE001 - measured work owns its own failures
                self._record_sampling_error(sample_exc)
                break
            next_sample += self.interval_seconds
            now = time.perf_counter()
            if next_sample <= now:
                next_sample = now + self.interval_seconds

    def _emit_sample(self) -> None:
        with self._lock:
            cpu_time = time.process_time()
            rss = _process_rss_bytes()
            amd_devices = _sysfs_amd_devices()
            cuda_devices = _cuda_devices()
            elapsed = max(0.0, time.perf_counter() - self._start_monotonic)
            wall_delta = elapsed - self._last_sample_elapsed if self._last_sample_elapsed is not None else None
            cpu_delta = cpu_time - self._last_sample_cpu if self._last_sample_cpu is not None else None
            cpu_utilization = (
                100.0 * cpu_delta / wall_delta
                if cpu_delta is not None and wall_delta is not None and wall_delta > 0
                else None
            )
            sample = {
                **self._base_record("sample", elapsed=elapsed),
                "sample_index": self._sample_count,
                "sampling_interval_seconds": self.interval_seconds,
                "process_cpu_time_seconds": cpu_time,
                "process_cpu_utilization_percent": cpu_utilization,
                "process_rss_bytes": rss,
                "host_memory": _host_memory(),
                "process_io": _process_io(),
                "devices": amd_devices + cuda_devices,
                "device_sensors": {
                    "amd_sysfs_available": bool(amd_devices),
                    "cuda_allocator_available": bool(cuda_devices),
                },
            }
            self._last_sample_elapsed = elapsed
            self._last_sample_cpu = cpu_time
            self._last_sample = sample
            self._write_record(sample)
            for device in sample["devices"]:
                backend = device["backend"]
                identity = str(device.get("pci_address", device.get("device_index", device.get("card", "unknown"))))
                peak = self._peak_device_memory.setdefault(
                    (backend, identity), {"backend": backend, "device_id": identity, "name": device.get("name")}
                )
                for field in ("vram_used_bytes", "gtt_used_bytes", "allocated_bytes", "reserved_bytes"):
                    value = device.get(field)
                    if isinstance(value, int):
                        peak[field] = max(peak.get(field, value), value)
            self._sample_count += 1
            if rss is not None:
                self._peak_rss_bytes = rss if self._peak_rss_bytes is None else max(self._peak_rss_bytes, rss)
            if cpu_utilization is not None:
                self._peak_cpu_utilization = (
                    cpu_utilization
                    if self._peak_cpu_utilization is None
                    else max(self._peak_cpu_utilization, cpu_utilization)
                )

    def _base_record(self, record_type: str, elapsed: float | None = None) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "record_type": record_type,
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": max(0.0, time.perf_counter() - self._start_monotonic) if elapsed is None else elapsed,
            "phase": self._phase,
            "request_id": self._request_id,
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._file.flush()

    def _write_exception_record(self, exc_type, exc, traceback) -> None:
        with self._lock:
            try:
                self._write_record(
                    {
                        **self._base_record("run_exception"),
                        "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                        "exception_message": str(exc),
                        "traceback": "".join(traceback_module.format_exception(exc_type, exc, traceback))[-16384:],
                    }
                )
            except Exception as record_exc:  # noqa: BLE001 - do not mask the original exception
                self._record_sampling_error(record_exc)

    def _record_sampling_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._sampling_error is not None:
                return
            self._sampling_error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
            if self._file is not None and not self._closed:
                try:
                    self._write_record({**self._base_record("telemetry_error"), "error": self._sampling_error})
                except Exception:  # noqa: BLE001 - no functioning output remains to record this further
                    pass

    @staticmethod
    def _cpu_utilization(cpu_seconds: float, wall_seconds: float) -> float | None:
        if wall_seconds <= 0:
            return None
        # 100% is one fully occupied logical CPU; multithreaded processes may exceed 100%.
        return 100.0 * cpu_seconds / wall_seconds
