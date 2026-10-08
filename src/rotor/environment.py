"""Actual hardware/software capability detection (never assumed; every field is probed on this host).

``detect_environment()`` returns a JSON-serializable dict recorded in every manifest and run directory.
Probes are best-effort and side-effect free: a missing tool yields ``None``/``False`` plus the probe error,
never an exception. ``ResourceMonitor`` records wall time, CPU time, peak RSS, and (CUDA only) peak device memory
for a code region.
"""

from __future__ import annotations

import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_TOOLS = (
    "g++",
    "clang++",
    "cmake",
    "ninja",
    "nvcc",
    "hipcc",
    "vulkaninfo",
    "glslc",
    "pdflatex",
    "lualatex",
    "xelatex",
    "latexmk",
    "bibtex",
    "biber",
    "git",
)
_ISA_FLAGS = ("avx2", "fma", "avx512f", "avx512_bf16", "avx512_vnni", "avx512_fp16", "amx_bf16")


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def _first_line(text: str | None) -> str | None:
    return text.strip().splitlines()[0].strip() if text and text.strip() else None


def cpu_info() -> dict:
    model, flags = None, set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if model is None and line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
            elif not flags and line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return {
        "model": model or platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "isa": {f: f in flags for f in _ISA_FLAGS},
        "arch": platform.machine(),
    }


def memory_info() -> dict:
    out: dict = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = int(v.split()[0]) * 1024
    except OSError:
        return {}
    return {
        "ram_total_bytes": out.get("MemTotal"),
        "ram_available_bytes": out.get("MemAvailable"),
        "swap_total_bytes": out.get("SwapTotal"),
        "swap_used_bytes": (out["SwapTotal"] - out["SwapFree"]) if "SwapTotal" in out else None,
    }


def pci_gpus() -> list[dict]:
    """Display controllers from sysfs (vendor/device ids, kernel driver, VRAM where the driver exposes it)."""
    vendors = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
    gpus = []
    base = Path("/sys/bus/pci/devices")
    if not base.is_dir():
        return gpus
    for dev in sorted(base.iterdir()):
        try:
            cls = (dev / "class").read_text().strip()
        except OSError:
            continue
        if not cls.startswith("0x03"):
            continue
        vendor = (dev / "vendor").read_text().strip()
        driver = (dev / "driver").resolve().name if (dev / "driver").exists() else None
        vram = dev / "mem_info_vram_total"
        gpus.append(
            {
                "pci": dev.name,
                "vendor": vendors.get(vendor, vendor),
                "device_id": (dev / "device").read_text().strip(),
                "driver": driver,
                "vram_bytes": int(vram.read_text()) if vram.exists() else None,
            }
        )
    names = _run(["lspci", "-mm"]) or ""
    for g in gpus:
        short = g["pci"].split(":", 1)[1] if ":" in g["pci"] else g["pci"]
        m = re.search(rf'^{re.escape(short)} "[^"]*" "[^"]*" "([^"]*)"', names, re.M)
        g["name"] = m.group(1) if m else None
    return gpus


def nvidia_info() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return {"nvidia_smi_ok": out is not None, "devices": [s.strip() for s in out.splitlines()] if out else []}


def vulkan_info() -> dict:
    icds = []
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        if Path(d).is_dir():
            icds += sorted(p.name for p in Path(d).glob("*.json"))
    summary = _run(["vulkaninfo", "--summary"], timeout=20)
    devices = re.findall(r"deviceName\s*=\s*(.+)", summary or "")
    return {
        "loader_present": any(
            Path(p).exists() for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/libvulkan.so.1")
        ),
        "installed_icds": icds,
        "vulkaninfo_available": summary is not None,
        "devices": [d.strip() for d in devices],
    }


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    cuda = torch.cuda.is_available()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "hip_build": getattr(torch.version, "hip", None),
        "cuda_available": cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "num_threads": torch.get_num_threads(),
        "mkldnn": torch.backends.mkldnn.is_available(),
    }


def tool_versions() -> dict:
    out = {}
    for t in _TOOLS:
        path = shutil.which(t)
        if not path:
            out[t] = None
            continue
        flag = "--summary" if t == "vulkaninfo" else "--version"
        text = _run([path, flag]) if t != "vulkaninfo" else "present"
        line = _first_line(text)
        if t == "nvcc" and text:
            line = next((ln for ln in text.splitlines() if "release" in ln), line)
        out[t] = {"path": path, "version": line}
    return out


def storage_info(paths: dict[str, Path] | None = None) -> dict:
    if paths is None:
        from rotor.storage import REPO_ROOT, load_storage

        paths = {"repo": REPO_ROOT}
        try:
            paths["nas_root"] = load_storage().nas_root
        except Exception as e:  # noqa: BLE001 - config errors are reported, not raised, by an audit
            paths["nas_root_error"] = Path(str(e))
    out = {}
    for k, p in paths.items():
        try:
            u = shutil.disk_usage(p)
            out[k] = {"path": str(p), "total_bytes": u.total, "free_bytes": u.free, "available": True}
        except OSError as e:
            out[k] = {"path": str(p), "available": False, "error": str(e)}
    return out


def library_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "safetensors", "transformers", "huggingface_hub", "yaml"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except ImportError:
            out[mod] = None
    return out


def git_commit() -> dict:
    from rotor.storage import REPO_ROOT

    rev = _first_line(_run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]))
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    return {"commit": rev, "dirty": bool(dirty.strip()) if dirty is not None else None}


def detect_environment(include_storage: bool = True) -> dict:
    """Full probe. Takes ~0.1-1 s (spawns a few short subprocesses)."""
    env = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpus_pci": pci_gpus(),
        "nvidia": nvidia_info(),
        "vulkan": vulkan_info(),
        "torch": torch_info(),
        "libraries": library_versions(),
        "tools": tool_versions(),
        "git": git_commit(),
    }
    if include_storage:
        env["storage"] = storage_info()
    return env


def capability_summary(env: dict) -> list[str]:
    """Human-readable warnings for benchmark validity (e.g. CPU fallback, swap pressure)."""
    notes = []
    t = env.get("torch", {})
    if not t.get("cuda_available"):
        notes.append("torch has no usable CUDA/HIP device: GPU kernel performance cannot be measured on this host")
    if any(g["vendor"] == "NVIDIA" for g in env.get("gpus_pci", [])) and not env.get("nvidia", {}).get("nvidia_smi_ok"):
        notes.append("NVIDIA GPU present on PCI bus but nvidia-smi fails (driver not loaded)")
    if env.get("tools", {}).get("nvcc") and not any(g["vendor"] == "NVIDIA" for g in env.get("gpus_pci", [])):
        notes.append("nvcc is installed but no NVIDIA GPU is on the PCI bus: CUDA code can compile but not run")
    mem = env.get("memory", {})
    if mem.get("swap_used_bytes"):
        notes.append(f"swap in use ({mem['swap_used_bytes'] / 2**30:.1f} GiB): memory-heavy timings may be distorted")
    if env.get("git", {}).get("dirty"):
        notes.append("git working tree is dirty: results are not tied to a clean commit")
    return notes


class ResourceMonitor:
    """Context manager measuring wall/CPU time and peak memory of a region (measured values only).

    Peak RSS comes from ``getrusage`` (process-lifetime high-water mark, so ``peak_rss_bytes`` is an upper bound
    for the region); CUDA peak memory is reset at entry and read at exit when CUDA is available.
    """

    def __enter__(self) -> ResourceMonitor:
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        self._cuda = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._cuda = True
        except ImportError:
            pass
        return self

    def __exit__(self, *exc) -> None:
        self.wall_seconds = time.perf_counter() - self._t0
        self.cpu_seconds = time.process_time() - self._c0
        self.peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.peak_device_bytes = None
        if self._cuda:
            import torch

            self.peak_device_bytes = int(torch.cuda.max_memory_allocated())

    def as_dict(self) -> dict:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_device_bytes": self.peak_device_bytes,
            "measured": True,
        }
