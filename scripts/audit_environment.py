#!/usr/bin/env python
"""Audit this host (CPU/ISA, RAM/swap, GPUs, CUDA/Vulkan, torch, compilers, LaTeX, storage, git) and the NAS layout.

Read-only: the NAS root is only listed (top-level entries), never written to. Exit status is 0 whenever the audit
itself completes; missing capabilities are reported, not treated as errors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rotor.environment import capability_summary, detect_environment

GIB = 2**30


def _gib(n: int | None) -> str:
    return "n/a" if n is None else f"{n / GIB:.1f} GiB"


def _yn(v: object) -> str:
    return "yes" if v else "no"


def nas_listing(storage) -> dict:
    """Top-level NAS root contents (read-only) plus the existence of each configured model's base path."""
    from rotor.storage import load_models

    out: dict = {"nas_root": str(storage.nas_root), "available": storage.nas_available(), "entries": []}
    if out["available"]:
        try:
            for p in sorted(storage.nas_root.iterdir(), key=lambda q: q.name.lower()):
                ro = any(os.path.abspath(p) == os.path.abspath(r) for r in storage.preexisting_read_only)
                out["entries"].append(
                    {
                        "name": p.name,
                        "kind": "dir" if p.is_dir() else "file",
                        "pre_existing_read_only": ro,
                        "project_managed": os.path.abspath(p)
                        in {
                            os.path.abspath(storage.base_root),
                            os.path.abspath(storage.derived_root),
                        },
                    }
                )
        except OSError as e:
            out["error"] = str(e)
    models = []
    for key, entry in load_models().get("models", {}).items():
        try:
            base = storage.base_path(entry["local_path"])
            exists = base.is_dir()
            has_weights = exists and any(base.glob("*.safetensors"))
            models.append(
                {
                    "key": key,
                    "hf_id": entry["hf_id"],
                    "revision": entry["revision"],
                    "base_path": str(base),
                    "base_exists": exists,
                    "has_safetensors": bool(has_weights),
                }
            )
        except Exception as e:  # noqa: BLE001 - audit reports configuration problems instead of aborting
            models.append({"key": key, "hf_id": entry.get("hf_id"), "error": str(e)})
    out["models"] = models
    return out


def print_summary(env: dict, nas: dict, warnings: list[str]) -> None:
    cpu, mem = env.get("cpu", {}), env.get("memory", {})
    print(f"host: {env.get('hostname')}  ({env.get('platform')})")
    print(f"\n== CPU ==\n{cpu.get('model')}  logical CPUs: {cpu.get('logical_cpus')}  arch: {cpu.get('arch')}")
    isa = cpu.get("isa", {})
    print("ISA: " + ", ".join(f"{k}={_yn(v)}" for k, v in isa.items()))
    print("\n== Memory ==")
    print(
        f"RAM total {_gib(mem.get('ram_total_bytes'))}, available {_gib(mem.get('ram_available_bytes'))}; "
        f"swap total {_gib(mem.get('swap_total_bytes'))}, used {_gib(mem.get('swap_used_bytes'))}"
    )
    print("\n== GPUs on the PCI bus ==")
    for g in env.get("gpus_pci", []) or []:
        print(
            f"{g.get('pci')}  {g.get('vendor')}  {g.get('name') or g.get('device_id')}  "
            f"driver={g.get('driver')}  VRAM={_gib(g.get('vram_bytes'))}"
        )
    if not env.get("gpus_pci"):
        print("none detected")
    nv = env.get("nvidia", {})
    print("\n== NVIDIA / CUDA ==")
    print(f"nvidia-smi ok: {_yn(nv.get('nvidia_smi_ok'))}  devices: {nv.get('devices') or 'none'}")
    tools = env.get("tools", {})
    nvcc = tools.get("nvcc")
    print(f"nvcc: {nvcc['version'] if nvcc else 'not installed'}")
    print(f"hipcc: {tools['hipcc']['version'] if tools.get('hipcc') else 'not installed'}")
    vk = env.get("vulkan", {})
    print("\n== Vulkan ==")
    print(f"loader present: {_yn(vk.get('loader_present'))}  vulkaninfo: {_yn(vk.get('vulkaninfo_available'))}")
    print(f"ICDs: {', '.join(vk.get('installed_icds', [])) or 'none'}")
    print(f"devices: {', '.join(vk.get('devices', [])) or 'none enumerated'}")
    t = env.get("torch", {})
    print("\n== torch ==")
    if t.get("available"):
        print(
            f"torch {t.get('version')}  cuda build={t.get('cuda_build')}  hip build={t.get('hip_build')}  "
            f"cuda_available={_yn(t.get('cuda_available'))}  devices={t.get('devices') or 'none'}  "
            f"threads={t.get('num_threads')}  mkldnn={_yn(t.get('mkldnn'))}"
        )
    else:
        print("torch not importable")
    print("libraries: " + ", ".join(f"{k}={v}" for k, v in env.get("libraries", {}).items()))
    print("\n== Compilers and tools ==")
    compilers = ("g++", "clang++", "cmake", "ninja", "glslc", "vulkaninfo")
    for name in compilers:
        info = tools.get(name)
        print(f"{name:<11}{info['version'] if info else 'not installed'}")
    print("\n== LaTeX ==")
    for name in ("pdflatex", "lualatex", "xelatex", "latexmk", "bibtex", "biber"):
        info = tools.get(name)
        print(f"{name:<11}{info['version'] if info else 'not installed'}")
    print("\n== Storage free space ==")
    for k, v in env.get("storage", {}).items():
        if v.get("available"):
            print(f"{k:<10}{v['path']}: {_gib(v['free_bytes'])} free of {_gib(v['total_bytes'])}")
        else:
            print(f"{k:<10}{v['path']}: unavailable ({v.get('error')})")
    git = env.get("git", {})
    print(f"\n== git ==\ncommit {git.get('commit')}  dirty={git.get('dirty')}")

    print(f"\n== NAS root {nas['nas_root']} (read-only listing) ==")
    if not nas["available"]:
        print("NAS root is not available (not mounted?)")
    for e in nas["entries"]:
        tag = " [pre-existing, read-only to this project]" if e["pre_existing_read_only"] else ""
        tag += " [project-managed]" if e["project_managed"] else ""
        print(f"  {e['kind']:<5}{e['name']}{tag}")
    if nas.get("error"):
        print(f"  listing error: {nas['error']}")
    print("\n== Configured models (base checkpoints) ==")
    for m in nas["models"]:
        if "error" in m:
            print(f"{m['key']:<16}error: {m['error']}")
            continue
        state = (
            "present (safetensors found)"
            if m["has_safetensors"]
            else ("directory exists but no *.safetensors" if m["base_exists"] else "NOT DOWNLOADED")
        )
        print(f"{m['key']:<16}{m['hf_id']} @ {m['revision'][:12]} -> {m['base_path']}: {state}")

    print("\n== Validity warnings ==")
    for w in warnings or ["none"]:
        print(f"- {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, dest="json_out", metavar="OUT", help="write the raw audit dict as JSON")
    ap.add_argument(
        "--report", action="store_true", help="publish an experiment run (kind 'audit') under reports/experiments/"
    )
    args = ap.parse_args(argv)

    from rotor.storage import StorageError, load_storage

    env = detect_environment()
    warnings = capability_summary(env)
    try:
        storage = load_storage()
        nas = nas_listing(storage)
    except (StorageError, OSError, KeyError) as e:
        nas = {"nas_root": None, "available": False, "entries": [], "models": [], "error": str(e)}
        warnings.append(f"storage configuration error: {e}")
    audit = {"environment": env, "warnings": warnings, "nas": nas}
    print_summary(env, nas, warnings)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(audit, indent=2, default=str) + "\n")
        print(f"\nwrote {args.json_out}")
    if args.report:
        from rotor.reporting import ExperimentRun, markdown_table

        with ExperimentRun(
            "audit",
            model="host",
            method="environment-audit",
            target="na",
            config={"json": str(args.json_out) if args.json_out else None},
            environment=env,
        ) as run:
            run.log("environment audit")
            run.metrics["warnings"] = warnings
            run.metrics["nas"] = nas
            run.metrics["cpu"] = env.get("cpu")
            run.metrics["memory"] = env.get("memory")
            run.metrics["gpus_pci"] = env.get("gpus_pci")
            run.metrics["storage"] = env.get("storage")
            run.add_section(
                "GPUs on the PCI bus (probed)",
                markdown_table(env.get("gpus_pci") or [], ["pci", "vendor", "name", "driver", "vram_bytes"]),
            )
            run.add_section(
                "Compilers and tools (probed)",
                markdown_table(
                    [
                        {"tool": k, "version": v["version"] if v else "not installed"}
                        for k, v in env.get("tools", {}).items()
                    ]
                ),
            )
            run.add_section(
                "Configured base checkpoints (probed, read-only)",
                markdown_table(nas["models"], ["key", "hf_id", "base_path", "base_exists", "has_safetensors"]),
            )
            run.add_section(
                "NAS root top-level entries (read-only listing)",
                markdown_table(nas["entries"], ["name", "kind", "pre_existing_read_only", "project_managed"]),
            )
            run.summary = "Probed host capabilities; every value is a measurement of this host, nothing is estimated."
        print(f"run record {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
