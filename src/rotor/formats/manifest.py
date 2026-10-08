"""Conversion manifest: provenance, per-layer specs, fit metrics, and byte accounting."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rotor.formats.spec import FORMAT_NAME, FORMAT_VERSION
from rotor.math.cascade import CascadeSpec


@dataclass
class BaseProvenance:
    hf_id: str | None
    revision: str | None
    path: str
    file_sha256: dict[str, str] = field(default_factory=dict)  # relative file name -> sha256
    tied_word_embeddings: bool | None = None
    architectures: list[str] = field(default_factory=list)


@dataclass
class LayerEntry:
    module: str  # HF module path in the text-only CausalLM, e.g. model.layers.0.mlp.up_proj
    source_tensor: str  # checkpoint tensor name, e.g. model.language_model.layers.0.mlp.up_proj.weight
    family: str
    spec: CascadeSpec
    storage_dtype: str
    has_bias: bool
    dense_params: int
    structured_params: int
    dense_bytes: int
    structured_bytes: int
    metrics: dict = field(default_factory=dict)  # measured fit metrics (rel. weight error, activation error, ...)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LayerEntry:
        d = dict(d)
        d["spec"] = CascadeSpec.from_dict(d["spec"])
        return cls(**d)


def environment_record() -> dict:
    import torch

    try:
        import transformers

        tf_version = transformers.__version__
    except ImportError:  # pragma: no cover
        tf_version = None
    try:
        git_rev = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=Path(__file__).parent
            ).stdout.strip()
            or None
        )
    except OSError:  # pragma: no cover
        git_rev = None
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": tf_version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "git_commit": git_rev,
    }


@dataclass
class Manifest:
    base: BaseProvenance
    layers: list[LayerEntry]
    conversion_config: dict
    environment: dict = field(default_factory=environment_record)
    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    notes: list[str] = field(default_factory=list)

    def totals(self) -> dict:
        dense = sum(entry.dense_bytes for entry in self.layers)
        struct = sum(entry.structured_bytes for entry in self.layers)
        return {
            "num_layers": len(self.layers),
            "dense_bytes": dense,
            "structured_bytes": struct,
            "byte_ratio": struct / dense if dense else None,
        }

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "base": asdict(self.base),
            "conversion_config": self.conversion_config,
            "environment": self.environment,
            "totals": self.totals(),
            "notes": self.notes,
            "layers": [entry.to_dict() for entry in self.layers],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        if d.get("format") != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} manifest: format={d.get('format')!r}")
        if int(d.get("format_version", -1)) > FORMAT_VERSION:
            raise ValueError(f"manifest format_version {d['format_version']} is newer than supported {FORMAT_VERSION}")
        return cls(
            base=BaseProvenance(**d["base"]),
            layers=[LayerEntry.from_dict(x) for x in d["layers"]],
            conversion_config=d.get("conversion_config", {}),
            environment=d.get("environment", {}),
            format=d["format"],
            format_version=int(d["format_version"]),
            notes=list(d.get("notes", [])),
        )

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        return cls.from_dict(json.loads(s))
