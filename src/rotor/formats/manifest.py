"""Conversion manifest: provenance, per-layer specs, fit metrics, and byte accounting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from rotor.environment import detect_environment
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
    # learned scalar counts of this layer by kind: angles / scales (all diagonals) / residual / bias.
    # Empty on manifests written before this field existed (``Manifest.totals`` then derives it from ``spec``).
    learned_scalars: dict = field(default_factory=dict)
    # checkpoint dtype string of the original dense weight (e.g. "BF16"); empty on older manifests.
    dense_dtype: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LayerEntry:
        d = dict(d)
        d["spec"] = CascadeSpec.from_dict(d["spec"])
        return cls(**d)


@dataclass
class Manifest:
    base: BaseProvenance
    layers: list[LayerEntry]
    conversion_config: dict
    environment: dict = field(default_factory=lambda: detect_environment(include_storage=False))
    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    notes: list[str] = field(default_factory=list)

    # File-level byte accounting measured by ``write_structured`` after the weights file was saved:
    # file_bytes, header_bytes, manifest_bytes, padding_bytes, tensor_bytes, num_tensors. Empty before writing.
    serialized: dict = field(default_factory=dict)

    def learned_scalar_count(self) -> dict:
        counts = {"angles": 0, "scales": 0, "residual": 0, "bias": 0}
        for entry in self.layers:
            if entry.learned_scalars:
                for kind in counts:
                    counts[kind] += int(entry.learned_scalars.get(kind, 0))
            else:  # legacy manifest: derive from the cascade spec (bias count is not recorded there)
                pb = entry.spec.param_breakdown()
                counts["angles"] += pb["angles"]
                counts["scales"] += pb["diagonals"]
                counts["residual"] += pb["residual"]
        counts["total"] = sum(counts.values())
        return counts

    def totals(self) -> dict:
        dense = sum(entry.dense_bytes for entry in self.layers)
        struct = sum(entry.structured_bytes for entry in self.layers)
        out = {
            "num_layers": len(self.layers),
            "dense_bytes": dense,
            "structured_bytes": struct,
            "byte_ratio": struct / dense if dense else None,
            "learned_scalar_count": self.learned_scalar_count(),
            "index_bytes": 0,
            "index_bytes_note": "0: topologies are deterministic functions of (kind, seed, width); no index is stored",
            "quantization_metadata_bytes": 0,
            "quantization_metadata_note": "0: float storage dtypes only; no scales/zero-points/codebooks are stored",
        }
        for key in ("file_bytes", "header_bytes", "manifest_bytes", "padding_bytes"):
            if key in self.serialized:
                out[key] = self.serialized[key]
        return out

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "base": asdict(self.base),
            "conversion_config": self.conversion_config,
            "environment": self.environment,
            "totals": self.totals(),
            "serialized": self.serialized,
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
            serialized=dict(d.get("serialized", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        return cls.from_dict(json.loads(s))
