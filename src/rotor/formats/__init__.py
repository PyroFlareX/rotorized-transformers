"""Versioned structured-weight format, manifests, and checkpoint I/O."""

from rotor.formats.checkpoint import load_structured_layers, read_manifest, write_structured
from rotor.formats.manifest import BaseProvenance, LayerEntry, Manifest
from rotor.formats.spec import FORMAT_NAME, FORMAT_VERSION, MANIFEST_FILENAME, WEIGHTS_FILENAME, bytes_per_param

__all__ = [
    "load_structured_layers",
    "read_manifest",
    "write_structured",
    "BaseProvenance",
    "LayerEntry",
    "Manifest",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "WEIGHTS_FILENAME",
    "bytes_per_param",
]
