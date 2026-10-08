"""Tensor discovery from safetensors headers -- local files or remote Hub files via HTTP range reads.

Only the JSON header of each shard is read (a few KiB to ~100 KiB); weights are never downloaded.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"
HF_API = "https://huggingface.co/api/models/{repo}/revision/{revision}"
_MAX_HEADER = 64 << 20


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str


def _parse_header(raw: bytes, shard: str) -> list[TensorInfo]:
    hdr = json.loads(raw)
    return [TensorInfo(k, tuple(v["shape"]), v["dtype"], shard) for k, v in hdr.items() if k != "__metadata__"]


def local_tensors(model_dir: Path) -> list[TensorInfo]:
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    out: list[TensorInfo] = []
    for sh in shards:
        with open(sh, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n > _MAX_HEADER:
                raise ValueError(f"implausible safetensors header size {n} in {sh}")
            out.extend(_parse_header(f.read(n), sh.name))
    return out


def _get(url: str, rng: tuple[int, int] | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def remote_repo_info(repo: str, revision: str = "main") -> dict:
    return json.loads(_get(HF_API.format(repo=repo, revision=revision)))


def remote_json(repo: str, filename: str, revision: str = "main") -> dict:
    return json.loads(_get(HF_RESOLVE.format(repo=repo, revision=revision, filename=filename)))


def remote_tensors(repo: str, revision: str = "main") -> list[TensorInfo]:
    info = remote_repo_info(repo, revision)
    shards = sorted(s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors"))
    out: list[TensorInfo] = []
    for sh in shards:
        url = HF_RESOLVE.format(repo=repo, revision=revision, filename=sh)
        (n,) = struct.unpack("<Q", _get(url, (0, 7)))
        if n > _MAX_HEADER:
            raise ValueError(f"implausible safetensors header size {n} in {repo}/{sh}")
        out.extend(_parse_header(_get(url, (8, 8 + n - 1)), sh))
    return out
