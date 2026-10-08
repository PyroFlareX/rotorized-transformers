"""Tests for the ``rotor-scaler`` format (manifest + tensors) and ``rotor.storage`` safety rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rotor.formats import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    BaseProvenance,
    LayerEntry,
    Manifest,
    bytes_per_param,
    load_structured_layers,
    read_manifest,
    write_structured,
)
from rotor.formats.checkpoint import BASE_METADATA_FILES
from rotor.formats.spec import BASE_METADATA_DIRNAME
from rotor.math import CascadeSpec, RotorCascade, RotorLinear, Topology
from rotor.storage import (
    REPO_ROOT,
    Storage,
    StorageError,
    atomic_dir,
    atomic_write_bytes,
    load_storage,
    sha256_file,
)

# ------------------------------------------------------------------ helpers


def _make_layer(spec: CascadeSpec, with_bias: bool, seed: int) -> RotorLinear:
    g = torch.Generator().manual_seed(seed)
    cascade = RotorCascade(spec, dtype=torch.float32)
    with torch.no_grad():
        for p in cascade.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    bias = torch.randn(spec.out_features, generator=g) if with_bias else None
    return RotorLinear(cascade, bias)


def _entry(module: str, spec: CascadeSpec, storage_dtype: str, has_bias: bool) -> LayerEntry:
    n_params = spec.param_count()
    bpp = bytes_per_param(storage_dtype)
    return LayerEntry(
        module=module,
        source_tensor=f"model.language_model.{module.removeprefix('model.')}.weight",
        family="ffn.up",
        spec=spec,
        storage_dtype=storage_dtype,
        has_bias=has_bias,
        dense_params=spec.dense_param_count(),
        structured_params=n_params,
        dense_bytes=spec.dense_param_count() * 2,
        structured_bytes=n_params * bpp,
        metrics={"rel_fro": 0.125, "note": "unit-test value"},
    )


def _provenance() -> BaseProvenance:
    return BaseProvenance(
        hf_id="unit/test",
        revision="abc123",
        path="/nowhere/base",
        file_sha256={"model.safetensors": "0" * 64},
        tied_word_embeddings=True,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )


SPECS = {
    "model.layers.0.mlp.up_proj": (CascadeSpec(6, 10, 2, 3, Topology("butterfly", 0)), False),
    "model.layers.0.mlp.down_proj": (CascadeSpec(10, 6, 3, 2, Topology("seeded", 7), residual_rank=2), True),
    "model.layers.1.mlp.gate_proj": (CascadeSpec(8, 8, 1, 4, Topology("brick", 0), residual_rank=1), True),
    "model.layers.1.mlp.up_proj": (CascadeSpec(5, 5, 2, 0, Topology("brick", 0)), False),
}


def _build(storage_dtype: str) -> tuple[Manifest, dict[str, RotorLinear]]:
    layers = {m: _make_layer(spec, bias, seed=i) for i, (m, (spec, bias)) in enumerate(SPECS.items())}
    entries = [_entry(m, spec, storage_dtype, bias) for m, (spec, bias) in SPECS.items()]
    manifest = Manifest(
        base=_provenance(),
        layers=entries,
        conversion_config={"name": "unit", "budget": {"byte_ratio": 0.25}},
        environment={"note": "fixed for test"},
        notes=["unit test"],
    )
    return manifest, layers


def _write(tmp_path: Path, storage_dtype: str, base_dir: Path | None = None) -> tuple[Path, Manifest, dict]:
    manifest, layers = _build(storage_dtype)
    out = tmp_path / f"ckpt-{storage_dtype}"
    out.mkdir()
    write_structured(out, manifest, layers, base_dir)
    return out, manifest, layers


# ------------------------------------------------------------------ manifest


def test_manifest_json_roundtrip_including_cascade_spec_and_topology() -> None:
    manifest, _ = _build("float32")
    restored = Manifest.from_json(manifest.to_json())
    assert restored.format == FORMAT_NAME and restored.format_version == FORMAT_VERSION
    assert restored.base == manifest.base
    assert restored.conversion_config == manifest.conversion_config
    assert restored.environment == manifest.environment
    assert restored.notes == manifest.notes
    assert len(restored.layers) == len(manifest.layers)
    for got, want in zip(restored.layers, manifest.layers, strict=True):
        assert got == want
        assert isinstance(got.spec, CascadeSpec)
        assert isinstance(got.spec.topology, Topology)
    assert restored.layers[1].spec.topology == Topology("seeded", 7)
    # second serialization is byte-identical (stable key order, no hidden state)
    assert restored.to_json() == manifest.to_json()


def test_manifest_totals_are_sums_of_layer_bytes() -> None:
    manifest, _ = _build("bfloat16")
    totals = manifest.totals()
    assert totals["num_layers"] == len(SPECS)
    assert totals["dense_bytes"] == sum(e.dense_bytes for e in manifest.layers)
    assert totals["structured_bytes"] == sum(e.structured_bytes for e in manifest.layers)
    assert totals["byte_ratio"] == pytest.approx(totals["structured_bytes"] / totals["dense_bytes"])
    assert json.loads(manifest.to_json())["totals"] == totals
    assert Manifest(base=_provenance(), layers=[], conversion_config={}).totals()["byte_ratio"] is None


def test_manifest_rejects_wrong_format_and_newer_version() -> None:
    d = _build("float32")[0].to_dict()

    with pytest.raises(ValueError, match="not a rotor-scaler manifest"):
        Manifest.from_dict({**d, "format": "something-else"})
    with pytest.raises(ValueError, match="not a rotor-scaler manifest"):
        Manifest.from_dict({k: v for k, v in d.items() if k != "format"})
    with pytest.raises(ValueError, match="newer than supported"):
        Manifest.from_dict({**d, "format_version": FORMAT_VERSION + 1})
    # the current version is accepted
    assert Manifest.from_dict({**d, "format_version": FORMAT_VERSION}).format_version == FORMAT_VERSION


def test_bytes_per_param() -> None:
    assert bytes_per_param("float32") == 4
    assert bytes_per_param("bfloat16") == 2
    assert bytes_per_param("float16") == 2
    with pytest.raises(ValueError, match="unsupported storage dtype"):
        bytes_per_param("int4")


# ------------------------------------------------------------------ structured checkpoint I/O


def test_write_structured_files_and_tensor_keys(tmp_path: Path) -> None:
    from safetensors import safe_open

    out, manifest, _ = _write(tmp_path, "float32")
    assert (out / MANIFEST_FILENAME).is_file() and (out / WEIGHTS_FILENAME).is_file()
    assert not (out / BASE_METADATA_DIRNAME).exists()  # base_dir=None -> nothing copied
    assert read_manifest(out).to_json() == manifest.to_json()

    expected: set[str] = set()
    for m, (spec, bias) in SPECS.items():
        expected |= {f"{m}.{s}" for s in ("angles", "d_in", "d_mid", "d_out")}
        if spec.residual_rank:
            expected |= {f"{m}.residual.U", f"{m}.residual.V"}
        if bias:
            expected.add(f"{m}.bias")
    with safe_open(str(out / WEIGHTS_FILENAME), framework="pt") as f:
        assert set(f.keys()) == expected
        up = SPECS["model.layers.0.mlp.up_proj"][0]
        assert tuple(f.get_tensor("model.layers.0.mlp.up_proj.angles").shape) == (2, up.angles_per_rotor())
        assert tuple(f.get_tensor("model.layers.0.mlp.up_proj.d_mid").shape) == (1, 10)
        assert tuple(f.get_tensor("model.layers.0.mlp.up_proj.d_in").shape) == (6,)
        assert tuple(f.get_tensor("model.layers.0.mlp.up_proj.d_out").shape) == (10,)
        assert f.get_tensor("model.layers.0.mlp.up_proj.angles").dtype == torch.float32


def test_float32_storage_roundtrip_is_exact(tmp_path: Path) -> None:
    out, manifest, layers = _write(tmp_path, "float32")
    loaded_manifest, loaded = load_structured_layers(out, dtype=torch.float32)
    assert loaded_manifest.to_json() == manifest.to_json()
    assert set(loaded) == set(layers)

    g = torch.Generator().manual_seed(123)
    for name, orig in layers.items():
        got = loaded[name]
        assert got.cascade.spec == orig.cascade.spec
        assert (got.bias is None) == (orig.bias is None)
        assert (got.cascade.residual is None) == (orig.cascade.residual is None)
        for (n1, p1), (n2, p2) in zip(orig.state_dict().items(), got.state_dict().items(), strict=True):
            assert n1 == n2
            assert torch.equal(p1, p2), f"{name}.{n1} differs after float32 round trip"
        x = torch.randn(3, orig.in_features, generator=g)
        assert torch.equal(orig(x), got(x))
        assert torch.equal(orig.cascade.dense_weight(), got.cascade.dense_weight())


def test_bfloat16_storage_roundtrip_within_tolerance_and_halves_tensor_bytes(tmp_path: Path) -> None:
    from safetensors import safe_open

    out32, _, _ = _write(tmp_path, "float32")
    out16, _, layers = _write(tmp_path, "bfloat16")
    p16, p32 = str(out16 / WEIGHTS_FILENAME), str(out32 / WEIGHTS_FILENAME)
    with safe_open(p16, framework="pt") as f16, safe_open(p32, framework="pt") as f32:
        assert set(f16.keys()) == set(f32.keys())
        for key in f16.keys():  # noqa: SIM118 - safetensors handle, not a dict
            t16, t32 = f16.get_tensor(key), f32.get_tensor(key)
            assert t16.dtype == torch.bfloat16 and t32.dtype == torch.float32
            assert t16.numel() * t16.element_size() * 2 == t32.numel() * t32.element_size()

    _, loaded = load_structured_layers(out16, dtype=torch.float32)
    g = torch.Generator().manual_seed(5)
    for name, orig in layers.items():
        got = loaded[name]
        for (n1, p1), (_, p2) in zip(orig.state_dict().items(), got.state_dict().items(), strict=True):
            # stored values are exactly the bfloat16 rounding of the float32 masters ...
            assert torch.equal(p2, p1.to(torch.bfloat16).to(torch.float32)), f"{name}.{n1}"
            # ... which keeps 8 significant bits -> relative error <= 2^-8
            torch.testing.assert_close(p2, p1, rtol=2**-7, atol=1e-6)
        x = torch.randn(4, orig.in_features, generator=g)
        with torch.no_grad():
            ref, approx = orig(x), got(x)
        assert float((approx - ref).norm() / ref.norm()) < 0.1


def test_load_structured_layers_casts_to_requested_dtype(tmp_path: Path) -> None:
    out, _, _ = _write(tmp_path, "float32")
    _, loaded = load_structured_layers(out, dtype=torch.float64)
    layer = loaded["model.layers.0.mlp.down_proj"]
    assert all(p.dtype == torch.float64 for p in layer.parameters())
    assert layer.cascade.residual is not None and layer.bias is not None


def test_write_structured_rejects_manifest_layer_mismatch(tmp_path: Path) -> None:
    manifest, layers = _build("float32")
    layers.pop("model.layers.1.mlp.up_proj")
    out = tmp_path / "bad"
    out.mkdir()
    with pytest.raises(ValueError, match="differ"):
        write_structured(out, manifest, layers, None)


def test_manifest_rejection_surfaces_through_read_manifest(tmp_path: Path) -> None:
    out, _, _ = _write(tmp_path, "float32")
    path = out / MANIFEST_FILENAME
    data = json.loads(path.read_text())
    data["format_version"] = FORMAT_VERSION + 5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="newer than supported"):
        read_manifest(out)
    with pytest.raises(ValueError, match="newer than supported"):
        load_structured_layers(out)


def test_base_metadata_copy_picks_only_small_known_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"a": 1}')
    (base / "tokenizer.json").write_text("{}")
    (base / "chat_template.jinja").write_text("{{ x }}")
    (base / "model.safetensors.index.json").write_text("{}")
    # things that must NOT be copied
    (base / "model.safetensors").write_bytes(b"\0" * 4096)
    (base / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 4096)
    (base / "pytorch_model.bin").write_bytes(b"\0" * 64)
    (base / "README.md").write_text("readme")
    (base / "subdir").mkdir()
    (base / "subdir" / "config.json").write_text("{}")

    out, _, _ = _write(tmp_path, "float32", base_dir=base)
    copied = {p.name for p in (out / BASE_METADATA_DIRNAME).iterdir()}
    assert copied == {"config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json"}
    assert copied <= set(BASE_METADATA_FILES)
    assert all(not name.endswith((".safetensors", ".bin")) for name in copied)
    assert (out / BASE_METADATA_DIRNAME / "config.json").read_text() == '{"a": 1}'
    assert not any(p.suffix == ".safetensors" for p in (out / BASE_METADATA_DIRNAME).rglob("*"))


def test_base_metadata_files_are_all_small_text_assets() -> None:
    assert not any(name.endswith((".safetensors", ".bin", ".pt", ".gguf")) for name in BASE_METADATA_FILES)


# ------------------------------------------------------------------ storage


def test_tmp_storage_fixture_is_isolated(tmp_storage: Storage, tmp_path: Path) -> None:
    assert tmp_storage.nas_root == tmp_path / "nas"
    assert tmp_storage.base_root == tmp_path / "nas" / "base"
    assert tmp_storage.rotor_derived_root == tmp_path / "nas" / "derived" / "rotor-scaler"
    assert tmp_storage.preexisting_read_only == (tmp_path / "nas" / "legacy",)
    assert load_storage().nas_root == tmp_storage.nas_root  # env override is honored by default resolution
    assert tmp_storage.nas_available()


def test_assert_writable_rejects_base_root_readonly_and_outside_nas(tmp_storage: Storage, tmp_path: Path) -> None:
    s = tmp_storage
    for bad in (
        s.base_root,
        s.base_root / "qwen" / "model.safetensors",
        s.nas_root / "legacy",
        s.nas_root / "legacy" / "sub" / "file.bin",
    ):
        with pytest.raises(StorageError, match="read-only"):
            s.assert_writable(bad)
    for outside in (tmp_path / "elsewhere", tmp_path / "nas-sibling" / "x", Path("/tmp/rotor-outside"), tmp_path):
        with pytest.raises(StorageError, match="must live under"):
            s.assert_writable(outside)
    # `..` escapes are normalized before the check
    with pytest.raises(StorageError):
        s.assert_writable(s.rotor_derived_root / ".." / ".." / ".." / "escape")
    with pytest.raises(StorageError, match="read-only"):
        s.assert_writable(s.rotor_derived_root / ".." / ".." / "base" / "x")

    ok = s.rotor_derived_root / "run-1"
    assert s.assert_writable(ok) == ok
    assert s.assert_writable(s.nas_root / "derived" / "other") == s.nas_root / "derived" / "other"


def test_assert_under_nas_and_base_path(tmp_storage: Storage, tmp_path: Path) -> None:
    s = tmp_storage
    assert s.assert_under_nas(s.nas_root / "anything") == s.nas_root / "anything"
    with pytest.raises(StorageError):
        s.assert_under_nas(tmp_path / "other")
    assert s.base_path("qwen/0.8b") == s.base_root / "qwen" / "0.8b"
    assert s.base_path(s.base_root / "abs") == s.base_root / "abs"
    with pytest.raises(StorageError):
        s.base_path("/etc/passwd")


def test_derived_path_stays_under_rotor_root(tmp_storage: Storage) -> None:
    s = tmp_storage
    p = s.derived_path("exp-a", "sub")
    assert p == s.rotor_derived_root / "exp-a" / "sub"
    assert s.derived_path() == s.rotor_derived_root
    with pytest.raises(StorageError):
        s.derived_path("..", "..", "..", "escape")
    with pytest.raises(StorageError):
        s.derived_path("..", "..", "base", "poison")
    with pytest.raises(StorageError):
        s.derived_path("/etc/escape")  # absolute part replaces the root; must be refused


def test_validate_rejects_roots_outside_nas_root(tmp_path: Path) -> None:
    nas = tmp_path / "nas"
    good = {
        "nas_root": nas,
        "base_root": nas / "base",
        "derived_root": nas / "derived",
        "rotor_derived_root": nas / "derived" / "rotor-scaler",
    }
    Storage(**good).validate()
    for key in ("base_root", "derived_root", "rotor_derived_root"):
        with pytest.raises(StorageError, match=key):
            Storage(**{**good, key: tmp_path / "outside" / key}).validate()


def test_load_storage_rejects_config_with_root_outside_nas(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "nas_root: {nas}\nbase_root: {nas}/base\nderived_root: {out}/derived\nrotor_derived_root: {nas}/d/r\n".format(
            nas=tmp_path / "nas", out=tmp_path / "out"
        )
    )
    with pytest.raises(StorageError, match="derived_root"):
        load_storage(cfg)


def test_atomic_dir_publishes_on_success(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("published")
    with atomic_dir(dest) as tmp:
        assert tmp.is_dir() and tmp != dest and tmp.parent == dest.parent
        assert not dest.exists()  # not visible until the block exits
        (tmp / "a.txt").write_text("hello")
    assert dest.is_dir() and (dest / "a.txt").read_text() == "hello"
    assert [p.name for p in dest.parent.iterdir()] == ["published"]  # no temp leftovers


def test_atomic_dir_cleans_up_and_leaves_no_dest_on_exception(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("failing")
    seen: list[Path] = []
    with pytest.raises(RuntimeError, match="boom"), atomic_dir(dest) as tmp:
        seen.append(tmp)
        (tmp / "partial.bin").write_bytes(b"x" * 100)
        raise RuntimeError("boom")
    assert not dest.exists()
    assert seen and not seen[0].exists()
    assert list(dest.parent.iterdir()) == []


def test_atomic_dir_cleans_up_on_keyboard_interrupt(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("interrupted")
    with pytest.raises(KeyboardInterrupt), atomic_dir(dest) as tmp:
        (tmp / "partial").write_text("x")
        raise KeyboardInterrupt
    assert not dest.exists() and list(dest.parent.iterdir()) == []


def test_atomic_dir_refuses_existing_destination(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("exists")
    dest.mkdir(parents=True)
    (dest / "keep.txt").write_text("original")
    with pytest.raises(StorageError, match="refusing to overwrite"), atomic_dir(dest):
        pytest.fail("body must not run when destination exists")  # pragma: no cover
    assert (dest / "keep.txt").read_text() == "original"
    assert [p.name for p in dest.parent.iterdir()] == ["exists"]


def test_atomic_write_bytes_writes_and_refuses_overwrite(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("file.bin")
    atomic_write_bytes(dest, b"payload")
    assert dest.read_bytes() == b"payload"
    with pytest.raises(StorageError, match="refusing to overwrite"):
        atomic_write_bytes(dest, b"other")
    assert dest.read_bytes() == b"payload"
    assert [p.name for p in dest.parent.iterdir()] == ["file.bin"]  # no temp files remain


def test_atomic_write_bytes_creates_parents_and_cleans_temp_on_failure(tmp_storage: Storage) -> None:
    dest = tmp_storage.derived_path("nested", "deeper", "f.bin")
    atomic_write_bytes(dest, b"1")
    assert dest.read_bytes() == b"1"
    with pytest.raises((TypeError, ValueError)):
        atomic_write_bytes(tmp_storage.derived_path("nested", "bad.bin"), "not-bytes")  # type: ignore[arg-type]
    assert sorted(p.name for p in dest.parent.parent.iterdir()) == ["deeper"]


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    p = tmp_path / "blob"
    data = bytes(range(256)) * 50
    p.write_bytes(data)
    assert sha256_file(p, chunk=100) == hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ repo default config


def test_repo_storage_yaml_nas_root_is_the_mandated_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROTOR_STORAGE_CONFIG", raising=False)
    s = load_storage(REPO_ROOT / "configs" / "storage.yaml")
    assert s.nas_root == Path("/mnt/NAS/MediaHome/Local-Models")
    assert s.base_root == Path("/mnt/NAS/MediaHome/Local-Models/base")
    assert s.rotor_derived_root == Path("/mnt/NAS/MediaHome/Local-Models/derived/rotor-scaler")
    assert s.preexisting_read_only  # legacy NAS content is protected
    for ro in s.preexisting_read_only:
        with pytest.raises(StorageError):
            s.assert_writable(ro / "x")
    with pytest.raises(StorageError):
        s.assert_writable(s.base_root / "anything")
    assert s.derived_path("exp") == s.rotor_derived_root / "exp"


def test_default_resolution_without_env_uses_repo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROTOR_STORAGE_CONFIG", raising=False)
    assert load_storage().nas_root == Path("/mnt/NAS/MediaHome/Local-Models")
