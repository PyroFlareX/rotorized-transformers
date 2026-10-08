"""End-to-end conversion of a tiny Qwen3.5 checkpoint: plan, atomic write, provenance, reload, safety refusals."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from rotor.converters.pipeline import RANDOM_CALIBRATION_NOTE, ConversionConfig, convert
from rotor.data import random_windows, windows_fingerprint
from rotor.formats import MANIFEST_FILENAME, WEIGHTS_FILENAME, read_manifest
from rotor.math.cascade import RotorLinear
from rotor.models.loading import load_base, load_converted
from rotor.storage import StorageError, sha256_file

BYTE_RATIO = 0.5
FAMILIES = ["ffn.up", "ffn.down"]
LAYERS = [0, 1]
REPLACED = [f"model.layers.{i}.mlp.{n}" for i in LAYERS for n in ("up_proj", "down_proj")]

SCHEDULE = {
    "phases": [{"train": ["scales"], "steps": 3}, {"train": ["angles", "scales"], "steps": 6}],
    "lrs": {"angles": 0.02, "scales": 0.02},
    "cosine_decay": True,
    "log_every": 0,
}


def quiet(_msg: str) -> None:
    return None


def make_config(name: str, objective: str = "weight", seed: int = 0) -> ConversionConfig:
    # model key only supplies provenance (hf_id/revision); the base dir is always passed explicitly.
    return ConversionConfig.from_dict(
        {
            "name": name,
            "model": "qwen3.5-0.8b",
            "families": FAMILIES,
            "layers": LAYERS,
            "budget": {"byte_ratio": BYTE_RATIO, "storage_dtype": "bfloat16"},
            "cascade": {
                "topology": {"kind": "butterfly", "seed": 0},
                "stages_per_rotor": "auto",
                "residual_rank": 0,
                "adapter": "pad_truncate",
            },
            "fit": {
                "objective": objective,
                "seed": seed,
                "device": "cpu",
                "init_angle_std": 0.01,
                "schedule": SCHEDULE,
            },
            "calibration": {
                "source": "random",
                "path": None,
                "seq_len": 16,
                "num_windows": 2,
                "seed": 0,
                "max_rows": 64,
            },
        }
    )


@pytest.fixture
def base_dir(tmp_storage, tiny_ckpt: Path) -> Path:
    dest = tmp_storage.base_root / "tiny"
    shutil.copytree(tiny_ckpt, dest)
    return dest


def snapshot(directory: Path) -> dict[str, tuple[int, int, str]]:
    """(size, mtime_ns, sha256) for every file under ``directory``."""
    out = {}
    for f in sorted(p for p in directory.rglob("*") if p.is_file()):
        st = f.stat()
        out[str(f.relative_to(directory))] = (st.st_size, st.st_mtime_ns, sha256_file(f))
    return out


def test_dry_run_plan_fits_budget_and_writes_nothing(tmp_storage, base_dir):
    plan = convert(make_config("dry"), storage=tmp_storage, base_dir=base_dir, dry_run=True, log=quiet)
    assert isinstance(plan, dict)
    assert plan["model"] == "qwen3.5-0.8b"
    assert Path(plan["base_dir"]) == base_dir.absolute()
    assert sorted(r["module"] for r in plan["layers"]) == sorted(REPLACED)
    for r in plan["layers"]:
        assert r["dense_bytes"] == 4 * r["in"] * r["out"]  # tiny checkpoint is float32
        assert 0 < r["structured_bytes"] <= int(BYTE_RATIO * r["dense_bytes"]), r
        assert r["num_rotors"] >= 1 and r["stages_per_rotor"] >= 1
        assert r["structured_bytes"] == 2 * r["params"]  # bfloat16 storage
    totals = plan["totals"]
    assert totals["dense_bytes"] == sum(r["dense_bytes"] for r in plan["layers"])
    assert totals["structured_bytes"] == sum(r["structured_bytes"] for r in plan["layers"])
    assert totals["byte_ratio"] <= BYTE_RATIO
    assert not tmp_storage.rotor_derived_root.exists()


def test_convert_writes_atomically_with_verified_provenance(tmp_storage, base_dir):
    before = snapshot(base_dir)
    cfg = make_config("tiny-weight")
    out = convert(cfg, storage=tmp_storage, base_dir=base_dir, log=quiet)

    # published atomically under the rotor derived root, no temp leftovers
    assert out == tmp_storage.rotor_derived_root / "tiny-weight"
    assert sorted(p.name for p in tmp_storage.rotor_derived_root.iterdir()) == ["tiny-weight"]
    assert not [p for p in tmp_storage.rotor_derived_root.rglob("*") if ".tmp-" in p.name]
    assert (out / MANIFEST_FILENAME).is_file() and (out / WEIGHTS_FILENAME).is_file()

    manifest = read_manifest(out)
    assert sorted(e.module for e in manifest.layers) == sorted(REPLACED)
    totals = manifest.totals()
    assert totals["num_layers"] == 4
    assert 0 < totals["byte_ratio"] <= BYTE_RATIO
    for e in manifest.layers:
        assert e.structured_bytes <= int(BYTE_RATIO * e.dense_bytes)
        assert e.dense_params == e.spec.in_features * e.spec.out_features
        assert e.structured_params == e.spec.param_count()
        assert e.metrics["objective"] == "weight"
        assert torch.isfinite(torch.tensor(e.metrics["weight"]["rel_fro"]))
    # the manifest-recorded plan equals the dry-run plan
    plan = convert(cfg, storage=tmp_storage, base_dir=base_dir, dry_run=True, log=quiet)
    assert totals["structured_bytes"] == plan["totals"]["structured_bytes"]
    assert totals["dense_bytes"] == plan["totals"]["dense_bytes"]

    # provenance hashes match an independent recomputation over the actual base files
    prov = manifest.base
    assert prov.hf_id == "Qwen/Qwen3.5-0.8B-Base"
    assert prov.revision and prov.tied_word_embeddings is True
    assert Path(prov.path) == base_dir.absolute()
    assert prov.file_sha256["model.safetensors"] == sha256_file(base_dir / "model.safetensors")
    assert prov.file_sha256["config.json"] == sha256_file(base_dir / "config.json")
    assert manifest.conversion_config["name"] == "tiny-weight"

    # base untouched: same files, sizes, mtimes and bytes
    assert snapshot(base_dir) == before


def test_second_convert_to_same_name_refuses_overwrite(tmp_storage, base_dir):
    cfg = make_config("once")
    out = convert(cfg, storage=tmp_storage, base_dir=base_dir, log=quiet)
    first = snapshot(out)
    with pytest.raises(StorageError, match="overwrite"):
        convert(cfg, storage=tmp_storage, base_dir=base_dir, log=quiet)
    assert snapshot(out) == first
    assert sorted(p.name for p in tmp_storage.rotor_derived_root.iterdir()) == ["once"]


def test_load_converted_replaces_modules_and_preserves_the_rest(tmp_storage, base_dir):
    out = convert(make_config("reload"), storage=tmp_storage, base_dir=base_dir, log=quiet)
    base = load_base(base_dir, dtype=torch.float32)
    conv, manifest = load_converted(out, dtype=torch.float32)
    assert sorted(e.module for e in manifest.layers) == sorted(REPLACED)

    for path in REPLACED:
        assert isinstance(conv.get_submodule(path), RotorLinear), path
    # every other FFN projection is still a plain dense linear
    for i in (2, 3):
        assert isinstance(conv.get_submodule(f"model.layers.{i}.mlp.up_proj"), torch.nn.Linear)
    assert isinstance(conv.get_submodule("model.layers.0.mlp.gate_proj"), torch.nn.Linear)

    # untouched parameters are bit-identical to the base
    base_sd, conv_sd = base.state_dict(), conv.state_dict()
    replaced_prefixes = tuple(p + "." for p in REPLACED)
    untouched = [k for k in base_sd if not k.startswith(replaced_prefixes)]
    assert untouched and set(untouched) <= set(conv_sd)
    assert any("embed_tokens" in k for k in untouched) and any("self_attn" in k for k in untouched)
    for k in untouched:
        assert torch.equal(base_sd[k], conv_sd[k]), k
    # replaced weights were removed from the dense state, structured ones added
    assert not any(k in conv_sd for k in (p + ".weight" for p in REPLACED))
    assert all(any(k.startswith(p + ".cascade.") for k in conv_sd) for p in REPLACED)

    # logits: finite, but different from the base (structured approximation at 50% bytes after few steps)
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        lb, lc = base(ids).logits, conv(ids).logits
    assert torch.isfinite(lc).all()
    assert lc.shape == lb.shape
    assert (lc - lb).abs().max() > 1e-4

    # tied embeddings preserved: no separate lm_head parameters were created
    assert manifest.base.tied_word_embeddings is True
    head, emb = conv.lm_head.weight, conv.model.embed_tokens.weight
    assert head.data_ptr() == emb.data_ptr() or torch.equal(head, emb)
    assert torch.equal(emb, base.model.embed_tokens.weight)
    assert torch.equal(head, base.lm_head.weight)


def test_conversion_is_deterministic_across_names(tmp_storage, base_dir):
    a = convert(make_config("det-a", seed=7), storage=tmp_storage, base_dir=base_dir, log=quiet)
    b = convert(make_config("det-b", seed=7), storage=tmp_storage, base_dir=base_dir, log=quiet)
    ta, tb = load_file(str(a / WEIGHTS_FILENAME)), load_file(str(b / WEIGHTS_FILENAME))
    assert ta.keys() == tb.keys() and ta
    for k in ta:
        assert torch.equal(ta[k], tb[k]), k
    ma, mb = read_manifest(a), read_manifest(b)
    assert [e.spec for e in ma.layers] == [e.spec for e in mb.layers]
    assert [e.metrics["weight"] for e in ma.layers] == [e.metrics["weight"] for e in mb.layers]
    assert ma.base.file_sha256 == mb.base.file_sha256


def test_different_fit_seed_changes_tensors(tmp_storage, base_dir):
    """Guards the determinism test: the seed actually influences the fitted parameters."""
    a = convert(make_config("seed-a", seed=0), storage=tmp_storage, base_dir=base_dir, log=quiet)
    b = convert(make_config("seed-b", seed=1), storage=tmp_storage, base_dir=base_dir, log=quiet)
    ta, tb = load_file(str(a / WEIGHTS_FILENAME)), load_file(str(b / WEIGHTS_FILENAME))
    assert any(not torch.equal(ta[k], tb[k]) for k in ta if k.endswith(".angles"))


def test_activation_objective_with_random_calibration(tmp_storage, base_dir):
    before = snapshot(base_dir)
    out = convert(make_config("tiny-act", objective="activation"), storage=tmp_storage, base_dir=base_dir, log=quiet)
    manifest = read_manifest(out)
    expected_fp = windows_fingerprint(random_windows(128, 16, 2, 0))
    assert manifest.conversion_config["windows_fingerprint"] == expected_fp
    assert manifest.conversion_config["fit"]["objective"] == "activation"
    assert RANDOM_CALIBRATION_NOTE in manifest.notes
    assert 0 < manifest.totals()["byte_ratio"] <= BYTE_RATIO
    for e in manifest.layers:
        assert e.metrics["objective"] == "activation"
        act = e.metrics["activation"]
        assert act["rows"] == 32  # 2 windows x 16 tokens, below max_rows
        assert torch.isfinite(torch.tensor(act["rel_fro"]))
    conv, _ = load_converted(out, dtype=torch.float32)
    with torch.no_grad():
        logits = conv(torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(5))).logits
    assert torch.isfinite(logits).all()
    assert snapshot(base_dir) == before


def test_missing_base_dir_raises_and_mentions_no_download(tmp_storage):
    cfg = make_config("nobase")
    # resolved from models.yaml -> <tmp base_root>/Qwen/Qwen3.5-0.8B-Base, which does not exist
    with pytest.raises(FileNotFoundError, match=r"(?i)no download") as ei:
        convert(cfg, storage=tmp_storage, log=quiet)
    msg = str(ei.value)
    assert "Qwen/Qwen3.5-0.8B-Base" in msg and str(tmp_storage.base_root) in msg
    explicit = tmp_storage.base_root / "does-not-exist"
    with pytest.raises(FileNotFoundError, match=r"(?i)no download"):
        convert(cfg, storage=tmp_storage, base_dir=explicit, dry_run=True, log=quiet)
    assert not explicit.exists()
    assert not tmp_storage.rotor_derived_root.exists()


def test_out_dir_inside_base_root_is_refused(tmp_storage, base_dir):
    before = snapshot(base_dir)
    for dest in (base_dir / "converted", tmp_storage.base_root / "elsewhere"):
        with pytest.raises(StorageError):
            convert(make_config("bad-out"), storage=tmp_storage, base_dir=base_dir, out_dir=dest, log=quiet)
        assert not dest.exists()
    assert snapshot(base_dir) == before


def test_out_dir_outside_nas_is_refused(tmp_storage, base_dir, tmp_path: Path):
    with pytest.raises(StorageError):
        convert(
            make_config("bad-out2"), storage=tmp_storage, base_dir=base_dir, out_dir=tmp_path / "outside", log=quiet
        )
    assert not (tmp_path / "outside").exists()


def test_explicit_out_dir_under_derived_root_is_atomic(tmp_storage, base_dir):
    dest = tmp_storage.rotor_derived_root / "custom" / "run1"
    out = convert(make_config("ignored-name"), storage=tmp_storage, base_dir=base_dir, out_dir=dest, log=quiet)
    assert out == dest and (dest / MANIFEST_FILENAME).is_file()
    assert not [p for p in tmp_storage.rotor_derived_root.rglob("*") if ".tmp-" in p.name]


def test_config_roundtrip_and_validation():
    cfg = make_config("rt")
    assert ConversionConfig.from_dict(copy.deepcopy(cfg.to_dict())).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError):
        ConversionConfig.from_dict({**cfg.to_dict(), "bogus": 1})
    with pytest.raises(ValueError):
        ConversionConfig.from_dict({**cfg.to_dict(), "fit": {"objective": "nonsense"}})


def test_attention_families_are_not_convertible(tmp_storage, base_dir):
    cfg = ConversionConfig.from_dict({**make_config("attn").to_dict(), "families": ["attn.q_gate"]})
    with pytest.raises(ValueError, match="not enabled for conversion"):
        convert(cfg, storage=tmp_storage, base_dir=base_dir, dry_run=True, log=quiet)
