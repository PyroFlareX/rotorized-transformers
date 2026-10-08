"""Qwen3.5 adapter: tensor classification, projection discovery, module resolution, safe replacement."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rotor.math.lowrank import LowRankLinear
from rotor.models.discovery import local_tensors
from rotor.models.qwen3_5 import (
    CONVERTIBLE_FAMILIES,
    TEXT_PREFIX,
    classify_tensor,
    discover_projections,
    layer_types,
    load_text_model,
    select_projections,
)
from rotor.models.replace import get_module, replace_module

EXPECTED_LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
LINEAR_ATTN_FAMILIES = {"deltanet.in_qkv", "deltanet.in_z", "deltanet.in_a", "deltanet.in_b", "deltanet.out"}
FULL_ATTN_FAMILIES = {"attn.q_gate", "attn.k", "attn.v", "attn.o"}
FFN_FAMILIES = {"ffn.gate", "ffn.up", "ffn.down"}


@pytest.fixture(scope="module")
def tiny_config(tiny_ckpt: Path) -> dict:
    return json.loads((tiny_ckpt / "config.json").read_text())


@pytest.fixture(scope="module")
def tiny_tensors(tiny_ckpt: Path):
    return local_tensors(tiny_ckpt)


@pytest.fixture(scope="module")
def tiny_projections(tiny_tensors, tiny_config):
    return discover_projections(tiny_tensors, tiny_config)


def test_layer_types_alternate_per_config(tiny_config):
    assert layer_types(tiny_config) == EXPECTED_LAYER_TYPES


def test_layer_types_from_full_attention_interval():
    cfg = {"text_config": {"num_hidden_layers": 8, "full_attention_interval": 4}}
    assert layer_types(cfg) == ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + [
        "full_attention"
    ]


def test_discover_covers_every_2d_text_layer_tensor(tiny_tensors, tiny_projections):
    two_d_layer = {t.name for t in tiny_tensors if t.name.startswith(TEXT_PREFIX + "layers.") and len(t.shape) == 2}
    assert two_d_layer, "tiny checkpoint should contain 2-D decoder projections"
    assert {p.tensor for p in tiny_projections} == two_d_layer
    # no embeddings / vision tensors sneak in
    assert all(".layers." in p.tensor and not p.tensor.startswith("model.visual.") for p in tiny_projections)


def test_discover_families_and_layer_types(tiny_projections, tiny_config):
    types = layer_types(tiny_config)
    by_layer: dict[int, set[str]] = {}
    for p in tiny_projections:
        assert p.layer_type == types[p.layer]
        assert p.dtype == "F32"
        by_layer.setdefault(p.layer, set()).add(p.family)
        # classify_tensor agrees with discovery
        assert classify_tensor(p.tensor) == p.family
        m = re.fullmatch(r"model\.language_model\.(layers\.\d+\..+)\.weight", p.tensor)
        assert m and p.module == f"model.{m.group(1)}"
    assert sorted(by_layer) == [0, 1, 2, 3]
    for layer, fams in by_layer.items():
        attn = LINEAR_ATTN_FAMILIES if types[layer] == "linear_attention" else FULL_ATTN_FAMILIES
        assert fams == attn | FFN_FAMILIES, (layer, fams)
    # alternation as seen through discovery
    first_type = {p.layer: p.layer_type for p in tiny_projections}
    assert [first_type[i] for i in range(4)] == EXPECTED_LAYER_TYPES


def test_discovered_module_paths_resolve_with_matching_shapes(tiny_ckpt, tiny_projections):
    model = load_text_model(tiny_ckpt, dtype=torch.float32)
    for p in tiny_projections:
        mod = get_module(model, p.module)
        assert tuple(mod.weight.shape) == (p.out_features, p.in_features), p.module
        assert (mod.out_features, mod.in_features) == (p.out_features, p.in_features)
    # values: loaded text model weight equals the checkpoint tensor
    sample = tiny_projections[0]
    with safe_open(str(tiny_ckpt / "model.safetensors"), framework="pt", device="cpu") as f:
        ckpt_w = f.get_tensor(sample.tensor)
    assert torch.equal(get_module(model, sample.module).weight.detach(), ckpt_w)


def test_tiny_ffn_shapes(tiny_projections):
    ffn = [p for p in tiny_projections if p.family.startswith("ffn.")]
    assert len(ffn) == 12
    for p in ffn:
        expected = (64, 96) if p.family == "ffn.down" else (96, 64)
        assert (p.out_features, p.in_features) == expected


def test_select_projections_filters_by_family_and_layer(tiny_projections):
    sel = select_projections(tiny_projections, ["ffn.up", "ffn.down"], layers=[0, 3])
    assert len(sel) == 4
    assert {p.family for p in sel} == {"ffn.up", "ffn.down"}
    assert {p.layer for p in sel} == {0, 3}
    assert len(select_projections(tiny_projections, list(CONVERTIBLE_FAMILIES))) == 12


@pytest.mark.parametrize("family", ["attn.q_gate", "attn.k", "attn.v", "attn.o", "deltanet.in_qkv", "deltanet.out"])
def test_select_projections_rejects_attention_families(tiny_projections, family):
    with pytest.raises(ValueError, match="not enabled for conversion"):
        select_projections(tiny_projections, [family])
    with pytest.raises(ValueError, match="not enabled for conversion"):
        select_projections(tiny_projections, ["ffn.up", family])


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("model.visual.blocks.0.attn.qkv.weight", "vision"),
        ("mtp.fc.weight", "mtp"),
        ("model.language_model.embed_tokens.weight", "embedding"),
        ("model.language_model.layers.3.linear_attn.in_proj_qkv.weight", "deltanet.in_qkv"),
        ("model.language_model.layers.3.linear_attn.in_proj_z.weight", "deltanet.in_z"),
        ("model.language_model.layers.3.linear_attn.out_proj.weight", "deltanet.out"),
        ("model.language_model.layers.3.self_attn.q_proj.weight", "attn.q_gate"),
        ("model.language_model.layers.3.self_attn.o_proj.weight", "attn.o"),
        ("model.language_model.layers.12.mlp.gate_proj.weight", "ffn.gate"),
        ("model.language_model.layers.12.mlp.up_proj.weight", "ffn.up"),
        ("model.language_model.layers.12.mlp.down_proj.weight", "ffn.down"),
        ("model.language_model.norm.weight", "final_norm"),
        ("model.language_model.layers.3.input_layernorm.weight", "text.other"),
        ("lm_head.weight", "lm_head"),
    ],
)
def test_classify_tensor(name, group):
    assert classify_tensor(name) == group


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(ids).logits


def test_replace_module_full_rank_lowrank_keeps_logits(tiny_ckpt):
    model = load_text_model(tiny_ckpt, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 128, (2, 12), generator=g)
    before = _logits(model, ids)

    for path in ("model.layers.0.mlp.up_proj", "model.layers.1.mlp.down_proj"):
        old = get_module(model, path)
        W = old.weight.detach().clone()
        new = LowRankLinear.from_dense(W, rank=min(W.shape))
        returned = replace_module(model, path, new)
        assert returned is old
        assert get_module(model, path) is new

    after = _logits(model, ids)
    assert torch.isfinite(after).all()
    torch.testing.assert_close(after, before, atol=1e-4, rtol=0)


def test_replace_module_changes_logits_when_rank_truncated(tiny_ckpt):
    """Sanity: the golden test above is not vacuous -- a rank-1 replacement does change the output."""
    model = load_text_model(tiny_ckpt, dtype=torch.float32)
    ids = torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(1))
    before = _logits(model, ids)
    path = "model.layers.0.mlp.up_proj"
    replace_module(model, path, LowRankLinear.from_dense(get_module(model, path).weight.detach(), rank=1))
    assert (_logits(model, ids) - before).abs().max() > 1e-4


def test_replace_module_rejects_shape_mismatch(tiny_ckpt):
    model = load_text_model(tiny_ckpt, dtype=torch.float32)
    path = "model.layers.0.mlp.up_proj"  # (96, 64)
    old = get_module(model, path)
    wrong = LowRankLinear.from_dense(torch.randn(64, 96, generator=torch.Generator().manual_seed(0)), rank=4)
    with pytest.raises(ValueError, match="shape mismatch"):
        replace_module(model, path, wrong)
    assert get_module(model, path) is old  # model untouched after the failed swap


def test_replace_module_rejects_missing_path(tiny_ckpt):
    model = load_text_model(tiny_ckpt, dtype=torch.float32)
    new = LowRankLinear.from_dense(torch.randn(96, 64, generator=torch.Generator().manual_seed(0)), rank=4)
    with pytest.raises(AttributeError):
        replace_module(model, "model.layers.0.mlp.nonexistent_proj", new)
    with pytest.raises(AttributeError):
        replace_module(model, "model.layers.99.mlp.up_proj", new)


@pytest.mark.network
def test_remote_qwen35_0_8b_headers_match_registry():
    """Header-only HTTP range reads (no weights downloaded)."""
    from rotor.models.discovery import remote_json, remote_tensors
    from rotor.storage import model_entry

    _, entry = model_entry("qwen3.5-0.8b")
    repo, revision = "Qwen/Qwen3.5-0.8B-Base", entry["revision"]
    assert entry["hf_id"] == repo
    tensors = remote_tensors(repo, revision)
    by_name = {t.name: t for t in tensors}
    up = by_name["model.language_model.layers.0.mlp.up_proj.weight"]
    assert up.shape == (3584, 1024)

    config = remote_json(repo, "config.json", revision)
    projs = discover_projections(tensors, config)
    assert len({p.layer for p in projs}) == 24
    ffn_up = [p for p in projs if p.family == "ffn.up"]
    assert len(ffn_up) == 24
    assert all((p.out_features, p.in_features) == (3584, 1024) for p in ffn_up)
    types = layer_types(config)
    assert len(types) == 24 and types[3] == "full_attention" and types[0] == "linear_attention"
