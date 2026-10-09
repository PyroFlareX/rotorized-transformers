"""Qwen3.5 (``model_type: qwen3_5``) adapter: tensor naming, projection families, text-only loading.

Verified against the published headers of Qwen/Qwen3.5-{0.8B,2B,4B}-Base (2026-10-08) and the
``transformers`` 5.19 ``qwen3_5`` implementation:

- Checkpoints are ``Qwen3_5ForConditionalGeneration`` (vision tower under ``model.visual.*``,
  text decoder under ``model.language_model.*``, multi-token-prediction head under ``mtp.*``).
- Decoder layers alternate ``linear_attention`` (Gated DeltaNet: ``linear_attn.in_proj_{qkv,z,a,b}``,
  ``out_proj``, ``conv1d``, ``A_log``, ``dt_bias``, ``norm``) and ``full_attention`` (gated attention:
  ``self_attn.{q,k,v,o}_proj`` with ``q_proj`` producing query+gate, ``q_norm``, ``k_norm``) per
  ``layer_types`` / ``full_attention_interval``. Every layer has a SwiGLU ``mlp.{gate,up,down}_proj``.
- ``tie_word_embeddings: true``: no ``lm_head.weight`` tensor is stored.
- ``AutoModelForCausalLM`` loads the text-only ``Qwen3_5ForCausalLM`` from the multimodal
  checkpoint; its module paths drop ``language_model`` (``model.layers.N...``) and it ignores
  ``mtp.*`` and ``model.visual.*``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rotor.models.discovery import TensorInfo

TEXT_PREFIX = "model.language_model."
CAUSAL_PREFIX = "model."

# tensor-name regex (relative to TEXT_PREFIX) -> family
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")
_FAMILIES = {
    "mlp.gate_proj": "ffn.gate",
    "mlp.up_proj": "ffn.up",
    "mlp.down_proj": "ffn.down",
    "self_attn.q_proj": "attn.q_gate",
    "self_attn.k_proj": "attn.k",
    "self_attn.v_proj": "attn.v",
    "self_attn.o_proj": "attn.o",
    "linear_attn.in_proj_qkv": "deltanet.in_qkv",
    "linear_attn.in_proj_z": "deltanet.in_z",
    "linear_attn.in_proj_a": "deltanet.in_a",
    "linear_attn.in_proj_b": "deltanet.in_b",
    "linear_attn.out_proj": "deltanet.out",
}
# Families the initial milestone may replace. Everything else keeps original behavior.
CONVERTIBLE_FAMILIES = ("ffn.gate", "ffn.up", "ffn.down")


@dataclass(frozen=True)
class Projection:
    tensor: str  # checkpoint tensor name
    module: str  # module path in Qwen3_5ForCausalLM
    layer: int
    layer_type: str  # linear_attention | full_attention
    family: str
    out_features: int
    in_features: int
    dtype: str


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def layer_types(config: dict) -> list[str]:
    tc = text_config(config)
    if tc.get("layer_types"):
        return list(tc["layer_types"])
    k = int(tc["full_attention_interval"])
    return ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(int(tc["num_hidden_layers"]))]


def classify_tensor(name: str) -> str:
    """Coarse group of any checkpoint tensor (used by inspection reports)."""
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith(TEXT_PREFIX):
        rel = name[len(TEXT_PREFIX) :]
        if rel.startswith("embed_tokens"):
            return "embedding"
        if rel.startswith("norm."):
            return "final_norm"
        m = _LAYER_RE.match(rel)
        if m and m.group(2) in _FAMILIES:
            return _FAMILIES[m.group(2)]
        return "text.other"
    if name.startswith("lm_head"):
        return "lm_head"
    return "other"


def discover_projections(tensors: list[TensorInfo], config: dict) -> list[Projection]:
    """All 2-D text-decoder projection weights, classified. Shapes come from the headers, not assumptions."""
    types = layer_types(config)
    out: list[Projection] = []
    for t in tensors:
        if not t.name.startswith(TEXT_PREFIX) or len(t.shape) != 2:
            continue
        m = _LAYER_RE.match(t.name[len(TEXT_PREFIX) :])
        if not m or m.group(2) not in _FAMILIES:
            continue
        idx = int(m.group(1))
        out.append(
            Projection(
                tensor=t.name,
                module=f"{CAUSAL_PREFIX}layers.{idx}.{m.group(2)}",
                layer=idx,
                layer_type=types[idx] if idx < len(types) else "unknown",
                family=_FAMILIES[m.group(2)],
                out_features=t.shape[0],
                in_features=t.shape[1],
                dtype=t.dtype,
            )
        )
    out.sort(key=lambda p: (p.layer, p.module))
    return out


def select_projections(
    projections: list[Projection], families: list[str], layers: list[int] | None = None
) -> list[Projection]:
    bad = sorted(set(families) - set(CONVERTIBLE_FAMILIES))
    if bad:
        raise ValueError(f"families {bad} are not enabled for conversion yet; allowed: {CONVERTIBLE_FAMILIES}")
    return [p for p in projections if p.family in families and (layers is None or p.layer in layers)]


def load_text_model(model_dir: Path, dtype=None, device=None):
    """Load the text-only causal LM from local files (vision/MTP weights are skipped)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype or torch.bfloat16, local_files_only=True
    )
    if device is not None:
        model = model.to(device)
    return model.eval()


def load_tokenizer(model_dir: Path):
    """Load the checkpoint's local tokenizer, including generic tiny-fixture tokenizers."""
    import json

    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    tokenizer_config = model_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        if config.get("tokenizer_class") == "TokenizersBackend":
            from transformers.tokenization_utils_tokenizers import TokenizersBackend

            return TokenizersBackend.from_pretrained(str(model_dir), local_files_only=True)
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
