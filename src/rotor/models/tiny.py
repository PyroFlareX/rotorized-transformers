"""Tiny random Qwen3.5 checkpoints for tests and CLI smoke runs (no download, < 1 MB).

Produces the same on-disk layout as the published checkpoints: a ``Qwen3_5ForConditionalGeneration``
safetensors file (``model.language_model.*``, ``model.visual.*``) plus ``config.json``. MTP tensors
are not emitted by ``transformers``' ``save_pretrained`` and are therefore absent from tiny fixtures.
"""

from __future__ import annotations

from pathlib import Path

TINY_TEXT_CONFIG = dict(
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    vocab_size=128,
    full_attention_interval=2,
    tie_word_embeddings=True,
    layer_types=["linear_attention", "full_attention"] * 2,
)
TINY_VISION_CONFIG = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)


def make_tiny_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Write a deterministic tiny Qwen3.5 checkpoint into ``out_dir`` (must not exist or be empty)."""
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    cfg = Qwen3_5Config(text_config=TINY_TEXT_CONFIG, vision_config=TINY_VISION_CONFIG, tie_word_embeddings=True)
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    model.save_pretrained(str(out_dir))
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    vocabulary.update({chr(codepoint): codepoint - 27 for codepoint in range(32, 127)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(""), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(str(out_dir))
    return out_dir
