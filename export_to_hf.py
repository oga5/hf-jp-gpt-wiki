# -*- coding: utf-8 -*-
"""
公開用: カスタムGPTモデルとSentencePieceトークナイザをHugging Face互換のフォルダにエクスポートします。
trust_remote_code=True で読み込めるよう、最小のバックボーン実装を同梱します。

使用例:
    python export_to_hf.py \
      --weights hyper_small_jp_wiki.pth \
      --spm_model jp_tok_wiki.model \
      --spm_vocab jp_tok_wiki.vocab \
      --out_dir hf_jp_gpt_wiki \
      --context_length 256 \
      --emb_dim 128 \
      --n_layers 4 \
      --n_heads 4 \
      --drop_rate 0.1 \
      --qkv_bias False \
      --fast

エクスポート後の読み込み例:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("hf_jp_gpt_wiki", trust_remote_code=True)
"""
import os
import json
import argparse
from collections import OrderedDict

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to .pth weights (state_dict)")
    p.add_argument("--spm_model", type=str, required=True, help="Path to SentencePiece .model")
    p.add_argument("--spm_vocab", type=str, required=False, default="", help="Path to SentencePiece .vocab (optional)")
    p.add_argument("--out_dir", type=str, default="hf_jp_gpt_wiki", help="Output directory for HF repo")
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--emb_dim", type=int, required=True)
    p.add_argument("--n_layers", type=int, required=True)
    p.add_argument("--n_heads", type=int, required=True)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--qkv_bias", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--vocab_size", type=int, default=None, help="Override vocab size if known; else inferred from checkpoint")
    p.add_argument("--fast", action="store_true", help="Use GPTModelFast backbone when loading")
    return p.parse_args()


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


MODEL_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
MODEL_CODE_FILE = "modeling_custom_gpt.py"


MODEL_CODE = r'''# Auto-loaded via trust_remote_code (standalone, vendored backbone)
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


# ---- Vendored minimal backbone (matches GPTModelFast structure/keys) ----

class FeedForwardFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv
        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec


class TransformerBlockFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForwardFast(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModelFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        
    def forward(self, in_idx: torch.LongTensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class CustomGPTConfig(PretrainedConfig):
    model_type = "custom-gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 256,
        emb_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        fast: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.drop_rate = float(drop_rate)
        self.qkv_bias = bool(qkv_bias)
        self.fast = bool(fast)


class CustomGPTForCausalLM(PreTrainedModel):
    config_class = CustomGPTConfig

    def __init__(self, config: CustomGPTConfig):
        super().__init__(config)
        cfg = {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "emb_dim": config.emb_dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "drop_rate": config.drop_rate,
            "qkv_bias": config.qkv_bias,
        }
        self.backbone = GPTModelFast(cfg)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        logits = self.backbone(input_ids)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}
'''


def write_model_code(out_dir: str):
    with open(os.path.join(out_dir, MODEL_CODE_FILE), "w", encoding="utf-8") as f:
        f.write(MODEL_CODE)


def write_config(out_dir: str, cfg: dict):
    cfg_out = dict(cfg)
    cfg_out.update({
        "_class_name": "CustomGPTConfig",
        "model_type": "custom-gpt",
        "architectures": ["CustomGPTForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_custom_gpt.CustomGPTConfig",
            "AutoModelForCausalLM": "modeling_custom_gpt.CustomGPTForCausalLM"
        },
    })
    with open(os.path.join(out_dir, CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, ensure_ascii=False, indent=2)


def write_tokenizer_config(out_dir: str, context_length: int):
    tok_cfg = {
        "model_max_length": int(context_length),
        "tokenizer_class": None,
    }
    with open(os.path.join(out_dir, TOKENIZER_CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=False, indent=2)


def copy_tokenizer_files(out_dir: str, spm_model: str, spm_vocab: str = ""):
    import shutil
    shutil.copy2(spm_model, os.path.join(out_dir, os.path.basename(spm_model)))
    if spm_vocab and os.path.exists(spm_vocab):
        shutil.copy2(spm_vocab, os.path.join(out_dir, os.path.basename(spm_vocab)))


def infer_vocab_size_from_state_dict(sd: dict) -> int:
    for key in ("tok_emb.weight", "backbone.tok_emb.weight"):
        if key in sd:
            return sd[key].shape[0]
    for key in ("out_head.weight", "backbone.out_head.weight"):
        if key in sd:
            return sd[key].shape[0]
    raise RuntimeError("Could not infer vocab_size from state_dict; please pass --vocab_size")


def remap_keys_with_prefix(sd: dict, prefix: str) -> OrderedDict:
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            new_sd[k] = v
        else:
            new_sd[f"backbone.{k}"] = v
    return new_sd


def main():
    args = parse_args()

    ensure_out_dir(args.out_dir)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(args.weights)
    state_dict = torch.load(args.weights, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    vocab_size = args.vocab_size or infer_vocab_size_from_state_dict(state_dict)

    cfg = {
        "vocab_size": int(vocab_size),
        "context_length": int(args.context_length),
        "emb_dim": int(args.emb_dim),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "drop_rate": float(args.drop_rate),
        "qkv_bias": bool(args.qkv_bias),
        "fast": bool(args.fast),
    }
    write_config(args.out_dir, cfg)

    copy_tokenizer_files(args.out_dir, args.spm_model, args.spm_vocab)
    write_tokenizer_config(args.out_dir, args.context_length)

    write_model_code(args.out_dir)

    remapped = remap_keys_with_prefix(state_dict, prefix="backbone.")

    torch.save(remapped, os.path.join(args.out_dir, MODEL_FILE))

    print(f"[OK] Exported to {args.out_dir}")
    print("Load with: from transformers import AutoModelForCausalLM;\n"
          "model = AutoModelForCausalLM.from_pretrained(\"%s\", trust_remote_code=True)" % args.out_dir)


if __name__ == "__main__":
    main()
