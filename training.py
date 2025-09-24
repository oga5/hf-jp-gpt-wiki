# -*- coding: utf-8 -*-
"""
Self-contained minimal training script for a small GPT (Japanese-friendly) using SentencePiece.

This file vendors the minimal Transformer backbone (fast variant) so it does not depend
on other project files.

Typical pipeline:
  1) Download dataset (example wiki40b_ja) -> see download_dataset.py
  2) Extract subset by keyword to a single text file -> see make_dataset.py
  3) Train a tiny GPT on the resulting text -> this script
  4) Export to Hugging Face format if desired -> see export_to_hf.py

Example:
  python training.py \
    --train-txt output_with_titles.txt \
    --spm-model jp_tok_wiki.model \
    --out-weights hyper_small_jp_wiki.pth \
    --context-length 256 \
    --emb-dim 128 \
    --n-layers 4 \
    --n-heads 4 \
    --drop-rate 0.1 \
    --batch-size 32 \
    --max-steps 2000 \
    --lr 3e-4

Note:
- This is a minimal example intended for small experiments. It performs language modeling
  with next-token prediction using greedy batching from a monolithic token stream.
- For serious training, consider proper dataset sharding, packing, and better optimization.
"""
import argparse
import math
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm


# --------------------
# Model (fast variant)
# --------------------
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
        self.trf_blocks = nn.Sequential(*[TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])] )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# --------------------
# Data utilities
# --------------------
class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def vocab_size(self) -> int:
        return self.sp.get_piece_size()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_token_stream(text: str, tokenizer: SentencePieceTokenizer) -> List[int]:
    # Simple whitespace-splitting not needed; SentencePiece works on raw text
    return tokenizer.encode(text)


class LMStreamDataset(IterableDataset):
    """
    Create (input, target) pairs from a monolithic token stream by sampling
    random contiguous chunks of length `context_length`.
    """
    def __init__(self, token_ids: List[int], context_length: int, samples_per_epoch: int):
        super().__init__()
        self.token_ids = token_ids
        self.context = context_length
        self.samples = samples_per_epoch

    def __iter__(self):
        rng = random.Random()
        N = len(self.token_ids)
        max_start = max(0, N - (self.context + 1))
        for _ in range(self.samples):
            if max_start <= 0:
                # pad a trivial example if stream too short
                x = [0] * self.context
                y = [0] * self.context
            else:
                s = rng.randint(0, max_start)
                seq = self.token_ids[s : s + self.context + 1]
                x = seq[:-1]
                y = seq[1:]
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# --------------------
# Training
# --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal GPT training (self-contained)")
    p.add_argument("--train-txt", type=str, required=True, help="Path to training text file")
    p.add_argument("--spm-model", type=str, required=True, help="Path to SentencePiece .model")
    p.add_argument("--out-weights", type=str, default="model.pth", help="Output path for weights")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--qkv-bias", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=10000, help="IterableDataset samples per epoch")
    return p.parse_args()


def auto_device(user_choice: str | None) -> torch.device:
    if user_choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if user_choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = auto_device(args.device)
    print(f"Using device: {device}")

    # Tokenizer and tokens
    tokenizer = SentencePieceTokenizer(args.spm_model)
    vocab_size = tokenizer.vocab_size()
    print(f"Vocab size: {vocab_size}")

    text = load_text(args.train_txt)
    token_stream = build_token_stream(text, tokenizer)
    print(f"Token count: {len(token_stream):,}")

    # Model
    cfg = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "drop_rate": args.drop_rate,
        "qkv_bias": bool(args.qkv_bias),
    }
    model = GPTModelFast(cfg).to(device)

    # Data
    dataset = LMStreamDataset(token_stream, args.context_length, samples_per_epoch=args.samples_per_epoch)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Optimizer / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    step = 0
    model.train()
    running_loss = 0.0
    while step < args.max_steps:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            # logits: [B, T, V] -> reshape for CE
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += loss.item()
            if step % 50 == 0:
                avg = running_loss / 50
                running_loss = 0.0
                print(f"step {step}/{args.max_steps} - loss {avg:.4f}")

            if step >= args.max_steps:
                break

    # Save weights
    os.makedirs(os.path.dirname(args.out_weights) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.out_weights)
    print(f"[OK] Saved weights -> {args.out_weights}")


if __name__ == "__main__":
    main()
