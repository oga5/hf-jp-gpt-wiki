
# This code is based on the project "LLMs-from-scratch" by Sebastian Raschka, 
# licensed under the Apache License 2.0.
#
# Original credit:
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Minimal CLI for running greedy text generation with a Hugging Face model
and a SentencePiece tokenizer downloaded from the Hub.

Example:
    python generate.py --prompt "Hello. What's the most interesting thing that's happened to you recently?" --max-new-tokens 50

Options:
    --repo-id      Hugging Face repo id (default: oga5/hf-jp-gpt-wiki)
    --spm-file     SentencePiece model filename in the repo (default: jp_tok_wiki.model)
    --device       cpu or cuda (default: auto-detect)
"""
import argparse
import os
from typing import Optional

import torch
import sentencepiece as spm
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal text generation CLI")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="oga5/hf-jp-gpt-wiki",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--spm-file",
        type=str,
        default="jp_tok_wiki.model",
        help="SentencePiece model filename inside the repo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run on (default: auto-detect)",
    )
    return parser.parse_args()


def auto_device(user_choice: Optional[str]) -> torch.device:
    if user_choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if user_choice == "cpu":
        return torch.device("cpu")
    # auto-detect
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resources(repo_id: str, spm_file: str, device: torch.device):
    # Load model (trust_remote_code in case the repo provides a custom model)
    model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    model.eval()
    model.to(device)

    # Download SentencePiece model file and initialize tokenizer
    spm_path = hf_hub_download(repo_id=repo_id, filename=spm_file)
    if not os.path.exists(spm_path):
        raise FileNotFoundError(f"Downloaded SPM file not found at {spm_path}")
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    return model, sp


def greedy_generate(model, sp: spm.SentencePieceProcessor, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    eos_id = sp.eos_id()
    input_ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Some custom models define context_length in config
    ctx = getattr(model.config, "context_length", None)
    if ctx is None:
        # Fallback to a common default context length if unspecified
        ctx = 2048

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -ctx:]
            out = model(input_ids=idx_cond)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_id.item() == eos_id:
                break
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return sp.decode(input_ids[0].tolist())


def main():
    args = parse_args()
    device = auto_device(args.device)
    print(f"Using device: {device}")

    model, sp = load_resources(args.repo_id, args.spm_file, device)
    output = greedy_generate(model, sp, args.prompt, args.max_new_tokens, device)

    print("\n=== Generation ===")
    print(output)


if __name__ == "__main__":
    main()
