# -*- coding: utf-8 -*-
"""
指定した Hugging Face データセットからキーワードを含むサンプルを抽出し、
単一のテキストファイルに書き出すシンプルなスクリプト。

例:
    python make_dataset.py --dataset fujiki/wiki40b_ja --split train \
        --keyword "ファミコン" --out output_with_titles.txt
"""
import argparse

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fujiki/wiki40b_ja")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--keyword", type=str, required=True)
    p.add_argument("--out", type=str, default="output_with_titles.txt")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split)

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "")
            title = item.get("title", "")
            if args.keyword in text:
                f.write(f"【{title}】\n{text}\n\n")
                count += 1

    print(f"[OK] Extracted {count} samples containing '{args.keyword}' -> {args.out}")


if __name__ == "__main__":
    main()
