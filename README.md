# LLMの事前学習のサンプル

このディレクトリは、https://huggingface.co/oga5/hf-jp-gpt-wiki で公開しているLLMのトレーニングに用いたコードです。
書籍「つくりながら学ぶ！LLM 自作入門」を参考にしています。(多くのコードを、そのまま利用しています)

ファイル一覧:
- `download_dataset.py`: データセットをダウンロードするスクリプト
- `make_dataset.py`: データセットを加工するスクリプト
- `training.py`: トレーニングを行うスクリプト
- `generate.py`: テキスト生成を行うスクリプト
- `export_to_hf.py`: Hugging Face にモデルをエクスポートするスクリプト

## 動作要件

- Python 3.9 以上
- インターネット接続（Hugging Face Hub からモデル等をダウンロード）
- CPU で実行可能（速度向上には GPU を推奨）

## インストール

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## データのダウンロード: `download_dataset.py`

Hugging Face Datasets から日本語の Wiki40B をダウンロードします（キャッシュされます）。

```bash
python download_dataset.py
```

## データ抽出（小さくする）: `make_dataset.py`

データセットから特定のキーワードを含む記事を抽出し、1つのテキストファイルにまとめます。

```bash
python make_dataset.py \
  --dataset fujiki/wiki40b_ja \
  --split train \
  --keyword "ファミコン" \
  --out output_with_titles.txt
```

## トレーニング: `training.py`

最小の GPT（fast 版バックボーンを同梱）で自己完結するトレーニングを行います。他ファイルへの依存はありません。

```bash
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
```

主な引数:

- `--train-txt`: 学習元となるテキストファイル（例: `output_with_titles.txt`）
- `--spm-model`: SentencePiece の `.model` ファイル
- `--out-weights`: 出力する学習済み重み（`.pth`）
- `--context-length`, `--emb-dim`, `--n-layers`, `--n-heads`, `--drop-rate`, `--qkv-bias`: モデル構成
- `--batch-size`, `--max-steps`, `--lr`: 学習ハイパーパラメータ

## 生成: `generate.py`

Hugging Face 上の公開リポジトリからモデルと SentencePiece を自動ダウンロードして、貪欲法（greedy）でテキスト生成します。

```bash
python generate.py --prompt "こんにちは。最近あった面白いことは、" --max-new-tokens 50
```

主なオプション:

- `--repo-id`: 使用する Hugging Face リポジトリ ID（既定: `oga5/hf-jp-gpt-wiki`）
- `--spm-file`: リポジトリ内の SentencePiece モデルファイル名（既定: `jp_tok_wiki.model`）
- `--device`: `cpu` または `cuda`（既定: 利用可能なら自動で CUDA、なければ CPU）

注意:

- リポジトリ側でカスタムモデルを提供しているため、読み込み時に `trust_remote_code=True` を使用します。信頼できるソースのみで使用してください。
- 依存を最小限に保つため、デフォルトでは貪欲法（greedy）で生成します。

## Hugging Face 形式へのエクスポート: `export_to_hf.py`

学習済み重みと SentencePiece ファイルから、`trust_remote_code=True` で読み込める最小構成の Hugging Face 互換フォルダを作ります。

```bash
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
```

出力先フォルダは以下を含みます:

- `pytorch_model.bin`
- `config.json`（`auto_map` 付き）
- `tokenizer_config.json`
- `modeling_custom_gpt.py`（最小バックボーン同梱）
- SentencePiece サイドカー（`.model` / `.vocab`）

読み込み例:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("hf_jp_gpt_wiki", trust_remote_code=True)
```

## ライセンス

- 本リポジトリのコードは Apache License 2.0 の下で提供します。詳細は `LICENSE.md` をご確認ください。

- トレーニング用コードの大部分は "LLMs from Scratch" のサンプルを元にしています（Apache 2.0）。ソース: https://github.com/rasbt/LLMs-from-scratch

- 学習データセット: [fujiki/wiki40b_ja](https://huggingface.co/datasets/fujiki/wiki40b_ja)。このデータセットは wiki40b の日本語部分を再整形したものです。本データセットを利用する場合は、以下の原著論文を引用してください:

```
@inproceedings{guo-etal-2020-wiki,
    title = "{W}iki-40{B}: Multilingual Language Model Dataset",
    author = "Guo, Mandy  and
      Dai, Zihang  and
      Vrande{\v{c}}i{\'c}, Denny  and
      Al-Rfou, Rami",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.297",
    pages = "2440--2452",
    abstract = "We propose a new multilingual language model benchmark that is composed of 40+ languages spanning several scripts and linguistic families. With around 40 billion characters, we hope this new resource will accelerate the research of multilingual modeling. We train monolingual causal language models using a state-of-the-art model (Transformer-XL) establishing baselines for many languages. We also introduce the task of multilingual causal language modeling where we train our model on the combined text of 40+ languages from Wikipedia with different vocabulary sizes and evaluate on the languages individually. We released the cleaned-up text of 40+ Wikipedia language editions, the corresponding trained monolingual language models, and several multilingual language models with different fixed vocabulary sizes.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```
