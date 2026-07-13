---
title: MOSS 音声文字起こし・話者分離
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.12"
app_file: app.py
suggested_hardware: t4-small
preload_from_hub:
  - OpenMOSS-Team/MOSS-Transcribe-Diarize
---

# MOSS 音声文字起こし・話者分離アプリ

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) を使い、アップロードした音声を時刻・話者ラベル付きで文字起こしする Gradio アプリです。結果は次の形式の UTF-8 テキストとしてダウンロードできます。

```text
[00:00:00.480] S01: こんにちは。
[00:00:12.260] S02: よろしくお願いします。
```

S01、S02 などは音声内だけで有効な匿名の話者ラベルです。

## ローカルで試す

Python 3.12 と `uv` を推奨します。初回推論時に Hugging Face から約0.9Bパラメータのモデルを取得します。

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv run python app.py
```

ブラウザで `http://127.0.0.1:7860` を開きます。デバイスは CUDA、Apple Silicon の MPS、CPU の順で自動選択されます。明示する場合は起動前に `MOSS_DEVICE=cuda:0`、`MOSS_DEVICE=mps`、または `MOSS_DEVICE=cpu` を設定してください。

### Local LLMで認識誤りを訂正する

[Ollama](https://ollama.com/) を起動して日本語対応モデルを用意すると、文字起こし後に同音異義語、漢字、助詞、句読点などを校正できます。話者ラベルとタイムスタンプは変更しません。

```bash
ollama list
ollama serve
```

アプリの「詳細設定」で「Local LLMで音声認識誤りを訂正」を有効にします。既定値は導入済みのQwen 3.6系モデル `qwen3.6:27b-mtp-q4_K_M` と `http://127.0.0.1:11434` です。別のモデルやURLは画面上で変更するか、`OLLAMA_MODEL`、`OLLAMA_BASE_URL` 環境変数で設定できます。「訂正用の用語・文脈」には製品名、人名、専門用語などを入力できます。

Ollamaへの接続失敗、応答形式の不正、発話の欠落があった場合は、誤った訂正を採用せずエラーを表示します。Hugging Face Spacesから手元の `127.0.0.1` には接続できないため、この機能は原則としてローカル実行向けです。

## Hugging Face Spacesへ配置する

1. Hugging Faceで Gradio SDK の新しい Space を作成します。
2. このリポジトリの内容を Space の Git リポジトリへ push します。
3. Space の **Settings → Hardware** で GPU を選びます（READMEの `suggested_hardware` は自動割り当てではありません）。
4. Build 完了後、音声をアップロードして動作確認します。

CPU Space でも起動できますが、利用可能なハードウェアは Hugging Face の契約プランに依存します。実用的な推論速度には GPU を推奨します。モデル読み込み後の同時推論数は、GPUメモリ不足を避けるため1に制限しています。

---

# MOSS-Transcribe-Diarize 日本語精度評価

日本語の長時間・複数話者音声について、文字起こしと話者分離を同じ条件で評価するための小さなハーネスです。

## 指標

- **CER**: 話者を無視した文字誤り率。Unicode NFKC、空白・句読点・記号を除去して評価します。
- **cpCER**: 話者ラベルの名前を最適に対応付けた、話者別連結文字誤り率。
- **Delta-cp**: `cpCER - CER`。文字自体ではなく話者帰属で増えた誤りの目安です。
- **DER**: Miss + False alarm + Speaker confusion。既定では参照境界の前後 0.25 秒を採点から除外します。
- **RTF**: 推論秒数 / 音声秒数。1未満なら実時間より高速です。

数値はすべて micro average です。漢字とかなの違いは誤りとして残します。フィラーも勝手に除去しません。

## 1. 評価データを作る

UTF-8 JSONLを用意します。1行が1録音で、時刻は秒です。

```json
{"id":"meeting_001","audio":"audio/meeting_001.wav","segments":[{"start":0.5,"end":2.8,"speaker":"A","text":"本日の議題を確認します。"}]}
```

参照データの最低条件:

1. 16 kHz mono WAVを推奨（元音源は必ず保存）。
2. 発話の重なりは、同じ時間帯に複数のsegmentを置く。
3. 言い直し、フィラー、固有名詞を聞こえた通りに記す。
4. 20本以上、合計2時間以上を最低ラインとし、話者数・雑音・重なり率で層別する。
5. 少なくとも10%は別の注釈者が二重注釈し、人間同士のCER/DERも記録する。

公開単一話者コーパスだけでは話者分離を検証できません。まず単一話者音声で日本語CERを広く測り、別に実会議・対談を手動注釈してcpCER/DERを測る二段構成を推奨します。合成会話は回帰テストには使えますが、実会話の代用にはしません。

正解が `話者名: 本文` のテキストで、タイムスタンプがない場合は次のように変換できます。

```bash
moss-ja-reference --transcript transcription.txt --audio audio.mp3 --output data/reference.jsonl
```

この形式ではCERとcpCERのみ有効です。採点時に必ず `--skip-der` を付けてください。

## 2. セットアップと推論

公式実装は Python 3.12 と Transformers 5.x でテストされています。このリポジトリもPython 3.12を使います。

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra inference --extra dev

moss-ja-infer \
  --manifest data/reference.jsonl \
  --output runs/moss-ja.jsonl \
  --device mps
```

Apple Siliconでは公式実装の `auto` がCPUを選ぶため、このラッパーはMPSがあれば自動利用します。MPSで未対応演算に遭遇した場合だけ `--device cpu` に切り替えてください。最初の実行時にモデルをダウンロードします。

推論は `do_sample=False` 固定です。長い音声で出力が途中終了したら `--max-new-tokens` を増やします。公平な比較では、同一音声・同一プロンプト・同一正規化・同一collarを固定してください。

## 3. 採点

```bash
moss-ja-score \
  --reference data/reference.jsonl \
  --hypothesis runs/moss-ja.jsonl \
  --output runs/score.json
```

採点器だけなら重い推論依存は不要です。

```bash
uv run --extra dev pytest
uv run moss-ja-score --reference examples/reference.jsonl --hypothesis examples/hypothesis.jsonl
```

## 4. HTML Diffレポート

```bash
moss-ja-report \
  --reference data/reference.jsonl \
  --hypothesis runs/moss-ja.jsonl \
  --score runs/score.json \
  --output runs/transcription-diff.html
```

生成したHTMLは外部ファイルやWebサーバーを必要とせず、そのままブラウザで開けます。CER正規化後のDiffと、句読点・改行を含む原文Diffを切り替えられます。

## 推奨する評価内訳

結果は総合値だけでなく、次の切り口で分けて残します。

- 音声: clean / 雑音 / 遠距離マイク / 電話
- 話者数: 1 / 2 / 3–4 / 5以上
- 重なり率: 0% / 0–10% / 10%以上
- 長さ: 30秒以下 / 30秒–10分 / 10分以上
- 内容: 日常会話 / 会議 / 講義 / 固有名詞・数字が多い領域

モデル選定では、まずCER、次にcpCERとDER、最後にRTFとメモリ使用量を見ます。CERだけが良くても、誰が話したかを誤るモデルは議事録用途では不十分です。
