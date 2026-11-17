# 人格QLoRA — キャラの性格・口調をモデルに焼き込む

キャラクター設定（`character.yaml`）を毎ターン長いプロンプトで説明する代わりに、
小さな LoRA アダプタとしてモデルの重みに学習させる機能。

| | 従来（プロンプトで説明） | 人格QLoRA |
|---|---|---|
| キャラ設定の入力 | 毎ターン全文（サンプルで約2,000字） | 名前と一言プロフィールだけ（約半分） |
| 口調の安定性 | 記憶が増える・会話が長くなると崩れやすい | 重みに入っているので崩れにくい |
| JSON出力 | たまに崩れて skip になる | 出力形式ごと学習するので安定 |
| 応答開始 | プロンプト処理ぶん遅い | 入力が短いぶん速い |

## 使い方（ダッシュボード → 🧬 人格QLoRA、または http://127.0.0.1:9000/persona ）

1. **環境をチェック** — torch / peft / bitsandbytes が揃っているか確認する。
   足りなければ下の「インストール」を参照。
2. **① データセット作成** — 教材を自動で作る。
   - `character.yaml` のセリフ例（「…」で囲まれた部分）
   - ローカルチャット（`/chat`）で実際に話した会話
   - 今の LLM（LM Studio 等）に **全文のキャラ設定** を渡して答えさせた応答（既定200件）
   合成には今の primary LLM を使うので、LM Studio を起動しておく。
3. **② 学習** — ベースモデルに LoRA を学習する（別プロセス）。
   **学習中は LM Studio のモデルをアンロード**して VRAM を空けること。
4. **③ Ollama へ登録** — 統合モデルを GGUF（既定 q8_0）に変換して Ollama モデルを作り、
   そのまま VRAM に読み込む。できた `model-q8_0.gguf` は LM Studio でも使える。
5. **④ 使う** — 押すだけで応答LLMが切り替わる（再起動不要）。
   「通常のLLMに戻す」でいつでも戻せる。

専用モデルが落ちている・応答しないときは、ヘルスチェックで通常のLLMに自動で戻り、
プロンプトも全文に戻る（配信は止まらない）。

## インストール

```bash
py -3.10 -m pip install -r requirements-train.txt
```

- RTX 50xx（Blackwell）で 4bit（QLoRA）を使うには **bitsandbytes 0.46 以降** が必要。
  古い bitsandbytes しか無い場合は自動で通常の LoRA（bf16）に切り替わる。
  bf16 LoRA は VRAM を多く使うので、16GB なら 3B クラスまでが目安。
- Ollama: https://ollama.com （登録・実行に使う）

### GGUF変換の準備（済）

Ollama 0.3x は Hugging Face 形式の Qwen2 を直接取り込めない（`unsupported MLX architecture`）ため、
llama.cpp の変換スクリプトで GGUF にしてから登録する。スクリプトは `tools/llama.cpp` に取得済み。
取り直す場合:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
```

その後 tools/llama.cpp で sparse-checkout に
`/convert_hf_to_gguf.py /convert_lora_to_gguf.py /gguf-py/ /conversion/` を設定する。

### このPCの準備状況（2026-09-29）

| 項目 | 状態 |
|---|---|
| GPU | RTX 5080 16GB / CUDA / bf16 |
| peft / accelerate / bitsandbytes | 0.21.1 / 1.15.0 / 0.50.2（4bit動作確認済） |
| ベースモデル | `<YOUR_PATH>/your-base-model`（config.yaml に設定済） |
| Ollama | 0.34.4（自動起動・事前ロード確認済） |
| GGUF変換 | `tools/llama.cpp`（変換→登録→応答まで確認済） |
| 先生役（会話例づくり） | LM Studio の Qwen3.6-35B-A3B Q4_0（読み込み約30秒、1回約2秒）。VRAMを約15GB使うので、学習の前にアンロードすること |

## VRAMの目安（QLoRA / 最大長1024 / バッチ1）

| ベースモデル | 4bit QLoRA | bf16 LoRA |
|---|---|---|
| Qwen2.5-0.5B / 1.5B | 3〜5GB | 4〜8GB |
| Qwen2.5-3B | 5〜7GB | 10〜12GB |
| Qwen2.5-7B（既定） | 9〜12GB | 16GB超（不可） |

## 設定（config.yaml の `persona:`）

```yaml
persona:
  enabled: false            # ④で「使う」を押すと true になる
  runtime_model: ''         # Ollama 上のモデル名
  backend: ollama           # 専用モデルを呼ぶ LLM バックエンド（llm.backends のキー）
  prompt_mode: slim         # slim=キャラ設定を省略 / full=従来どおり全文
  fallback_to_base: true    # 専用モデルが落ちたら通常の primary に戻す
  base_model: Qwen/Qwen2.5-7B-Instruct
  epochs: 3.0
  learning_rate: 0.0002
  lora_r: 16
  lora_alpha: 32
  max_seq_len: 1024
  grad_accum: 8
  synthetic_samples: 200
  include_chat_history: true
```

## Ollama の自動起動

Ollama を使う設定（`llm.primary: ollama` または人格モデル使用中）なら、サーバ起動時に

1. Ollama が起動していなければ `ollama serve` をウィンドウ無しで立ち上げる
2. 使うモデルを VRAM に読み込んでおく（1発話目のロード待ちをなくす）

を自動で行う。配信中に Ollama が落ちても、次の応答で起動し直してから再送する。
自分で起動した `ollama serve` だけをサーバ終了時に止める（手動で起動していたものは触らない）。

```yaml
ollama:
  auto_start: true          # 起動していなければ立ち上げる
  preload_on_start: true    # 起動時にモデルを読み込む
  executable: ''            # ollama.exe のパス（空なら自動検出）
  keep_alive: 30m           # 最後の応答からVRAMに残す時間（-1で無期限）
  auto_pull: false          # 未取得モデルを自動ダウンロードする（数GB）
  startup_timeout_sec: 30
  stop_on_exit: true
```

注意: LM Studio と Ollama の両方にモデルを載せると VRAM が足りなくなることがある。
人格モデルを使うときは LM Studio のモデルをアンロードしておくと安定する。

## 出力ファイル

```
models/persona/<キャラ名>/
  dataset/train.jsonl, eval.jsonl, meta.json   … 教材
  <日時>/job.json          … 学習条件
  <日時>/adapter/          … LoRAアダプタ（数十MB）
  <日時>/merged/           … ベースに統合したフル重み（Ollama登録用）
  <日時>/Modelfile         … ollama create に使ったもの
  <日時>/manifest.json     … loss・試し出力・登録名
  <日時>/train.log         … 学習ログ全文
```

## コマンドラインから使う

```bash
# 学習環境の確認
py -3.10 -m backend.persona.trainer --check
# ダッシュボードが作った job.json で学習だけ回す
py -3.10 -m backend.persona.trainer --job models/persona/サンプル/20260929-170000/job.json
```

## Ollama 以外で使う

- **llama.cpp server**: `merged/` を `convert_hf_to_gguf.py` で GGUF にして
  `llama-server -m model.gguf` で起動し、config.yaml に
  `llm.backends.llamacpp: {base_url: http://localhost:8080/v1}` を足して
  `persona.backend: llamacpp` にする。
- **LM Studio**: 同じく GGUF にしたものを LM Studio のモデルフォルダに置いて読み込み、
  `persona.backend: lmstudio`、`persona.runtime_model` に LM Studio 上のモデル名を入れる。

## 設計メモ

- 学習データのシステムプロンプトは、実行時と同じ `build_system_prompt(slim_persona=True)` で作る。
  学習時と実行時でプロンプトがずれると口調が乗らないため。
- 教師データは「全文のキャラ設定を渡したLLM」の応答を、「slimプロンプト」と組にして保存する
  （プロンプトで表現していた人格を重みに移す蒸留）。
- 損失はアシスタントの応答部分だけで計算する。
- 学習は配信サーバとは別プロセス。torch 等を配信サーバに読み込まないので、
  学習用パッケージが無くても配信は普通に動く。