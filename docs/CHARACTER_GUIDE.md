# キャラクターガイド — 自分のキャラクターを作る

キャラクターは「`config.yaml` の `characters` エントリ」と「キャラクター設定フォルダ」の 2 つで構成されます。同梱の **Character1** がそのまま記入例になっているので、それを複製して書き換えるのが最短です。

本パッケージに学習済みモデル（QLoRA）は含まれません。まずはプロンプト設定だけでキャラクターを作り、必要になったら §7 の QLoRA を検討してください。

## 1. キャラクターの構成

```
config.yaml
└─ characters:            ← 出演の並び・声・LLM・アバターの割り当て

backend/llm/prompts/characters/<キャラクター名>/
├─ character.yaml         ← 初期設定（L0a）: 名前・説明・性格・ルール
├─ memory.md              ← 長期記憶の初期値（自動でも追記・整理される）
├─ avatar.yaml            ← アバタープリセット（VRM のモデルパス・待機モーション等）
└─ （実行時に生成）memories.db / viewer_profiles.db / chat_history.jsonl
```

- `character.yaml` がプロンプト階層の **L0a（初期設定）** になります。ここに書いた内容は実行時に書き換えられることがなく、キャラクターの土台として常にプロンプトに入ります（階層の詳細は [ARCHITECTURE.md](ARCHITECTURE.md)）。
- 記憶（memories.db）はキャラクターごとに分離されます。複数キャラクターを運用しても、互いの記憶・意見・意図が混ざることはありません。

## 2. 手順

### 2.1 設定フォルダを複製する

```bash
backend/llm/prompts/characters/Character1/
```

をコピーし、フォルダ名を自分のキャラクター名に変えます。

### 2.2 character.yaml を書く

キーは 4 つだけです。

| キー | 内容 |
|---|---|
| `name` | キャラクターの名前（`config.yaml` の `characters[].name` と一致させる） |
| `description` | 自己紹介・肩書き・世界観などの説明文 |
| `personality` | 性格・口調・一人称・語尾などの定義 |
| `rules` | 振る舞いのルール（守ってほしいこと・禁止事項） |

書き方のコツ:

- **セリフ例を入れる**と口調が安定します。`description` / `personality` / `rules` の中に、

  ```
  1. 配信開始の挨拶
  「（セリフ本文）」
  ```

  の形（番号付き見出し＋「…」）で書くと、そのまま人格 QLoRA の学習データ抽出にも使われます（§7）。
- ルールには「分からないことは分からないと言う」「視聴者の個人情報を聞き出さない」のような、守らせたい線引きを具体的に書きます。
- Character1 の `character.yaml` に各キーの記入例とセリフ例 3 件が入っているので、文量と粒度の目安にしてください。

### 2.3 memory.md に初期の記憶を書く

最初から覚えておいてほしいことを箇条書きで書きます（空でも動作します）。配信や会話を通じて自動で追記・整理されていくファイルなので、ここは「生まれた時点で知っていること」だけに絞るのがおすすめです。

### 2.4 アバターを用意する

- **VRM を使う場合**: モデルファイルを `models/` 以下に置き、`avatar.yaml` の `model_path` と待機モーション（`idle_motion`）を指定します。キャリブレーション（腕の下げ量など）はダッシュボードからも調整できます。
- **Live2D を使う場合**: モデル一式を `models/live2d/` 以下に置き、`config.yaml` のアバター設定でパスを指定します。
- **動画アバターを使う場合**: モーション動画とカタログ（`catalog.json`）を `models/motions/<名前>/` に置きます。

アバター種別の割り当ては `config.yaml` の `characters[].avatar.type`（vrm / live2d / video）で行い、書かなければ全体設定（`avatar:` セクション）を使います。

### 2.5 config.yaml にエントリを追加する

```yaml
characters:
  - name: Character1        # character.yaml の name と一致させる
    id: Character1          # アバター画面の ?char= や WebSocket で使う識別子
    enabled: true
    prompt_file: ''         # 空なら characters/<name>/character.yaml を読む
    data_dir: backend/llm/prompts
    aliases:
      - Character1ちゃん    # コメントで呼ばれたと判定する別名
    voice:
      backend: ''           # 空なら tts.primary を使う
      voice: Character1     # Irodori-TTS の参照音声名（§3）
      speaker_id: null      # VOICEVOX の話者 ID
      speed: null
      pitch: null
    llm:
      backend: ''           # 空なら llm.primary / fallback を使う
      model: ''             # キャラ専用モデル（QLoRA の登録名など）を使うときだけ指定
    fallback_lines:         # 生成が詰まったときの繋ぎ台詞（汎用文のみ）
      - ちょっと一息つきましょうか。のんびりいきましょう。
    avatar:
      type: vrm
      vrm:
        model_path: models/default.vrm
        idle_motion: breathe
```

主な項目:

| 項目 | 意味 |
|---|---|
| `name` / `id` | 表示名と識別子。`id` はアバター画面の振り分けに使います |
| `prompt_file` | character.yaml の場所を直接指定したい場合のみ。空なら `<data_dir>/characters/<name>/character.yaml` |
| `aliases` | 名指し判定に使う別名（名前そのものは自動で含まれます） |
| `voice` | 声の割り当て（次節）。空の項目は全体の TTS 設定を使います |
| `llm` | キャラクター専用の LLM バックエンド／モデル（QLoRA の登録名など） |
| `fallback_lines` | 生成が反復ブロック等で詰まったときにそのまま喋る繋ぎ台詞。話題に依存しない汎用文だけを置き、事実の主張や数値は入れません |
| `avatar` | キャラクター専用のアバター設定。無ければ全体設定を使います |

### 2.6 起動して確認する

本体を起動し、`/chat` のローカルチャットで会話してみます。口調・一人称・ルールが意図どおりかをここで詰めます。キャラクターの追加・切替はダッシュボード（キャスト設定）からも行えます。

## 3. ボイスの登録

### Irodori-TTS の場合

1. TTS サーバ側の `voices/` フォルダに、キャラクターの参照音声（.wav）を置きます。
2. ファイル名（拡張子なし）を、キャラクター設定の `voice.voice` に書きます。

感情ごとに声を変えることもできます。`use_emotion_voice: true` と `emotion_voice_prefix` を設定すると、発話の感情に応じて `<prefix>-<emotion>` という名前の参照音声に切り替わります（例: `Character1-happy.wav`）。感情別の音声が無い感情では、通常のボイスが使われます。

参照音声は、ノイズの少ない明瞭な録音を数十秒程度用意すると安定します。声の権利（自分や許諾を得た声優の声など）には十分注意してください。

### VOICEVOX の場合

使いたい話者の `speaker_id` をキャラクター設定に書きます。話者 ID は VOICEVOX 側の話者一覧で確認できます。`speed` / `pitch` もキャラクターごとに上書きできます。

## 4. 2 体目を追加する（掛け合い）

`characters` に 2 体目（例: Character2）を並べると掛け合い配信になります。手順は 1 体目と同じで、設定フォルダと `config.yaml` のエントリをもう 1 組用意するだけです（同梱の Character2 の `character.yaml` が簡易な記入例になっています）。

- 誰がコメントに答えるか、相方がどう受けるかは演出層（TurnDirector）が決めます。確率やターン上限は `config.yaml` の `ensemble` セクションで調整できます。
- アバター画面はキャラクターごとに `/avatar?char=<id>` を分けて OBS に貼ります。
- 人格 QLoRA の専用モデルはリーダー（1 体目）にだけ載る設計です。相方にも専用モデルを使いたい場合は、`characters[].llm.model` に相方用の登録名を指定してください。

## 5. 記憶を育てる

作った直後のキャラクターは、`memory.md` に書いたこと以外を知りません。配信や会話を重ねると、

- エピソード記憶（話した内容）が蓄積され、自律発話の話題に使われる
- 意見・意図・選好が出所タグ付きで形成され、プロンプトの L0b〜L2 に反映される
- 夜間の睡眠時統合で、1 日の経験が固定化される

という形で少しずつ育ちます。仕組みの詳細は [ARCHITECTURE.md](ARCHITECTURE.md) の「プロンプト階層と記憶」を参照してください。育ち方をリセットしたい場合は、そのキャラクターの `memories.db` を退避・削除すれば初期状態に戻ります（`memory.md` の内容は残ります）。

## 6. （任意）QLoRA で口調を学習させる

プロンプトだけでは口調の癖が安定しない場合、人格 QLoRA で「気質・口調」の層をモデル側に焼き込む選択肢があります。考え方は次のとおりです。

- **学習させるのは口調だけ**です。事実・意見・出来事のような変わりうる情報はモデルに焼かず、記憶の仕組み（§5）側で扱います。モデルに焼いた知識は後から直せません。
- **学習データ**は、character.yaml のセリフ例と実際の会話履歴から自動生成され、教師モデルの合成データで補われます（ダッシュボードの Persona 画面、または `/persona` から実行）。
- **学習→変換→登録**のパイプライン（QLoRA 学習 → GGUF 変換 → LM Studio / Ollama への登録 → キャラクターへの割当）が本体に内蔵されています。詳細な手順と VRAM の目安は [persona_qlora.md](persona_qlora.md) を参照してください。
- **モデルは各自で用意します。** 本パッケージに学習済みモデルは含まれません。学習には CUDA 対応 GPU と学習用依存（`requirements-train.txt`）が必要です。
- 学習した専用モデルは、`config.yaml` の `persona` セクションを有効化するか、キャラクターの `llm.model` に登録名を指定して使います。切替前に通常の会話で口調を確かめ、合わなければベースモデルに戻せるよう、元の設定は控えておいてください。
