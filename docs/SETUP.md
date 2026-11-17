# セットアップ手順

AI VTuber System を自分の環境で動かすまでの手順です。構成は「本体（Python）＋ ローカル LLM ＋ TTS サーバ」の 3 つが基本で、クラウド LLM や Web 検索は任意です。

## 1. 前提環境

- Windows 10 / 11（起動バッチを使う場合。`run.py` から起動すれば他 OS でも動きます）
- Python 3.11 以上
- ローカル LLM を使う場合: Ollama または LM Studio
- TTS: Irodori-TTS サーバ、または VOICEVOX
- ローカル LLM・TTS を快適に動かすための GPU（無くてもクラウド LLM 中心の構成は可能です）

## 2. 本体のインストール

```bash
git clone <このリポジトリ>
cd <クローン先>

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

人格 QLoRA の学習まで行う場合は、追加で学習用依存を入れます（CUDA 版 PyTorch が別途必要です）。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

## 3. ローカル LLM の用意

### Ollama を使う場合

1. Ollama をインストールして起動します（本体側で自動起動もできます。後述）。
2. モデルを取得します。

   ```bash
   ollama pull llama3.1:8b
   ```

3. `config.yaml` の `llm` セクションで、使用するバックエンドを主系にします。

   ```yaml
   llm:
     primary: ollama
     fallback: lmstudio
     backends:
       ollama:
         base_url: http://localhost:11434
         model: llama3.1:8b
   ```

Ollama の自動起動は `config.yaml` の `ollama:` セクションで制御します（`auto_start: true` で本体起動時に `ollama serve` を立ち上げ、`preload_on_start` でモデルを先読みします）。

### LM Studio を使う場合

1. LM Studio を起動し、使いたいモデルをロードしてローカルサーバ（既定 http://localhost:1234/v1）を有効にします。
2. `config.yaml` の `llm.backends.lmstudio.model` に、ロードしたモデルの識別子を書きます（同梱の `config.yaml` ではこちらが主系になっています）。

### クラウド LLM（任意・即応用）

ハイブリッド構成では、コメント返答などの即応をクラウド LLM に振ります。

```yaml
llm:
  hybrid:
    enabled: true
    fast_backend: gemini   # 即応系に使うバックエンド名
```

使用するサービスの API キーを `.env` に設定してください（§5）。クラウド側が不調なときは自動でローカルが受けるので、キーが無くても配信自体は動きます。

## 4. TTS の用意

### Irodori-TTS を使う場合（推奨）

Irodori-TTS は参照音声から声を再現する TTS です。サーバ（Irodori-TTS-Server）を別途用意し、本体から自動起動させることができます。

1. Irodori-TTS-Server を任意のフォルダに配置します。
2. `config.yaml` で配置先と自動起動を指定します。

   ```yaml
   tts:
     primary: irodori
     irodori_server:
       auto_start: true
       dir: Irodori-TTS/Irodori-TTS-Server   # サーバのフォルダ（相対・絶対どちらでも可）
   ```

3. 声は、サーバ側の `voices/` フォルダに参照音声（.wav）を置き、ファイル名（拡張子なし）をキャラクター設定の `voice` に書くことで使います。詳しくは [CHARACTER_GUIDE.md](CHARACTER_GUIDE.md) を参照してください。
4. モデルや精度などサーバ側の設定は、サーバ側の `.env` で行います。

既定では http://localhost:8088/v1 に接続します。サーバを手動で起動する運用でも構いません（その場合は `auto_start: false`）。

### VOICEVOX を使う場合

1. VOICEVOX を起動します（既定 http://localhost:50021）。
2. `config.yaml` の `tts.primary` を `voicevox` にし、キャラクター設定で `speaker_id` を指定します。

## 5. .env の設定

`.env.example` を `.env` にコピーし、使う機能の項目だけ記入します。`config.yaml` 側には環境変数名（`api_key_env`）だけを書く設計で、キーの実値は `.env` と環境変数から読みます。

| 変数 | 用途 | 必須条件 |
|---|---|---|
| `YOUTUBE_API_KEY` | YouTube Data API（pytchat のフォールバック） | Data API でコメント取得する場合 |
| `GEMINI_API_KEY` | Gemini（ハイブリッドの即応系など） | Gemini バックエンドを使う場合 |
| `ANTHROPIC_API_KEY` | Claude バックエンド | Claude バックエンドを使う場合 |
| `OPENAI_API_KEY` | OpenAI バックエンド | OpenAI バックエンドを使う場合 |
| `GOOGLE_SEARCH_API_KEY` / `GOOGLE_CSE_ID` | Web 検索（Google Custom Search） | 検索プロバイダに Google を使う場合 |
| `BRAVE_SEARCH_API_KEY` | Web 検索（Brave Search） | 検索プロバイダに Brave を使う場合 |
| `TWITCASTING_ACCESS_TOKEN` | ツイキャスのコメント取得 | ツイキャスを使う場合 |
| `TWITCH_OAUTH_TOKEN` | Twitch のコメント取得 | Twitch を使う場合 |

`.env` は `.gitignore` で除外されています。コミットしないでください。

## 6. config.yaml の要点

| セクション | 要点 |
|---|---|
| `server` | 本体の待受（既定 `127.0.0.1:9000`） |
| `youtube` | `video_id`（空ならダッシュボードから手動開始）、コメントソースの主系（`primary_source: pytchat`） |
| `llm` | バックエンド定義、`primary` / `fallback`、`hybrid`（即応系の振り分け） |
| `tts` | `primary` の選択、バックエンド定義、`irodori_server`（自動起動） |
| `characters` | 出演キャラクターの並び。1 体だけでも、複数並べて掛け合いでも可。書き方は [CHARACTER_GUIDE.md](CHARACTER_GUIDE.md) |
| `ensemble` | 掛け合いの確率・ターン上限・コメントへの譲り方 |
| `avatar` | 全体のアバター設定（種別は vrm / live2d / video）。キャラごとに上書きも可 |
| `orchestrator` | 自律発話の間隔（`idle_threshold_sec`）、記憶の保持量、ストリーミング TTS の有無など |
| `knowledge` | ニュース RSS、Web 検索、自律話題の重み、事実質問への回答方式 |
| `local_chat` | ローカルチャット（配信なしの対話）の設定。`autostart: true` で起動直後から会話できます |
| `stt` | マイク音声認識（faster-whisper）の有無とモデル |
| `persona` | 人格 QLoRA の設定。既定は `enabled: false`。使う場合だけ有効化します（[persona_qlora.md](persona_qlora.md)） |
| `safety` | 応答フィルタの有効化と NG ワード |
| `ollama` | Ollama の自動起動・先読み・自動 pull |

## 7. 起動方法

Windows では `start.bat`（または `起動.bat`）が最短です。依存の入った Python を自動で探し、サーバを起動してダッシュボードをブラウザで開きます。

```bat
start.bat
```

直接起動する場合:

```bash
python run.py
```

起動したら http://127.0.0.1:9000 がダッシュボードです。主な画面:

| パス | 内容 |
|---|---|
| `/` | ダッシュボード（開始・停止、設定、診断） |
| `/chat` | ローカルチャット（動作確認はまずここで） |
| `/avatar` | アバター画面（OBS 用） |
| `/subtitle` | 字幕画面（OBS 用） |
| `/persona` | 人格 QLoRA の管理 |

## 8. まずローカルで試す

配信前に、ローカルチャットで一通り確認するのが安全です。

1. 本体を起動し、`/chat` を開きます。
2. メッセージを送って、返答が返り、音声と字幕が出ることを確認します。
3. ダッシュボードの「ローカルサービス診断」で、Ollama / LM Studio / TTS の接続状態を確認できます。
4. マイクで話しかけたい場合は `stt.enabled: true`（既定で有効）で、ダッシュボードから音声入力を使います。

## 9. OBS への組み込み

1. OBS のソースに「ブラウザ」を 2 つ追加します。
   - アバター: `http://127.0.0.1:9000/avatar`
   - 字幕: `http://127.0.0.1:9000/subtitle`
2. 掛け合い（複数キャラクター）の場合は、アバターをキャラクターの数だけ追加し、URL を `http://127.0.0.1:9000/avatar?char=<キャラクターのid>` の形にします（省略時はリーダー）。
3. 音声は本体アプリが音声出力デバイスへ再生します。OBS では、そのデバイスを取り込む音声ソース（デスクトップ音声や仮想オーディオデバイス）を用意してください。出力先は `config.yaml` の `audio_output.device` で指定できます。
4. アバターの位置やキャリブレーション（腕の下げ量など）はダッシュボードから調整できます。

## 10. YouTube 配信との接続

1. YouTube Studio で配信枠を作成し、video_id（URL の `v=` 以降）を確認します。
2. ダッシュボードで video_id を設定して配信セッションを開始します（`config.yaml` の `youtube.video_id` に書いておくと、起動と同時に自動開始します）。
3. コメント取得は pytchat が既定で、API キーは不要です。YouTube Data API をフォールバックに使う場合は `YOUTUBE_API_KEY` を設定してください。
4. OBS から YouTube への送信開始と、Studio 側の「ライブ配信を開始」「ライブ配信を終了」の操作は、本システムの対象外です。各自の配信手順で行ってください。**終了時は、OBS の送信停止だけでなく Studio 側でも配信を終了してください**（枠がライブ状態のまま残ると、次回の配信に影響します）。

Twitch / ツイキャスも `config.yaml` の該当セクションと `.env` のトークンで接続できます。

## 11. Unity 連携

同梱の Unity プロジェクト（`AIVunity/`）と `backend/studio_unity/` の C# スクリプトを使うと、VRM アバターを Unity 側で描画できます。

- **UniVRM は本パッケージに同梱していません。** VRM を扱うには、UniVRM を UPM（Unity Package Manager）経由で別途導入してください。未導入のままではプロジェクトが開けない場合があります。
- 既存の Unity プロジェクトへ組み込む場合は、`Unity連携セットアップ.bat`（`backend/studio_unity/setup_unity_project.py`）が C# スクリプトと通信設定の導入を支援します。
- 手順の詳細は [docs/vrchat_world_integration/unity_setup_guide.md](vrchat_world_integration/unity_setup_guide.md) を参照してください。

## 12. トラブルシューティングの基本

| 症状 | 確認先 |
|---|---|
| ダッシュボードが開かない | ポート 9000 が他プロセスに使われていないか。`start.bat` は起動済みならブラウザだけ開きます |
| ローカルサービスが軒並み未接続 | ダッシュボードの「ローカルサービス診断」（`/api/diagnose`）で、Ollama / LM Studio / TTS のどれが落ちているか切り分ける |
| LLM が応答しない・空応答 | 使用中バックエンドの `base_url` とモデル名が、実際にロードされているモデルと一致しているか。Ollama で思考過程だけ返るモデルは `extra_payload` の `think: false` と `format: json`（同梱 config の ollama 定義参照）を確認 |
| 声が出ない（字幕だけ出る） | TTS サーバが起動しているか（Irodori は `tts.irodori_server.dir` の指す先が正しいか）。TTS が連続失敗すると字幕のみモードに入る設計で、クールダウン後に自動復帰します |
| 声がキャラクターと違う | キャラクター設定の `voice`（Irodori の参照音声名）/ `speaker_id`（VOICEVOX）が、そのキャラクターのものになっているか |
| コメントが拾えない | video_id が正しいか、配信枠がライブ状態か。pytchat はライブ開始前・終了後の枠では接続できません |
| 自律発話が始まらない | `orchestrator.idle_threshold_sec` の間隔と、`knowledge.autonomous_topic.enabled` を確認 |
| ログを見たい | 本体のコンソール出力のほか、自動起動した外部サーバのログは `logs/` に出ます |

問題の切り分けでは、まず `/chat` のローカルチャットで「生成→発話」が成立するかを確認し、成立するなら配信側（コメント取得・OBS）の問題、成立しないなら LLM / TTS 側の問題、と二分するのが近道です。
