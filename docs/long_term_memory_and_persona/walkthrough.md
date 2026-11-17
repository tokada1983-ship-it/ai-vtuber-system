# 長期記憶保持＆人格維持システム (A・B・C統合) — ウォークスルー

## 概要
AI VTuberが「何ヶ月もリスナーを覚え続け」「配信を通じて自身の思い出を深め」「どれだけ記憶が増えても絶対にキャラが崩れない」を実現するため、提案した3つの柱（A・B・C）を完全統合したシステムを構築しました。

---

## 変更・追加されたモジュール一覧

| カテゴリ | モジュール / ファイル | 変更区分 | 役割・機能 |
|---|---|---|---|
| **A: リスナー関係性** | `backend/memory/viewer_store.py` | **新規** | 視聴者ごとの来訪回数・初回/最終来訪日・親密度Lv1〜5・話題メモのSQLite管理 |
| **A: リスナー関係性** | `backend/memory/memory_store.py` | 拡張 | `memories` テーブルに `viewer_name` カラム追加。特定視聴者のエピソード検索に対応 |
| **A: リスナー関係性** | `backend/memory/memory_consolidator.py` | 拡張 | 記憶抽出時に視聴者名を検出し、`viewer_name` 紐付け＆`ViewerProfileStore` へメモ追記 |
| **B: 自己内省・思い出** | `backend/memory/reflection_engine.py` | **新規** | 配信終了時にその日の会話を振り返り、AI視点の日記を生成して `memory.md`（コア記憶）にローテーション追記 |
| **C: 人格死守** | `backend/llm/persona_validator.py` | **新規** | 発話直前の一人称（「私」「僕」→「わらわ」）や口調の崩れを自動検出・自動補正する安全弁 |
| **C: 人格死守** | `backend/llm/prompt_builder.py` | 拡張 | 最下部に「Constitutional Persona Anchor」を常駐配置。視聴者プロファイル枠の追加 |
| **全体統合** | `backend/orchestrator.py` | 拡張 | `ViewerProfileStore`, `ReflectionEngine`, `PersonaValidator` をライフサイクルに統合 |
| **設定** | `backend/config.py` & `config.yaml` | 拡張 | `enable_viewer_profiles`, `enable_reflection`, `enable_persona_validation` 等の項目追加 |
| **安定性** | `backend/logger.py` | 拡張 | `structlog` 未インストール環境でも標準 `logging` に安全フォールバックする耐性追加 |
| **テスト** | `tests/test_long_term_persona.py` | **新規** | 各モジュールの単体テスト（全件合格） |

---

## 主な機能詳細

### 1. リスナー関係性管理 (`ViewerProfileStore`)
- **親密度の自動ステップアップ**:
  - 来訪 1回: `Lv1 (初見)` — はじめてのリスナーとして歓迎しつつ丁寧に接する
  - 来訪 2〜4回: `Lv2 (顔見知り)`
  - 来訪 5〜9回: `Lv3 (常連)`
  - 来訪 10〜19回: `Lv4 (大常連)`
  - 来訪 20回以上: `Lv5 (古参・盟友)` — 強い信頼とくだけた親愛感を表現
- **プロンプトへの自動注入**:
  コメント受信時、以下のようなコンテキストが自動でLLMに渡されます：
  ```markdown
  ### コメント相手の情報
  - 呼称: アリス (関係性: 常連 / 親密度Lv3)
  - 累計来訪回数: 5回
  - 相手に関する記憶: 猫が好きと言っていた, 新しいキーボードを買った
  ```

### 2. 自己内省・振り返り日記 (`ReflectionEngine`)
- 配信停止時（`stop()` が呼ばれた時）、その日の会話ログとエピソード記憶から、キャラクターの一人称視点で素直な感想・日記をLLMが100〜200文字で執筆。
- コア記憶 `backend/llm/prompts/characters/{キャラ名}/memory.md` の「## 最近の思い出・日記」セクションにアトミック保存。
- 設定した件数（既定5件）を超えると古い日記は自動ローテーションされ、コンテキスト超過を防ぎながら最新の思い出を維持。
- 次回の配信開始時や独り言（`_speak_idle`）で、これらの思い出が自然と話題に登ります。

### 3. 人格死守（Constitutional Persona Anchor & Validator）
- **プロンプト最下部アンカー**:
  システムプロンプトの最後、出力フォーマットの直前に「不変の憲法（Constitutional Rule）」を配置。過去の記憶やユーザー発言の長文に流されても、LLMのアテンションをキャラクターの核に引き戻します。
- **発話前の自動バリデータ**:
  万が一LLMが「私は〜」「僕が〜」と出力した場合でも、発話・TTS音声合成の直前に「わらわは〜」「わらわが〜」へ自動置換・補正。現代語の語尾（「〜ですね」）も「〜じゃな」へ自然補正し、キャラ崩れを水際で阻止します。

---

## 検証結果

`tests/test_long_term_persona.py` を作成し、全4テストケースを実行しました。

```
Ran 4 tests in 0.109s
OK
```

1. ✅ `test_viewer_profile_store`: 初回来訪から常連化、メモ重複制御、プロンプト生成が正常に動作
2. ✅ `test_persona_validator`: 一人称逸脱（私・僕 → わらわ）および語尾の自動修復が正常に動作
3. ✅ `test_memory_store_viewer_name`: 視聴者紐づきエピソードの保存と検索（`by_viewer`）が正常に動作
4. ✅ `test_reflection_engine_diary_append`: `memory.md` への日記セクション追記と上限ローテーションが正常に動作
