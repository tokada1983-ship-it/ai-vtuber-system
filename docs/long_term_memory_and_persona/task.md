# 長期記憶保持＆人格維持システム — タスクリスト

## フェーズ 1: A. リスナー関係性管理 (ViewerProfileStore)
- [x] `backend/memory/viewer_store.py` の新規作成（SQLiteベースの視聴者プロファイル管理）
  - 来訪回数、初回/最終来訪日時、親密度レベル（1〜5）、あだ名、特徴メモの保存・取得
  - プロンプト用の関係性テキスト生成（例: `[視聴者: 〇〇さん (常連・親密度Lv3)] 前回の話題: 新PC購入`）
- [x] `backend/memory/memory_store.py` の拡張
  - `memories` テーブルに `viewer_name` カラム追加（特定視聴者のエピソード紐付け）
- [x] `backend/memory/memory_consolidator.py` の拡張
  - コメント発言者（author）に紐づく情報を抽出した際、`viewer_name` をセットして保存

## フェーズ 2: B. 自己内省・思い出深化 (ReflectionEngine)
- [x] `backend/memory/reflection_engine.py` の新規作成
  - 配信会話ログ（ConversationLog）と最新エピソード記憶から、キャラクター視点の「配信日記・振り返り」をLLMで生成
  - `memory.md`（コア記憶）に `## 最近の思い出・日記` セクションを設け、最新3〜5日分をローテーション追記
- [x] 独り言（`_speak_idle`）へのお題提供
  - 最新の日記・内省内容をお題プロンプトに反映し、過去の出来事について自律的に語れるようにする

## フェーズ 3: C. 人格死守 (Persona Anchor & Validator)
- [x] `backend/llm/prompt_builder.py` の強化
  - システムプロンプト最下部に「Constitutional Persona Anchor」を設置（一人称、語尾、根本世界観を最優先死守）
- [x] `backend/llm/persona_validator.py` の新規作成
  - 発話前の一人称崩れ検出（例: 「わらわ」が「私」「僕」になっていないか）と自動補正
  - キャラクター設定に応じた口調・語尾の検証

## フェーズ 4: バックエンド統合 & 設定
- [x] `backend/config.py` & `config.yaml` の更新
  - `enable_viewer_profiles`, `enable_reflection`, `max_diary_entries`, `enable_persona_validation` を追加
- [x] `backend/orchestrator.py` への統合
  - コメント受信時のプロファイル参照・来訪更新
  - 発話前の `PersonaValidator` 実行
  - `stop()` 時の内省（日記生成）パイプライン組み込み

## フェーズ 5: 単体テスト・検証
- [x] 各モジュールの単体テスト作成・実行 (`tests/test_long_term_persona.py`)
  - 視聴者プロファイルの蓄積・親密度テスト (パス)
  - 一人称・語尾補正のバリデーションテスト (パス)
  - 視聴者紐づきエピソード記憶テスト (パス)
  - 配信終了時リフレクションの生成・ローテーションテスト (パス)
- [x] `walkthrough.md` の作成と報告
