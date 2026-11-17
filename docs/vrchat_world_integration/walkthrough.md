# VRChatワールド対応：Unity専用スタジオアプリ開発とPython連携 — ウォークスルー

## 変更内容のまとめ

### 1. Python バックエンドの拡張
*   **`backend/avatar/ws_hub.py`**:
    *   `"unity"` 接続用のチャンネルを追加。
    *   移動指示 (`send_unity_move`)、ギミック指示 (`send_unity_gimmick`)、汎用アクション指示 (`send_unity_action`) を追加。
    *   既存の `send_speech_start` や `send_motion` 時にも Unity へ同時にパラメータをブロードキャストするよう拡張（表情とリップシンクの完全連動）。
*   **`backend/main.py`**:
    *   WebSocketメッセージハンドラ `_handle_ws_cmd_inner` に、Unity用の遠隔操作コマンド（`unity_move`, `unity_gimmick`, `unity_action`）を追加。受信したパラメータを `hub` を通じてUnityクライアントへ中継。
*   **`backend/orchestrator.py`**:
    *   LLM（AI）の思考や発話の出力から、アクション指示タグ（例: `[action: walk_to, target: chair]` や `[アクション: 椅子に座る]`）を自動パースする `_extract_unity_actions` メソッドを実装。
    *   読み上げ音声テキスト（`speech`）からはアクションタグを除去した上でTTSへ流し、それと同時に非同期で `_handle_unity_actions` を呼び出してWebSocket経由でUnity内のキャラクターに「歩行開始」や「着席」を指示するロジックを統合。
*   **`backend/config.py` & `config.yaml`**:
    *   `world` 設定（`enabled`, `model_path`, `spawn_point` 等）のモデルとデフォルト値を定義。

### 2. Unity用 C# スクリプトの生成と配置 (backend\studio_unity\)
- **`UnityWebSocketClient.cs`**:  
  Unity標準の `System.Net.WebSockets` ライブラリ（外部アセット不要）を使用し、PythonのサーバーとWebSocketで常時接続。受信したJSON形式のコマンドをスレッドセーフなキューに蓄積し、Unityのメインスレッド上でアバター制御スクリプトへディスパッチする。
- **`AvatarController.cs`**:  
  Unityの `NavMeshAgent` を使って、指定された目的地（座標）までアバターを自動歩行させる。障害物回避や、速度に応じた歩行・待機アニメーションの自動ブレンド、椅子オブジェクトへの接近および座るモーション（`sit` / `stand`）のトリガー処理を制御する。
- **`LipsyncController.cs`**:  
  アバター（VRM）のリップシンク（口の開閉量 A, I, U, E, O / aa）を制御する。VRM 0.x (`VRMBlendShapeProxy`) および VRM 1.0 (`VRM10Instance`) の両規格をリフレクションを用いて動的に判別して動作する。音声がない場合は発話時間から口パクをシミュレートする機能付き。

### 3. ドキュメントの整備
- **[unity_setup_guide.md](unity_setup_guide.md)**:  
  Unity Hubでのプロジェクト新規作成から、UniVRMパッケージの導入、床のNavMeshのベイク、3Dワールド配置、アバターへの提供C#スクリプトの適用、Animator Controllerのパラメータ設定までの全手順を解説した詳細なガイドを作成。

### 3. 自動セットアップツール (新規追加)
*   **`Unity連携セットアップ.bat` (プロジェクトルート)**:  
  ユーザーがダブルクリックしてUnityプロジェクトのフォルダパスをコピペするだけで、自動で全設定を行ってくれるバッチファイル。
*   **`setup_unity_project.py` (backend/studio_unity/内)**:  
  バッチから呼び出される実体。指定されたUnityの `Assets/Scripts/AIVtuber/` 以下に自動で3つのC#スクリプトをコピー・配置し、さらにUnityのパッケージ管理ファイル `manifest.json` を書き換えて通信用の `NativeWebSocket` パッケージを自動ダウンロードするように追加設定する。

---

## 構文チェックと検証結果

- Python側の全変更ファイル（`ws_hub.py`, `main.py`, `orchestrator.py`, `config.py`）について、コンパイル構文チェックを実行し、エラーのない正常なコードであることを確認済みです。

---

## 運用方法

1. **Unityプロジェクトのセットアップ**: プロジェクトのルートにある **`Unity連携セットアップ.bat`** を起動し、Unityのプロジェクトフォルダのパスを入力します（自動セットアップ完了）。
2. **Pythonサーバー起動**: 通常通り `起動.bat` 等でバックエンドを立ち上げます。
3. **Unityアプリ起動**: 作成したUnityプロジェクトで再生（Play）すると、自動的にWebSocket接続され、アバターがスタンドバイ状態になります。
4. **AIとの会話**: AI VTuberが「ちょっと椅子に座るね [action: sit, target: Gimmick_Chair_01]」と発言すると、自動的にその椅子に向けてアバターが歩き出し、椅子にぴったり座る一連のモーションがリアルタイムで発生します。
