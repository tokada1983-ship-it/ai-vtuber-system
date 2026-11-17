# VRChatワールド対応：Unity専用スタジオアプリ開発とPython連携 — タスクリスト

## 1. Python バックエンドの拡張
- [ ] `backend/avatar/ws_hub.py` に `"unity"` チャネルを追加する
- [ ] `backend/main.py` に WebSocket エンドポイント `/ws/unity` を追加し、Unityクライアントとのメッセージ送受信を可能にする
- [ ] `backend/orchestrator.py` に LLM 応答内のアクションタグ（例: `[action: walk_to, target: chair]`）をパースし、WebSocket 経由でUnityアプリへコマンドを転送するロジックを実装する
- [ ] `backend/config.py` に `WorldConfig` を追加し、Unity連携関連の設定（モデルパス、スポーンポイント等）が可能な枠組みを用意する

## 2. Unity用 C# スクリプトの作成と配置
- [ ] `backend/studio_unity/` ディレクトリを作成する
- [ ] `UnityWebSocketClient.cs`（WebSocket接続とコマンド受信中継）を作成・配置する
- [ ] `AvatarController.cs`（NavMeshAgentを用いた歩行およびアニメーション切り替え制御、ギミック座る等）を作成・配置する
- [ ] `LipsyncController.cs`（音声音量パラメータによるVRMリップシンク制御）を作成・配置する

## 3. Unity プロジェクト構築・連携ガイドの作成
- [ ] `docs/vrchat_world_integration/unity_setup_guide.md` を作成し、パッケージインポート（UniVRM, NativeWebSocket）、シーン構成、NavMesh設定、C#スクリプト適用、Animator Controller設定等の詳細な手順を記載する

## 4. 結合テストと調整
- [ ] Python バックエンドを起動し、作成したC#スクリプトをアタッチしたUnityサンプルプロジェクトを再生してWebSocket接続が確立されることをテストする
- [ ] Pythonから移動コマンドなどを送信し、Unity内のアバターが目的地へ移動することを確認する
- [ ] LLMテキストからのアクションパースとアバター行動開始の自動連動テストを実施する
