# VRChatワールド対応：Unity専用スタジオアプリ開発とPython連携 — 実装計画

本計画は**「提案4: Unityでアバター・ワールド描画用の専用アプリを構築し、PythonからWebSocket経由で遠隔操作する」**アプローチに基づく実装計画です。

Unityエンジンを使用することで、VRChatのワールド（FBX/Unityアセット）をそのまま配置でき、Unityの標準機能である `NavMesh` を使った「障害物を避けた賢い自動歩行」や「物理演算（揺れもの）」「ライティング」「ギミック（椅子に座る等）」を完全に制御できます。

---

## 全体アーキテクチャ

```mermaid
graph TD
    subgraph Python Backend (FastAPI)
        main[main.py / WebSocket Server]
        orch[orchestrator.py]
        hub[ws_hub.py]
    end

    subgraph Unity Studio App (Windows Desktop)
        ws_client[Unity WebSocket Client]
        nav[NavMeshAgent / 経路探索]
        anim[AnimatorController / 歩行・表情]
        vrm_load[UniVRM / アバターロード]
    end

    orch -->|アクション指示| hub
    hub -->|WebSocket通信 /ws/unity| ws_client
    ws_client -->|歩行・座る指示| nav
    ws_client -->|リップシンク・表情| anim
```

---

## 提案する変更

### 1. Python バックエンド側の拡張
*   **`backend/avatar/ws_hub.py`**:
    *   接続チャネルに `"unity"` を追加します（既存は `dashboard`, `avatar`, `subtitle`）。
    *   Unityクライアントとの通信を可能にします。
*   **`backend/main.py`**:
    *   WebSocketエンドポイント `/ws/unity` を追加し、`hub` を経由して接続します。
    *   ダッシュボードやLLM（orchestrator）からの移動コマンド等をUnityクライアントへ転送するルートを作成します。
*   **`backend/orchestrator.py`**:
    *   LLM応答からアクションタグ（例: `[action: walk_to, target: chair]`）をパースし、WebSocket経由でUnityアプリへコマンドを送信する仕組みを実装します。

### 2. Unity アプリケーション用のC#スクリプトの提供
Unityプロジェクトにドラッグ＆ドロップするだけで動くC#スクリプト群を作成し、プロジェクトの `backend/studio_unity/` 等のディレクトリに保存します。
*   **`UnityWebSocketClient.cs`**:
    *   Pythonの WebSocket サーバー（`ws://localhost:9000/ws/unity`）に自動接続します。
    *   JSONコマンド（`move_to`, `set_emotion`, `play_animation`, `lipsync` 等）を受信し、アバター制御スクリプトへ転送します。
*   **`AvatarController.cs`**:
    *   `NavMeshAgent` を使って、指定された目的地（座標）へアバターを自動歩行させます。
    *   移動速度に応じて、待機（Idle）と歩行（Walk）のアニメーションをスムーズに切り替えます。
    *   目的地が「椅子」などのギミックオブジェクトの場合、そこへ移動した後に「座るポーズ（Sit）」を再生します。
*   **`LipsyncController.cs`**:
    *   音声データや音量パラメータを受け取り、VRMアバターの Expression（表情：`aa` 等の口の開き具合）をリアルタイム制御します。

---

## ユーザーが行う Unity プロジェクトの構築手順（ガイド）

Unityアプリを動かすため、以下の手順をまとめたテキストガイドを作成します：
1. **Unityプロジェクトの作成**: Unity 2021.3 LTS または 2022.3 LTS を使用して3Dプロジェクトを新規作成。
2. **パッケージの導入**:
   - `UniVRM` (VRMモデル読み込み用パッケージ)
   - `NativeWebSocket` または `websocket-sharp` (WebSocket通信用)
3. **ワールドとアバターの配置**:
   - VRChatワールドのメッシュ（FBXやモデルアセット）をシーンに配置。
   - 床に `NavMesh`（ナビゲーションメッシュ）をベイク（自動作成）して、歩行可能エリアを定義。
4. **スクリプトの適用**:
   - 提供するC#スクリプトをアバターや管理オブジェクトにアタッチ。
5. **実行**: Unityエディタで再生ボタンを押すと、Pythonバックエンドと自動接続され、AI VTuberがワールド内を自律的に動き回り始めます。

---

## 検討事項

> [!IMPORTANT]
> **アニメーションデータ（歩行・待機・座る）について**  
> アバターを歩かせる・座らせるためのアニメーションクリップ（FBXや`.anim`形式）が必要です。Unityの標準アセット（Starter Assets）や、Mixamo等のフリー素材から「Walk」「Idle」「Sit」のアニメーションをダウンロードし、UnityのAnimator Controllerに登録してもらう手順をガイドに明記します。

---

## 検証計画

### 手動検証
1. Unityエディタを起動し、テスト用の平面（床）と障害物を配置してNavMeshをベイク。
2. Pythonサーバー（`run.py`）を起動。
3. Unity側で再生を実行し、WebSocketの接続成功（`connected`）ログを確認。
4. PythonダッシュボードまたはAPIから `{"cmd": "move_to", "x": 5, "z": 3}` を送信し、Unity内のアバターが障害物を避けて目的地まで歩くことを確認。
5. 椅子オブジェクトを置いて「座る」コマンドが機能するか確認。
