# Unityスタジオアプリの構築とPython連携ガイド

本ガイドは、提供されたC#スクリプトを使用して、Unityでアバターとワールド（背景）を読み込み、PythonシステムとWebSocketで連携するためのセットアップ手順です。

---

## 1. Unityプロジェクトの準備

1. **Unity Hub** を起動し、新しいプロジェクトを作成します。
   - 推奨バージョン: **Unity 2021.3 LTS** または **2022.3 LTS**
   - テンプレート: **3D (Built-in Render Pipeline)** または **3D (URP)** を選択。
2. 作成したUnityプロジェクトを開きます。

---

## 2. 自動セットアップツールの使用（推奨・最も簡単）

プロジェクトのルートフォルダにある **`Unity連携セットアップ.bat`** を使用すると、C#スクリプトの配置と通信用WebSocketパッケージのインストール設定を全自動で行えます。

1. **`Unity連携セットアップ.bat`** をダブルクリックして起動します。
2. プロンプトが表示されたら、作成した **Unityプロジェクトのルートフォルダのパス**（`Assets` フォルダや `Packages` フォルダがあるフォルダ）を入力し、Enterキーを押します。
   - スクリプトが自動的に `Assets/Scripts/AIVtuber/` を作成して3つのC#ファイルを配置します。
   - `Packages/manifest.json` に通信用の `NativeWebSocket` パッケージの設定を自動追記します。
3. Unityエディタに戻ると、自動的に通信パッケージのダウンロードとインポートが実行されます。

---

## 3. 追加パッケージ（UniVRM）の導入

アバターのVRMロードを行うために、以下のパッケージを手動でインポートします。

### A. UniVRM (アバター読み込みパッケージ)
VRMアバターをUnityで扱うための公式パッケージです。
1. [UniVRMのリリース一覧](https://github.com/vrm-c/UniVRM/releases) から、ご自身のVRMモデルのバージョンに応じた `.unitypackage` をダウンロードします。
   - VRM 0.x モデルの場合: `UniVRM-0.XXX.unitypackage`
   - VRM 1.0 モデルの場合: `VRM10-0.XXX.unitypackage`
2. ダウンロードしたファイルをUnityエディタの Project ウィンドウにドラッグ＆ドロップし、すべてのファイルをインポートします。
- `LipsyncController.cs`

---

## 3. シーンの構成 (ワールドとアバターの配置)

### A. ワールド（背景）のインポートと配置
1. VRChat用に用意したワールドデータ（FBX、OBJ、GLB等）をUnityプロジェクトにインポートし、シーン上に配置します。
2. 配置したワールドモデルのゲームオブジェクトを選択し、Inspectorウィンドウの右上にある **Static** チェックボックスにチェックを入れます（NavigationStaticとして扱えるようにするため）。

### B. 歩行エリア（NavMesh）のベイク（自動作成）
アバターが壁や障害物を避けて歩けるように、床の歩行可能エリアを定義します。
1. メニューから `Window > AI > Navigation` を開き、Navigationウィンドウを表示します。
2. Navigationウィンドウの **Bake** タブを開き、右下の **Bake** ボタンをクリックします。
   - シーン上の床や障害物に基づいて、歩行可能なエリア（青いメッシュ）が自動生成されます。

### C. アバター（VRM）のインポートと設定
1. ご使用の `.vrm` アバターファイルをUnityプロジェクトの `Assets` にインポートします（自動的にプレハブに変換されます）。
2. プレハブアセットをシーン上（ワールドの出現地点）にドラッグ＆ドロップして配置します。
3. シーン上のアバターオブジェクトを選択し、Inspectorウィンドウで以下を設定します：
   - **NavMeshAgent** コンポーネントを追加します (`Add Component > NavMeshAgent`)。
     - *設定パラメータの推奨値*: `Speed` = 2.0 (歩行速度), `Angular Speed` = 360 (方向転換速度), `Acceleration` = 8.0, `Stopping Distance` = 0.1
   - **LipsyncController** スクリプトを追加します (`Add Component > LipsyncController`)。
   - **AvatarController** スクリプトを追加します (`Add Component > AvatarController`)。
     - `Animator` および `Agent` フィールドに、アバター自身のAnimatorとNavMeshAgentをアタッチします。

---

## 4. Animator Controller の作成と設定

アバターの「待機」「歩行」「座る」などのアニメーションを制御するための設定を行います。

1. Project ウィンドウで右クリックし、`Create > Animator Controller` を選択して名前を `AvatarAnimator` とします。
2. アバターオブジェクトを選択し、その `Animator` コンポーネントの `Controller` フィールドに `AvatarAnimator` をアタッチします。
3. `AvatarAnimator` をダブルクリックして Animator ウィンドウを開き、以下のパラメータを追加します：
   - **Speed** (Float): 移動アニメーション用
   - **Sit** (Trigger): 椅子に座る用
   - **Stand** (Trigger): 立ち上がる用
   - **Emotion_happy**, **Emotion_sad**, **Emotion_angry** などの各種感情 (Trigger)
4. Animator のステート（State）を作成し、それぞれのアニメーションクリップ（待機、歩行、着席）を登録します：
   - `Speed` パラメータが 0 より大きくなったら「歩行（Walk）」へ遷移、0 になったら「待機（Idle）」へ遷移するトランジション（移行矢印）を作成します。
   - `Sit` トリガーが引かれたら「着席（Sit）」へ遷移し、`Stand` トリガーで「待機（Idle）」へ戻るトランジションを作成します。

> [!TIP]
> **アニメーション素材について**  
> 「Walk」「Idle」「Sit」などの基本アニメーションは、Unity標準のスターターアセットや、[Mixamo](https://www.mixamo.com/) などの無料のモーションアセットサイトからダウンロードしてインポートできます。

---

## 5. 通信マネージャーの作成と接続テスト

1. シーン上に空のゲームオブジェクトを作成し、名前を `WebSocketManager` とします。
2. `WebSocketManager` に **UnityWebSocketClient** スクリプトをアタッチします。
3. スクリプトのパラメータを設定します：
   - `Server Url` = `ws://localhost:9000/ws/unity` (デフォルトのままでOK)
   - `Avatar Controller` = シーン上のアバターオブジェクトをアタッチ。
   - `Lipsync Controller` = シーン上のアバターオブジェクトをアタッチ。

---

## 6. 動かし方

1. 先に Python バックエンド（`起動.bat` または `run.py`）を起動します。
2. Unityエディタに戻り、再生（Play）ボタンを押します。
3. Unityの Console ウィンドウに `[WebSocketClient] Connected to python server successfully.` と表示されれば、Python側との通信連携に成功です！

### テストコマンドの送信例
ダッシュボード（Web）のコンソールから、あるいはテスト用にWebSocketで以下のJSONコマンドを送信することで、Unity内のアバターを遠隔操作できます。
- **移動**: `{"cmd": "unity_move", "x": 3.0, "y": 0.0, "z": 5.0}` (座標 3.0, 0, 5.0 へNavMeshで歩いて移動)
- **座る**: `{"cmd": "unity_action", "action": "sit", "target": "Gimmick_Chair"}`
- **立ち上がる**: `{"cmd": "unity_action", "action": "stand"}`
- **LLMでの自律動作**: LLM（Character1等）の発話中に `[action: walk_to, target: 1.0,0,3.5]` のような文字列が含まれていると、AI VTuberは自動的にその指示を解釈し、歩き出します。
