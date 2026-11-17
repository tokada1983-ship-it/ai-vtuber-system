import os
import shutil
import json
import sys
import glob
import subprocess

BAKE_SCRIPT_CONTENT = """using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

public static class NavMeshBakeHelper
{
    [MenuItem("AIVtuber/Bake NavMesh Automatically")]
    public static void Build()
    {
        Debug.Log("[AIVtuber] Starting automatic NavMesh baking...");

        // 1. 現在のシーンを開く、無ければ最初のシーンを探す
        string activeScenePath = EditorSceneManager.GetActiveScene().path;
        if (string.IsNullOrEmpty(activeScenePath))
        {
            string[] sceneGuids = AssetDatabase.FindAssets("t:Scene");
            if (sceneGuids.Length > 0)
            {
                activeScenePath = AssetDatabase.GUIDToAssetPath(sceneGuids[0]);
                EditorSceneManager.OpenScene(activeScenePath);
                Debug.Log("[AIVtuber] Opened scene: " + activeScenePath);
            }
            else
            {
                Debug.LogError("[AIVtuber] No scene found in project to bake NavMesh.");
                return;
            }
        }

        // 2. メッシュを持つ全オブジェクトを自動で Navigation Static に設定する
        var renderers = Object.FindObjectsOfType<MeshRenderer>();
        int staticCount = 0;
        foreach (var r in renderers)
        {
            // 床・壁・障害物をベイク対象にするため static フラグを設定
            GameObjectUtility.SetStaticEditorFlags(r.gameObject, 
                StaticEditorFlags.ContributeGI | 
                StaticEditorFlags.OccluderStatic | 
                StaticEditorFlags.BatchingStatic | 
                StaticEditorFlags.NavigationStatic | 
                StaticEditorFlags.OffMeshLinkGeneration);
            staticCount++;
        }

        var terrains = Object.FindObjectsOfType<Terrain>();
        foreach (var t in terrains)
        {
            GameObjectUtility.SetStaticEditorFlags(t.gameObject, StaticEditorFlags.NavigationStatic);
            staticCount++;
        }

        Debug.Log("[AIVtuber] Set " + staticCount + " objects to Navigation Static.");

        // 3. NavMeshをビルドする
        UnityEditor.AI.NavMeshBuilder.ClearAllNavMeshes();
        UnityEditor.AI.NavMeshBuilder.BuildNavMesh();
        Debug.Log("[AIVtuber] NavMesh baking completed successfully.");

        // 4. 保存
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
        AssetDatabase.SaveAssets();
        Debug.Log("[AIVtuber] Scene and assets saved successfully.");
    }
}
"""

def find_unity_executable():
    """PC内の最新の Unity.exe のパスを探す"""
    paths = [
        r"C:\Program Files\Unity\Hub\Editor\*\Editor\Unity.exe",
        r"C:\Program Files\Unity\Editor\Unity.exe",
        r"D:\Program Files\Unity\Hub\Editor\*\Editor\Unity.exe",
        r"E:\Program Files\Unity\Hub\Editor\*\Editor\Unity.exe",
    ]
    for p in paths:
        matches = glob.glob(p)
        if matches:
            return sorted(matches)[-1]
    return None

def setup_unity(unity_path):
    unity_path = unity_path.strip().strip('"').strip("'")
    if not os.path.exists(unity_path):
        print(f"[エラー] 指定されたパスが存在しません: {unity_path}")
        return False
    
    assets_path = os.path.join(unity_path, "Assets")
    packages_path = os.path.join(unity_path, "Packages")
    manifest_file = os.path.join(packages_path, "manifest.json")
    
    if not os.path.exists(assets_path) or not os.path.exists(manifest_file):
        print("[エラー] 指定されたフォルダは有効なUnityプロジェクトではないようです。")
        print("ルートフォルダ（Assets や Packages フォルダがある場所）を指定してください。")
        return False

    print("\n[1/4] C#スクリプトをコピーしています...")
    # アバター用スクリプトの配置
    target_scripts_dir = os.path.join(assets_path, "Scripts", "AIVtuber")
    os.makedirs(target_scripts_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_to_copy = ["UnityWebSocketClient.cs", "AvatarController.cs", "LipsyncController.cs"]
    
    for script in scripts_to_copy:
        src = os.path.join(current_dir, script)
        dst = os.path.join(target_scripts_dir, script)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f" -> コピー完了: Assets/Scripts/AIVtuber/{script}")
        else:
            print(f" -> [警告] コピー元ファイルが見つかりません: {src}")

    # エディタ用自動ベイクスクリプトの配置
    target_editor_dir = os.path.join(assets_path, "Editor", "AIVtuber")
    os.makedirs(target_editor_dir, exist_ok=True)
    bake_script_file = os.path.join(target_editor_dir, "NavMeshBakeHelper.cs")
    with open(bake_script_file, "w", encoding="utf-8") as f:
        f.write(BAKE_SCRIPT_CONTENT)
    print(" -> エディタ拡張を配置しました: Assets/Editor/AIVtuber/NavMeshBakeHelper.cs")

    print("\n[2/4] WebSocketパッケージを manifest.json に追加しています...")
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        dependencies = manifest.get("dependencies", {})
        
        package_key = "com.github.endel.nativewebsocket"
        package_val = "https://github.com/endel/NativeWebSocket.git"
        
        if package_key not in dependencies:
            dependencies[package_key] = package_val
            manifest["dependencies"] = dependencies
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(" -> manifest.json に NativeWebSocket を追加しました。")
        else:
            print(" -> NativeWebSocket は既に manifest.json に存在します。")
    except Exception as e:
        print(f" -> [警告] manifest.json の書き換えに失敗しました: {e}")

    print("\n[3/4] Unity.exe を探索して自動ベイクを実行します...")
    unity_exe = find_unity_executable()
    if not unity_exe:
        print(" -> [警告] PC内に Unity.exe (Editor) が見つかりませんでした。")
        print("    自動ベイク処理はスキップします。手動でUnityを開いて「AIVtuber」メニューからベイクしてください。")
    else:
        print(f" -> Unity検出: {unity_exe}")
        print(" -> バックグラウンドでUnityを起動し、自動ベイク処理を開始します...")
        print("    ※ 実行には1〜3分程度かかる場合があります。Unityエディタを閉じた状態で実行してください。")
        
        # Unityバッチ実行コマンド
        cmd = [
            unity_exe,
            "-projectPath", unity_path,
            "-batchmode",
            "-nographics",
            "-executeMethod", "NavMeshBakeHelper.Build",
            "-quit",
            "-logFile", "-"
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
            for line in process.stdout:
                line_str = line.strip()
                # 重要な進捗ログのみ絞って出力
                if "[AIVtuber]" in line_str or "Error" in line_str or "Exception" in line_str:
                    print(f"    [Unity] {line_str}")
            process.wait()
            if process.returncode == 0:
                print(" -> [成功] NavMeshの自動ベイクが完全に完了しました！")
            else:
                print(f" -> [警告] Unityがエラーコード {process.returncode} で終了しました。")
        except Exception as e:
            print(f" -> [エラー] 自動ベイクの実行中にエラーが発生しました: {e}")

    print("\n[4/4] 完了処理中...")
    print("=" * 60)
    print(" セットアップおよび自動ベイク処理が完了しました！")
    print(" ※ VRMアバターのロードには UniVRM のインストールが別途必要です。")
    print("=" * 60)
    return True

if __name__ == "__main__":
    print("==================================================")
    print("       AI VTuber Unityプロジェクト連携スクリプト")
    print("==================================================")
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Unityプロジェクトのフォルダパスを入力してください: ")
        
    setup_unity(path)
    input("\nEnterキーを押して終了します...")
