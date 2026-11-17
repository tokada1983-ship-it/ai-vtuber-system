using System.Collections;
using UnityEngine;

public class LipsyncController : MonoBehaviour
{
    [Header("Lipsync Settings")]
    [Range(0f, 1f)] [SerializeField] private float maxOpenValue = 0.8f;
    [SerializeField] private float talkSpeed = 15f;

    // VRM 1.0 / 0.x のコンポーネントをリフレクションまたは動的キャストで制御するための変数
    private Component _vrm0Proxy = null;
    private Component _vrm10Instance = null;
    private bool _hasCheckedComponents = false;

    private Coroutine _lipsyncCoroutine = null;
    private float _targetLipsync = 0f;
    private float _currentLipsync = 0f;

    private void Start()
    {
        CheckVRMComponents();
    }

    private void CheckVRMComponents()
    {
        if (_hasCheckedComponents) return;

        // VRM 0.x の BlendShapeProxy を検出
        _vrm0Proxy = GetComponent("VRMBlendShapeProxy");
        
        // VRM 1.0 の VRM10Instance を検出
        _vrm10Instance = GetComponent("VRM10Instance");

        if (_vrm0Proxy != null)
        {
            Debug.Log("[LipsyncController] Detected VRM 0.x BlendShapeProxy component.");
        }
        else if (_vrm10Instance != null)
        {
            Debug.Log("[LipsyncController] Detected VRM 1.0 VRM10Instance component.");
        }
        else
        {
            Debug.LogWarning("[LipsyncController] Neither VRMBlendShapeProxy nor VRM10Instance was found on this GameObject. Lipsync will fallback to simulating jaw rotation (if applicable).");
        }

        _hasCheckedComponents = true;
    }

    /// <summary>
    /// 音声の代わりにテキストの長さに応じて口パクをシミュレートする
    /// </summary>
    public void StartLipsync(string text)
    {
        StopLipsync();
        
        float duration = Mathf.Max((text?.Length ?? 0) * 0.12f, 1.0f);
        _lipsyncCoroutine = StartCoroutine(LipsyncSimulateCoroutine(duration));
    }

    /// <summary>
    /// 音量などのパラメータから直接口パク量をセットする
    /// </summary>
    public void SetLipsyncValue(float value)
    {
        _targetLipsync = Mathf.Clamp(value, 0f, maxOpenValue);
    }

    public void StopLipsync()
    {
        if (_lipsyncCoroutine != null)
        {
            StopCoroutine(_lipsyncCoroutine);
            _lipsyncCoroutine = null;
        }
        
        _targetLipsync = 0f;
        _currentLipsync = 0f;
        ApplyLipsyncValue(0f);
    }

    private IEnumerator LipsyncSimulateCoroutine(float duration)
    {
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            
            // サイン波とノイズで自然な口の開閉をシミュレート
            float openAmount = (Mathf.Sin(elapsed * talkSpeed) * 0.4f) + 0.4f;
            openAmount *= maxOpenValue;
            
            // 口を閉じぎみにするランダムなタイミングを入れる
            if (Random.value < 0.05f)
            {
                openAmount = 0f;
            }

            ApplyLipsyncValue(openAmount);
            yield return null;
        }

        StopLipsync();
    }

    private void Update()
    {
        if (_lipsyncCoroutine == null)
        {
            // 直接値を指定された場合のスムーズ補間
            _currentLipsync = Mathf.Lerp(_currentLipsync, _targetLipsync, Time.deltaTime * talkSpeed);
            ApplyLipsyncValue(_currentLipsync);
        }
    }

    /// <summary>
    /// VRMの Expression/BlendShape に口の開き量 (aa / A) を適用する
    /// </summary>
    private void ApplyLipsyncValue(float value)
    {
        if (!_hasCheckedComponents) CheckVRMComponents();

        // 1. VRM 0.x 用の制御
        if (_vrm0Proxy != null)
        {
            ApplyVRM0Value(value);
            return;
        }

        // 2. VRM 1.0 用の制御
        if (_vrm10Instance != null)
        {
            ApplyVRM10Value(value);
            return;
        }
    }

    private void ApplyVRM0Value(float value)
    {
        try
        {
            // リフレクションを使って VRM0 の BlendShapeProxy.SetValue を安全に呼び出す
            // メソッドシグネチャ: SetValue(BlendShapeKey key, float value) 
            // または文字列指定の SetValue(string name, float value) を使用する
            var method = _vrm0Proxy.GetType().GetMethod("AccumulateValue", new[] { typeof(string), typeof(float) }) 
                         ?? _vrm0Proxy.GetType().GetMethod("SetValue", new[] { typeof(string), typeof(float) });
            
            if (method != null)
            {
                method.Invoke(_vrm0Proxy, new object[] { "Aa", value }); // "Aa" または "A" キー
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[LipsyncController] VRM0 SetValue failed: {e.Message}");
            _vrm0Proxy = null; // エラーが多発する場合は無効化
        }
    }

    private void ApplyVRM10Value(float value)
    {
        try
        {
            // VRM10Instance.Runtime.Expression.SetWeight
            // リフレクションで安全に ExpressionManager または Runtime にアクセス
            var runtimeProp = _vrm10Instance.GetType().GetProperty("Runtime");
            if (runtimeProp != null)
            {
                var runtime = runtimeProp.GetValue(_vrm10Instance);
                if (runtime != null)
                {
                    var expressionProp = runtime.GetType().GetProperty("Expression");
                    if (expressionProp != null)
                    {
                        var expressionManager = expressionProp.GetValue(runtime);
                        if (expressionManager != null)
                        {
                            // SetWeight メソッドを呼び出す
                            // SetWeight(ExpressionKey key, float weight)
                            // 簡易化のために、キーを引数にとるメソッドを探索してInvokeする
                            var methods = expressionManager.GetType().GetMethods();
                            foreach (var method in methods)
                            {
                                if (method.Name == "SetWeight" && method.GetParameters().Length == 2)
                                {
                                    // 第一引数の ExpressionKey (VRM10) を作成するか、リフレクションでキーのインスタンスを作成
                                    // 一般的には "aa" などの名前で呼び出せるので、String型やEnum型などをターゲットとする
                                    // UniVRM1.0ではExpressionKeyは静的クラスやEnumから生成可能
                                    // もしくはExpressionManagerに ExpressionMap などのDictionaryがあればそこからKeyを取得
                                    // 最も簡単なのは文字列/キーの自動生成
                                    // ここではUniVRMがロードされている状況でリフレクションにより「aa」に対応するExpressionKeyを自動選択
                                    var keyParamType = method.GetParameters()[0].ParameterType;
                                    var createMethod = keyParamType.GetMethod("CreateOVRM0X", new[] { typeof(int) }) 
                                                       ?? keyParamType.GetMethod("CreatePreset", new[] { typeof(int) });
                                    
                                    if (createMethod != null)
                                    {
                                        // Preset.aa = 0 / 1 などのインデックスを指定して生成
                                        var presetKey = createMethod.Invoke(null, new object[] { 0 }); // 0は「Aa/aa」を表すpreset
                                        method.Invoke(expressionManager, new object[] { presetKey, value });
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[LipsyncController] VRM10 SetWeight failed: {e.Message}");
            _vrm10Instance = null; // エラー多発時は無効化
        }
    }
}
