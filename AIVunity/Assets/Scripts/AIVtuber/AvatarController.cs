using System.Collections;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class AvatarController : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private Animator animator;
    [SerializeField] private NavMeshAgent agent;

    [Header("Animator Parameters")]
    [SerializeField] private string speedParam = "Speed";
    [SerializeField] private string sitTrigger = "Sit";
    [SerializeField] private string standTrigger = "Stand";

    [Header("Settings")]
    [SerializeField] private float reachDistance = 0.5f;

    private bool _isSitting = false;
    private Transform _targetObject = null;
    private Coroutine _interactionCoroutine = null;

    private void Awake()
    {
        if (agent == null) agent = GetComponent<NavMeshAgent>();
        if (animator == null) animator = GetComponent<Animator>();
    }

    private void Update()
    {
        // NavMeshAgentの現在の速度に基づいてアニメーションのSpeedパラメータを更新
        if (animator != null && agent != null && agent.isActiveAndEnabled)
        {
            float currentSpeed = agent.velocity.magnitude / agent.speed;
            animator.SetFloat(speedParam, currentSpeed);
        }
    }

    /// <summary>
    /// 指定された座標へ自動移動を開始する
    /// </summary>
    public void MoveTo(Vector3 destination)
    {
        if (_isSitting)
        {
            StandUp();
        }

        StopActiveInteraction();
        
        if (agent != null && agent.isActiveAndEnabled)
        {
            agent.isStopped = false;
            agent.SetDestination(destination);
            Debug.Log($"[AvatarController] Moving to coordinates: {destination}");
        }
    }

    /// <summary>
    /// 表情を変更する（Unityアバター側の表情切り替えトリガー）
    /// </summary>
    public void SetEmotion(string emotion)
    {
        if (animator == null) return;
        
        // 既存の感情レイヤーのウェイト変更やパラメータ変更を行う
        // 例: animator.SetTrigger("Emotion_" + emotion);
        // 今回は標準のAnimatorパラメータ（Int/Trigger等）で表現できるように、トリガーまたはブレンド値として処理します
        animator.SetTrigger("Emotion_" + emotion);
        Debug.Log($"[AvatarController] Emotion changed to: {emotion}");
    }

    /// <summary>
    /// 特定のモーションアニメーションを再生する
    /// </summary>
    public void PlayMotion(string motion)
    {
        if (animator == null) return;
        animator.SetTrigger("Motion_" + motion);
        Debug.Log($"[AvatarController] Motion triggered: {motion}");
    }

    /// <summary>
    /// アクションを解析・実行する
    /// </summary>
    public void ExecuteAction(string action, string targetName)
    {
        Debug.Log($"[AvatarController] Executing action: {action} on target: {targetName}");
        
        switch (action.ToLower())
        {
            case "walk_to":
            case "move_to":
                if (!string.IsNullOrEmpty(targetName))
                {
                    Transform target = FindTargetTransform(targetName);
                    if (target != null)
                    {
                        MoveTo(target.position);
                    }
                }
                break;

            case "sit":
                if (!string.IsNullOrEmpty(targetName))
                {
                    Transform chair = FindTargetTransform(targetName);
                    if (chair != null)
                    {
                        StopActiveInteraction();
                        _interactionCoroutine = StartCoroutine(GoToAndSitCoroutine(chair));
                    }
                }
                else
                {
                    // 近くの椅子を探す
                    Transform nearestChair = FindNearestWithTag("Chair");
                    if (nearestChair != null)
                    {
                        StopActiveInteraction();
                        _interactionCoroutine = StartCoroutine(GoToAndSitCoroutine(nearestChair));
                    }
                    else
                    {
                        // ターゲットなしでその場で即座に座る
                        SitDown();
                    }
                }
                break;

            case "stand":
                StandUp();
                break;
        }
    }

    private void StopActiveInteraction()
    {
        if (_interactionCoroutine != null)
        {
            StopCoroutine(_interactionCoroutine);
            _interactionCoroutine = null;
        }
    }

    private IEnumerator GoToAndSitCoroutine(Transform chair)
    {
        MoveTo(chair.position);

        // 目的地に到達するまで待機
        while (agent.isActiveAndEnabled && agent.pathPending)
        {
            yield return null;
        }

        while (agent.isActiveAndEnabled && agent.remainingDistance > reachDistance)
        {
            yield return null;
        }

        // 到着したら椅子の正面を向いて座る
        agent.isStopped = true;
        
        // 回転を合わせる
        float rotateTime = 0.5f;
        float elapsed = 0f;
        Quaternion startRot = transform.rotation;
        Quaternion targetRot = chair.rotation; // 椅子の向きに合わせる

        while (elapsed < rotateTime)
        {
            elapsed += Time.deltaTime;
            transform.rotation = Quaternion.Slerp(startRot, targetRot, elapsed / rotateTime);
            yield return null;
        }

        transform.position = chair.position; // 微調整して密着させる
        SitDown();
        _interactionCoroutine = null;
    }

    private void SitDown()
    {
        if (_isSitting) return;
        _isSitting = true;
        agent.enabled = false; // 座っている間は移動を止める
        if (animator != null)
        {
            animator.SetTrigger(sitTrigger);
        }
        Debug.Log("[AvatarController] Avatar sat down.");
    }

    private void StandUp()
    {
        if (!_isSitting) return;
        _isSitting = false;
        agent.enabled = true;
        if (animator != null)
        {
            animator.SetTrigger(standTrigger);
        }
        Debug.Log("[AvatarController] Avatar stood up.");
    }

    private Transform FindTargetTransform(string name)
    {
        GameObject go = GameObject.Find(name);
        if (go != null) return go.transform;

        // タグで探す
        try
        {
            GameObject[] tagged = GameObject.FindGameObjectsWithTag(name);
            if (tagged.Length > 0) return tagged[0].transform;
        }
        catch (System.Exception) { /* ignored */ }

        // 部分一致でヒエラルキーから探す
        foreach (GameObject obj in FindObjectsOfType<GameObject>())
        {
            if (obj.name.Contains(name))
            {
                return obj.transform;
            }
        }

        Debug.LogWarning($"[AvatarController] Target object '{name}' not found in scene.");
        return null;
    }

    private Transform FindNearestWithTag(string tag)
    {
        try
        {
            GameObject[] objects = GameObject.FindGameObjectsWithTag(tag);
            GameObject nearest = null;
            float minDist = float.MaxValue;
            Vector3 currentPos = transform.position;

            foreach (GameObject obj in objects)
            {
                float dist = Vector3.Distance(obj.transform.position, currentPos);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = obj;
                }
            }

            return nearest != null ? nearest.transform : null;
        }
        catch (System.Exception)
        {
            return null;
        }
    }
}
