using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class UnityWebSocketClient : MonoBehaviour
{
    [Header("Connection Settings")]
    [SerializeField] private string serverUrl = "ws://localhost:9000/ws/unity";
    [SerializeField] private bool autoConnect = true;

    [Header("Target Controllers")]
    [SerializeField] private AvatarController avatarController;
    [SerializeField] private LipsyncController lipsyncController;

    private ClientWebSocket _webSocket = null;
    private CancellationTokenSource _cts;
    private readonly Queue<string> _messageQueue = new Queue<string>();
    private readonly object _queueLock = new object();

    private void Start()
    {
        if (autoConnect)
        {
            ConnectToServer();
        }
    }

    public void ConnectToServer()
    {
        Disconnect();
        _cts = new CancellationTokenSource();
        _webSocket = new ClientWebSocket();
        Task.Run(() => ReceiveLoop(serverUrl, _cts.Token));
        Debug.Log($"[WebSocketClient] Connecting to {serverUrl}...");
    }

    public void Disconnect()
    {
        _cts?.Cancel();
        if (_webSocket != null)
        {
            try
            {
                _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
            }
            catch (Exception) { /* ignored */ }
            _webSocket.Dispose();
            _webSocket = null;
        }
        Debug.Log("[WebSocketClient] Disconnected.");
    }

    private async Task ReceiveLoop(string url, CancellationToken token)
    {
        try
        {
            await _webSocket.ConnectAsync(new Uri(url), token);
            Debug.Log("[WebSocketClient] Connected to python server successfully.");

            byte[] buffer = new byte[4096];
            while (_webSocket.State == WebSocketState.Open && !token.IsCancellationRequested)
            {
                var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", token);
                    break;
                }

                string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                lock (_queueLock)
                {
                    _messageQueue.Enqueue(message);
                }
            }
        }
        catch (Exception e)
        {
            if (!token.IsCancellationRequested)
            {
                Debug.LogError($"[WebSocketClient] Error: {e.Message}");
                // 3秒後に再接続を試みる
                await Task.Delay(3000, token);
                if (!token.IsCancellationRequested)
                {
                    Debug.Log("[WebSocketClient] Reconnecting...");
                    ConnectToServer();
                }
            }
        }
    }

    private void Update()
    {
        // メインスレッドでキューからメッセージを処理
        string message = null;
        lock (_queueLock)
        {
            if (_messageQueue.Count > 0)
            {
                message = _messageQueue.Dequeue();
            }
        }

        if (message != null)
        {
            ProcessMessage(message);
        }
    }

    private void ProcessMessage(string json)
    {
        try
        {
            // Unity標準のJsonUtilityで簡易パースするためにラッパークラスを使用
            WsMessage msg = JsonUtility.FromJson<WsMessage>(json);
            if (msg == null || string.IsNullOrEmpty(msg.type)) return;

            switch (msg.type)
            {
                case "speech_start":
                    if (avatarController != null) avatarController.SetEmotion(msg.emotion);
                    if (lipsyncController != null) lipsyncController.StartLipsync(msg.text);
                    break;

                case "speech_end":
                    if (avatarController != null) avatarController.SetEmotion("neutral");
                    if (lipsyncController != null) lipsyncController.StopLipsync();
                    break;

                case "motion":
                    if (avatarController != null) avatarController.PlayMotion(msg.motion);
                    break;

                case "move_to":
                    if (avatarController != null) avatarController.MoveTo(new Vector3(msg.x, msg.y, msg.z));
                    break;

                case "action":
                    if (avatarController != null) avatarController.ExecuteAction(msg.action, msg.target);
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[WebSocketClient] Parse error: {e.Message}. Raw message: {json}");
        }
    }

    private void OnDestroy()
    {
        Disconnect();
    }

    // JSONパース用のデータ構造
    [Serializable]
    private class WsMessage
    {
        public string type;
        public string text;
        public string emotion;
        public string motion;
        public string action;
        public string target;
        public float x;
        public float y;
        public float z;
    }
}
