"""キャラクター人格のQLoRA化

性格・口調・世界観を「毎回プロンプトで説明する」のではなく、
小さなアダプタ（LoRA）としてモデルの重みに焼き込む。

  dataset.py  … キャラ設定・過去の会話・今のLLMの応答から学習データを作る
  trainer.py  … 4bit量子化したベースモデルにLoRAを学習する（別プロセスで動く）
  export.py   … 学習結果を Ollama / llama.cpp で使える形に書き出す
  manager.py  … 上記をダッシュボードから操作するための司令塔

torch / transformers / peft / bitsandbytes は学習するときだけ必要。
配信サーバ本体はこれらが無くても起動する（import は学習プロセス側でだけ行う）。
"""