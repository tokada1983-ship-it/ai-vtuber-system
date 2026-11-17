"""人格QLoRAの学習スクリプト（配信サーバとは別プロセスで動く）

    py -3.10 -m backend.persona.trainer --job models/persona/<キャラ>/<run>/job.json
    py -3.10 -m backend.persona.trainer --check      # 学習環境の確認だけ

進捗は標準出力に「@@PERSONA {json}」の1行で流す。ダッシュボード（PersonaManager）は
この行だけを拾って進捗バーを動かす。それ以外の行はそのままログとして残る。

学習の中身:
  - ベースモデルを4bit(NF4)で読み込み、全Linear層にLoRAを挿す（QLoRA）
  - 損失はアシスタントの応答部分だけで計算する（プロンプトは覚えさせない）
  - bitsandbytes が使えない環境では自動で通常のLoRA（bf16）に切り替える
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROGRESS_PREFIX = "@@PERSONA "


def emit(event: str, **data) -> None:
    print(PROGRESS_PREFIX + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


# ── 環境チェック ─────────────────────────────────────────────

def check_environment() -> dict:
    info: dict = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info["torch"] = None
        info["error"] = f"torch: {e}"
    for mod in ("transformers", "peft", "accelerate"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    info["bitsandbytes"] = _bnb_version()
    info["bnb_4bit_ok"] = bnb_usable() if info.get("cuda") else False
    info["ready"] = bool(info.get("cuda") and info.get("transformers") and info.get("peft"))
    return info


def _bnb_version() -> str | None:
    try:
        import importlib.metadata as md
        return md.version("bitsandbytes")
    except Exception:
        return None


def bnb_usable() -> bool:
    """4bit量子化が実際にGPUで動くか（入っていても未対応GPUだと実行時に落ちる）"""
    if not _bnb_version():
        return False
    try:
        import io
        import contextlib
        import torch
        with contextlib.redirect_stdout(io.StringIO()):
            import bitsandbytes as bnb
            layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16, quant_type="nf4").cuda()
            layer(torch.randn(1, 64, device="cuda", dtype=torch.float16))
        return True
    except Exception:
        return False


def _dtype_kwargs(dtype) -> dict:
    """transformers 5 で torch_dtype が dtype に改名されたので両対応する"""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


# ── データ ──────────────────────────────────────────────────

def load_rows(path: str | Path) -> list[list[dict]]:
    rows = []
    p = Path(path) if path else None
    if p is None or not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            msgs = json.loads(line)["messages"]
        except (json.JSONDecodeError, KeyError):
            continue
        if msgs and msgs[-1].get("role") == "assistant":
            rows.append(msgs)
    return rows


def encode(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """
    プロンプト部分のラベルを -100 にして、応答部分だけを学習対象にする。
    チャットテンプレートはベースモデル付属のもの（実行時にOllama等が使うものと同じ）。
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if full.startswith(prompt):
        completion = full[len(prompt):]
    else:
        completion = messages[-1]["content"] + (tokenizer.eos_token or "")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids}


def collate(batch: list[dict], pad_id: int, device):
    import torch
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device),
            "attention_mask": mask.to(device)}


# ── 学習 ────────────────────────────────────────────────────

def train(job: dict) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    out_dir = Path(job["output_dir"])
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(job.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません（学習にはNVIDIA GPUが必要です）")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quant = job.get("quantization", "auto")
    use_4bit = quant == "4bit" or (quant == "auto" and bnb_usable())
    if quant == "4bit" and not bnb_usable():
        raise RuntimeError("bitsandbytes の4bitがこのGPUで動きません。"
                           "pip install -U bitsandbytes（0.46以降）を試すか quantization を none にしてください")
    emit("stage", stage="load_model", base_model=job["base_model"],
         quantization="4bit" if use_4bit else "none")

    tokenizer = AutoTokenizer.from_pretrained(job["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {**_dtype_kwargs(dtype), "device_map": {"": 0}}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(job["base_model"], **load_kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=int(job.get("lora_r", 16)),
        lora_alpha=int(job.get("lora_alpha", 32)),
        lora_dropout=float(job.get("lora_dropout", 0.05)),
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    ))
    # LoRAの重みはfp32で持つ（bf16のままだと小さな更新が丸められて学習が進まない）
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # データ
    max_len = int(job.get("max_seq_len", 1024))
    train_rows = [e for e in (encode(tokenizer, m, max_len) for m in load_rows(job["train_path"])) if e]
    eval_rows = [e for e in (encode(tokenizer, m, max_len) for m in load_rows(job.get("eval_path", ""))) if e]
    if not train_rows:
        raise RuntimeError("学習データが空です（max_seq_len を超える例は除外されます）")

    bs = max(1, int(job.get("batch_size", 1)))
    accum = max(1, int(job.get("grad_accum", 8)))
    epochs = float(job.get("epochs", 3))
    steps_per_epoch = max(1, math.ceil(len(train_rows) / (bs * accum)))
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(job.get("learning_rate", 2e-4)), weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)
    emit("stage", stage="train", examples=len(train_rows), eval_examples=len(eval_rows),
         total_steps=total_steps, trainable_params=trainable, total_params=total_params)

    def evaluate() -> float | None:
        if not eval_rows:
            return None
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            for i in range(0, len(eval_rows), bs):
                losses.append(model(**collate(eval_rows[i:i + bs], tokenizer.pad_token_id, device)).loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    step, micro, started = 0, 0, time.time()
    last_loss, eval_loss = None, None
    running: list[float] = []
    while step < total_steps:
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for i in range(0, len(order), bs):
            batch = collate([train_rows[j] for j in order[i:i + bs]], tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=dtype):
                loss = model(**batch).loss / accum
            loss.backward()
            running.append(loss.item() * accum)
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_loss = sum(running) / len(running)
            running.clear()
            elapsed = time.time() - started
            emit("progress", step=step, total_steps=total_steps, loss=round(last_loss, 4),
                 lr=scheduler.get_last_lr()[0], epoch=round(step / steps_per_epoch, 2),
                 eta_sec=int(elapsed / step * (total_steps - step)))
            if step % steps_per_epoch == 0 or step >= total_steps:
                eval_loss = evaluate()
                if eval_loss is not None:
                    emit("eval", step=step, eval_loss=round(eval_loss, 4))
                # エポックごとに保存しておく（途中で止めても使えるように）
                model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
            if step >= total_steps:
                break

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    emit("stage", stage="saved", adapter_dir=str(adapter_dir))

    samples = sample_outputs(model, tokenizer, job, device)
    result = {
        "final_loss": round(last_loss, 4) if last_loss is not None else None,
        "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
        "steps": step, "examples": len(train_rows),
        "quantization": "4bit" if use_4bit else "none",
        "trainable_params": trainable, "train_seconds": int(time.time() - started),
        "samples": samples,
    }
    model = optimizer = None
    torch.cuda.empty_cache()
    return result


def sample_outputs(model, tokenizer, job: dict, device, n: int = 3) -> list[dict]:
    """学習直後に数件しゃべらせて、口調が乗ったかを目で確かめられるようにする"""
    import torch
    rows = load_rows(job.get("eval_path", "")) or load_rows(job["train_path"])
    out = []
    model.eval()
    model.config.use_cache = True
    for msgs in rows[:n]:
        prompt = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        user = next((m["content"] for m in reversed(msgs[:-1]) if m["role"] == "user"), "(独り言)")
        out.append({"input": user[-120:], "output": text.strip()[:200],
                    "reference": msgs[-1]["content"][:200]})
        emit("sample", **out[-1])
    return out


def merge(job: dict) -> str:
    """
    アダプタをベースモデルに統合したフル重み（safetensors）を書き出す。
    Ollama の FROM に直接指定でき、llama.cpp で GGUF にも変換できる。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(job["output_dir"])
    merged_dir = out_dir / "merged"
    emit("stage", stage="merge", merged_dir=str(merged_dir))
    base = AutoModelForCausalLM.from_pretrained(
        job["base_model"], **_dtype_kwargs(torch.float16), low_cpu_mem_usage=True, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(out_dir / "adapter")).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(str(out_dir / "adapter")).save_pretrained(merged_dir)
    return str(merged_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="キャラクター人格のQLoRA学習")
    ap.add_argument("--job", help="job.json のパス")
    ap.add_argument("--check", action="store_true", help="学習環境を確認して終了する")
    args = ap.parse_args(argv)

    if args.check:
        emit("environment", **check_environment())
        return 0
    if not args.job:
        ap.error("--job か --check を指定してください")

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["output_dir"])
    manifest_path = out_dir / "manifest.json"
    manifest = {**job, "status": "running", "started_at": time.time()}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = train(job)
        manifest.update(result)
        if job.get("merge"):
            manifest["merged_dir"] = merge(job)
        manifest.update(status="done", finished_at=time.time(),
                        adapter_dir=str(out_dir / "adapter"))
        emit("done", **{k: manifest.get(k) for k in
                        ("final_loss", "eval_loss", "steps", "adapter_dir", "merged_dir")})
        return 0
    except Exception as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
        emit("error", error=manifest["error"])
        return 1
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())