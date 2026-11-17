"""学習済みアダプタを実行環境（Ollama）に載せる

3通りの載せ方がある:

  gguf    … 統合した重み（merged/）を llama.cpp の convert_hf_to_gguf.py で GGUF にし、
            それを FROM にして Ollama モデルを作る。どのアーキテクチャでも通るので既定。
            できた .gguf は LM Studio / llama.cpp でもそのまま使える。
  merged  … merged/（safetensors）を直接 Ollama に取り込む。Ollama 0.3x では
            対応アーキテクチャが限られ、Qwen2 は取り込めない（unsupported MLX architecture）。
  adapter … Ollama 上の既存モデル（FROM）に ADAPTER を重ねる。
            FROM が学習時と同じベースモデルである必要がある。
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)


def runtime_model_name(character: str, run: str) -> str:
    """Ollamaのモデル名（英小文字・数字・-._ のみ）を作る"""
    slug = re.sub(r"[^a-z0-9]+", "-", character.lower()).strip("-")
    if not slug:
        # 日本語名は短いハッシュにする（同じキャラなら毎回同じ名前になる）
        slug = "char-" + hashlib.md5(character.encode("utf-8")).hexdigest()[:6]
    run_slug = re.sub(r"[^a-z0-9]+", "-", run.lower()).strip("-")
    return f"persona-{slug}-{run_slug}"[:60]


def find_ollama() -> str | None:
    from backend.llm.ollama_service import find_ollama as _find
    return _find()


GGUF_OUTTYPES = ("q8_0", "f16", "bf16")


def gguf_path(run_dir: Path, outtype: str = "q8_0") -> Path:
    return run_dir / f"model-{outtype}.gguf"


async def convert_to_gguf(run_dir: Path, llama_cpp_dir: str | Path, outtype: str = "q8_0",
                          log=None) -> tuple[bool, str]:
    """merged/ を GGUF に変換する。戻り値は (成功したか, 出力の末尾)"""
    import sys
    if outtype not in GGUF_OUTTYPES:
        return False, f"未対応の形式: {outtype}（{' / '.join(GGUF_OUTTYPES)}）"
    llama_cpp_dir = Path(llama_cpp_dir).resolve()
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        return False, (f"{script} がありません。llama.cpp を取得してください"
                       "（docs/persona_qlora.md の「GGUF変換の準備」参照）")
    merged = run_dir / "merged"
    if not merged.exists():
        return False, "merged/ がありません。学習時に「統合モデルも書き出す」を有効にしてください"
    out = gguf_path(run_dir, outtype).resolve()
    return await _run([sys.executable, str(script), str(merged.resolve()), "--outtype", outtype,
                       "--outfile", str(out)], cwd=llama_cpp_dir, log=log)


async def _run(args: list[str], cwd: Path, log=None) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    tail: list[str] = []
    assert proc.stdout is not None
    async for raw in proc.stdout:
        # 進捗バーは \r で上書きされるので、最後の断片だけを1行として扱う
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n").split("\r")[-1]
        line = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", line).strip()
        if not line:
            continue
        tail = (tail + [line])[-20:]
        if log:
            r = log(line)
            if asyncio.iscoroutine(r):
                await r
    code = await proc.wait()
    return code == 0, "\n".join(tail)


def write_modelfile(run_dir: Path, mode: str = "gguf", ollama_base: str = "",
                    temperature: float = 0.8, num_ctx: int = 4096,
                    gguf_outtype: str = "q8_0") -> Path:
    """Modelfile を run_dir に書き出してパスを返す"""
    lines: list[str] = []
    if mode == "gguf":
        path = gguf_path(run_dir, gguf_outtype)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} がありません（先にGGUFへ変換してください）")
        lines.append(f"FROM ./{path.name}")
    elif mode == "merged":
        merged = run_dir / "merged"
        if not merged.exists():
            raise FileNotFoundError("merged/ がありません。学習時に「統合モデルも書き出す」を有効にしてください")
        lines.append("FROM ./merged")
    elif mode == "adapter":
        if not ollama_base:
            raise ValueError("adapter モードでは、Ollama上のベースモデル名（FROM）が必要です")
        if not (run_dir / "adapter").exists():
            raise FileNotFoundError("adapter/ がありません")
        lines += [f"FROM {ollama_base}", "ADAPTER ./adapter"]
    else:
        raise ValueError(f"unknown mode: {mode}")
    lines += [
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_ctx {num_ctx}",
    ]
    path = run_dir / "Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def ollama_create(run_dir: Path, model_name: str, quantize: str = "",
                        mode: str = "gguf", log=None) -> tuple[bool, str]:
    """
    ollama create を実行する。戻り値は (成功したか, 出力の末尾)。
    quantize は merged モードのときだけ使う（Ollama 0.3x の形式: int4 / int8 など）。
    """
    exe = find_ollama()
    if not exe:
        return False, "ollama が見つかりません（https://ollama.com からインストールしてください）"
    args = [exe, "create", model_name, "-f", "Modelfile"]
    if mode == "merged" and quantize:
        args += ["--quantize", quantize]
    logger.info("ollama create", model=model_name, run_dir=str(run_dir), mode=mode)
    return await _run(args, cwd=run_dir, log=log)
