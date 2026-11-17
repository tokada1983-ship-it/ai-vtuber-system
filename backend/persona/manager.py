"""PersonaManager: 人格QLoRAの一連の作業（データ作成→学習→登録→切替）を管理する

学習は重い（数十分〜数時間、VRAMを使い切る）ので配信サーバとは別プロセスで走らせ、
標準出力の進捗行だけを拾ってダッシュボードに流す。同時に走らせるジョブは1つだけ。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

from backend.config import AppConfig, save_config
from backend.logger import get_logger
from backend.persona import export
from backend.persona.dataset import PersonaDatasetBuilder
from backend.persona.trainer import PROGRESS_PREFIX

logger = get_logger(__name__)

# 学習ジョブに渡す設定（PersonaConfig の同名フィールドが既定値になる）
_TRAIN_KEYS = ("base_model", "epochs", "learning_rate", "lora_r", "lora_alpha",
               "lora_dropout", "max_seq_len", "batch_size", "grad_accum")


class PersonaManager:

    def __init__(self, config: AppConfig, hub, orchestrator):
        self._cfg = config
        self._hub = hub
        self._orch = orchestrator
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._env: dict | None = None
        self._job: dict = {"kind": "", "status": "idle"}
        self._log: deque[str] = deque(maxlen=80)
        self._last_broadcast = 0.0

    # ── パス ──────────────────────────────────────────────────

    @property
    def character(self) -> str:
        return self._cfg.character.name

    def _root(self, character: str | None = None) -> Path:
        return Path(self._cfg.persona.output_dir) / (character or self.character)

    def _dataset_dir(self) -> Path:
        return self._root() / "dataset"

    def _run_dir(self, run: str) -> Path:
        path = (self._root() / run).resolve()
        if self._root().resolve() not in path.parents:
            raise ValueError("不正なrun名です")
        return path

    # ── 状態 ──────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        return self._job.get("status") == "running"

    def dataset_meta(self) -> dict | None:
        meta = self._dataset_dir() / "meta.json"
        if not meta.exists():
            return None
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_runs(self) -> list[dict]:
        root = self._root()
        if not root.exists():
            return []
        runs = []
        for d in sorted(root.iterdir(), reverse=True):
            manifest = d / "manifest.json"
            if not (d.is_dir() and manifest.exists()):
                continue
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append({
                "run": d.name,
                "status": m.get("status", "?"),
                "base_model": m.get("base_model", ""),
                "final_loss": m.get("final_loss"),
                "eval_loss": m.get("eval_loss"),
                "examples": m.get("examples"),
                "quantization": m.get("quantization"),
                "has_adapter": (d / "adapter" / "adapter_config.json").exists(),
                "has_merged": (d / "merged").exists(),
                "runtime_model": m.get("runtime_model", ""),
                "samples": m.get("samples", [])[:3],
                "error": m.get("error", ""),
                "finished_at": m.get("finished_at"),
            })
        return runs

    def status(self) -> dict:
        p = self._cfg.persona
        return {
            "character": self.character,
            "config": p.model_dump(),
            "active": self._orch.persona_model_active,
            "llm_backend": self._orch._status.llm_backend,
            "job": {**self._job, "log": list(self._log)[-30:]},
            "dataset": self.dataset_meta(),
            "runs": self.list_runs(),
            "environment": self._env,
            "ollama": bool(export.find_ollama()),
        }

    async def _broadcast(self, force: bool = False) -> None:
        # 進捗は細かく来るので0.5秒に1回に間引く
        now = time.monotonic()
        if not force and now - self._last_broadcast < 0.5:
            return
        self._last_broadcast = now
        await self._hub.broadcast("dashboard", {"type": "persona_job",
                                                **self._job, "log": list(self._log)[-5:]})

    def _begin(self, kind: str, **extra) -> None:
        if self.busy:
            raise RuntimeError(f"別の処理（{self._job.get('kind')}）を実行中です")
        self._log.clear()
        self._job = {"kind": kind, "status": "running", "started_at": time.time(), **extra}

    async def _finish(self, status: str, **extra) -> None:
        self._job.update(status=status, finished_at=time.time(), **extra)
        await self._broadcast(force=True)

    # ── 環境チェック ───────────────────────────────────────────

    async def check_environment(self) -> dict:
        """学習プロセスと同じPythonで torch / peft / bitsandbytes を確認する"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "backend.persona.trainer", "--check",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        env: dict = {"ready": False}
        for line in out.decode("utf-8", errors="replace").splitlines():
            if line.startswith(PROGRESS_PREFIX):
                env = json.loads(line[len(PROGRESS_PREFIX):])
                env.pop("event", None)
        self._env = env
        return env

    # ── ① データセット作成 ─────────────────────────────────────

    async def build_dataset(self, synthetic: int | None = None) -> dict:
        self._begin("dataset", step=0, total_steps=0)
        await self._broadcast(force=True)
        p = self._cfg.persona
        n = p.synthetic_samples if synthetic is None else max(0, int(synthetic))
        history = (Path(self._cfg.character.data_dir) / "characters" / self.character
                   / "chat_history.jsonl") if p.include_chat_history else None
        builder = PersonaDatasetBuilder(self._cfg, chat_history_path=history)

        teacher = None
        try:
            examples = builder.from_samples() + builder.from_history()
            self._log.append(f"セリフ例・会話ログから {len(examples)} 件")
            rejected = 0
            if n > 0:
                teacher = self._build_teacher()
                if teacher is None:
                    raise RuntimeError("先生役のLLMを作れません（persona.teacher_backend を確認してください）")

                async def progress(done: int, total: int, speech: str) -> None:
                    self._job.update(step=done, total_steps=total)
                    if speech:
                        self._log.append(speech[:60])
                    await self._broadcast()

                synth, rejected = await builder.synthesize(teacher, n, progress=progress)
                self._log.append(f"合成 {len(synth)} 件（不採用 {rejected} 件）")
                examples += synth
            report = builder.write(examples, self._dataset_dir(), rejected=rejected)
            if report.train == 0:
                await self._finish("failed", error="教材が1件も作れませんでした（LLMが起動しているか確認してください）")
            else:
                await self._finish("done", report=report.to_dict())
            return report.to_dict()
        except Exception as e:
            logger.warning("persona dataset failed", error=str(e))
            await self._finish("failed", error=str(e))
            raise
        finally:
            if teacher is not None:
                await teacher.aclose()

    def _build_teacher(self):
        """
        会話例を作る先生役のLLM。人格モデルではない通常のLLMに、全文のキャラ設定で答えさせる。
        普段の会話用とは別のモデル（大きく賢いもの）を persona.teacher_model で指定できる。
        """
        from backend import factory
        p = self._cfg.persona
        name = p.teacher_backend or self._cfg.llm.primary
        cfg = self._cfg.llm.get_backend(name)
        if cfg is None:
            return None
        if p.teacher_model:
            cfg = cfg.model_copy(update={"model": p.teacher_model})
        self._log.append(f"先生役: {name} / {cfg.model or '(自動)'}")
        return factory.create_llm(name, cfg)

    # ── ② 学習 ────────────────────────────────────────────────

    async def start_training(self, overrides: dict | None = None) -> dict:
        meta = self.dataset_meta()
        if not meta or not meta.get("train"):
            raise RuntimeError("先にデータセットを作成してください")
        overrides = overrides or {}
        p = self._cfg.persona
        run = overrides.get("run") or time.strftime("%Y%m%d-%H%M%S")
        run_dir = self._run_dir(run)
        run_dir.mkdir(parents=True, exist_ok=True)

        job = {k: overrides.get(k, getattr(p, k)) for k in _TRAIN_KEYS}
        job.update(
            character=self.character,
            train_path=str(Path(meta["path"]).resolve()),
            eval_path=str(Path(meta.get("eval_path", "")).resolve()) if meta.get("eval_path") else "",
            output_dir=str(run_dir),
            quantization=overrides.get("quantization", "auto"),
            merge=bool(overrides.get("merge", True)),
            dataset=meta,
        )
        (run_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

        self._begin("train", run=run, step=0, total_steps=0, stage="starting")
        await self._broadcast(force=True)
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "backend.persona.trainer", "--job", str(run_dir / "job.json"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        self._task = asyncio.create_task(self._pump(run_dir), name="persona_train")
        logger.info("persona training started", run=run, base_model=job["base_model"])
        return {"run": run, "output_dir": str(run_dir)}

    async def _pump(self, run_dir: Path) -> None:
        """学習プロセスの出力を読み、進捗行をダッシュボードへ流す"""
        assert self._proc is not None and self._proc.stdout is not None
        log_file = open(run_dir / "train.log", "a", encoding="utf-8")
        error = ""
        try:
            async for raw in self._proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                log_file.write(line + "\n")
                if not line.startswith(PROGRESS_PREFIX):
                    # transformers の警告などは末尾だけ見せる
                    self._log.append(line[-200:])
                    continue
                try:
                    ev = json.loads(line[len(PROGRESS_PREFIX):])
                except json.JSONDecodeError:
                    continue
                kind = ev.pop("event", "")
                if kind == "progress":
                    self._job.update(ev)
                elif kind == "stage":
                    self._job["stage"] = ev.get("stage", "")
                    self._job.update({k: v for k, v in ev.items() if k != "stage"})
                    self._log.append(f"[{ev.get('stage')}]")
                elif kind == "eval":
                    self._job["eval_loss"] = ev.get("eval_loss")
                elif kind == "sample":
                    self._job.setdefault("samples", []).append(ev)
                    self._log.append(f"試し: {ev.get('output', '')[:60]}")
                elif kind == "error":
                    error = ev.get("error", "")
                await self._broadcast(force=kind != "progress")
            code = await self._proc.wait()
        finally:
            log_file.close()
            self._proc = None
        if self._job.get("status") == "cancelled":
            await self._broadcast(force=True)
        elif code == 0:
            await self._finish("done")
            await self._hub.send_log("info", f"人格QLoRAの学習が完了しました（{run_dir.name}）")
        else:
            await self._finish("failed", error=error or f"exit code {code}")
            await self._hub.send_log("error", f"人格QLoRAの学習に失敗: {error or code}")

    async def cancel(self) -> bool:
        if self._proc is None:
            return False
        self._job["status"] = "cancelled"
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=15)
        except asyncio.TimeoutError:
            self._proc.kill()
        return True

    # ── ③ Ollamaへ登録 ────────────────────────────────────────

    async def register(self, run: str, model_name: str = "", mode: str = "gguf",
                       ollama_base: str = "", quantize: str = "") -> dict:
        run_dir = self._run_dir(run)
        model_name = model_name or export.runtime_model_name(self.character, run)
        self._begin("register", run=run, model=model_name)
        await self._broadcast(force=True)
        try:
            outtype = quantize if mode == "gguf" and quantize else self._cfg.persona.gguf_outtype

            async def log(line: str) -> None:
                self._log.append(line[-160:])
                await self._broadcast()

            # ollama create はサーバが動いていないと失敗するので先に起動する
            from backend.llm import ollama_service
            be = self._cfg.llm.get_backend("ollama")
            if not await ollama_service.get_service(be.base_url if be else "").ensure_running():
                raise RuntimeError(ollama_service.get_service(be.base_url if be else "").last_error)

            if mode == "gguf":
                self._log.append(f"GGUFへ変換中（{outtype}）…")
                await self._broadcast(force=True)
                ok, tail = await export.convert_to_gguf(
                    run_dir, self._cfg.persona.llama_cpp_dir, outtype, log=log)
                if not ok:
                    raise RuntimeError(tail[-300:] or "GGUFへの変換に失敗しました")
            persona_be = self._cfg.llm.get_backend(self._cfg.persona.backend)
            export.write_modelfile(run_dir, mode=mode, ollama_base=ollama_base,
                                   temperature=persona_be.temperature if persona_be else 0.8,
                                   gguf_outtype=outtype)

            ok, tail = await export.ollama_create(run_dir, model_name, quantize=quantize,
                                                  mode=mode, log=log)
            if ok:
                # 登録したらすぐVRAMへ読み込んでおく（④で切り替えた直後から速く応答できる）
                await ollama_service.get_service(be.base_url if be else "").ensure_model(model_name)
        except Exception as e:
            await self._finish("failed", error=str(e))
            raise
        if not ok:
            await self._finish("failed", error=tail[-300:])
            raise RuntimeError(tail[-300:] or "ollama create に失敗しました")
        self._update_manifest(run_dir, runtime_model=model_name)
        await self._finish("done", model=model_name)
        return {"run": run, "runtime_model": model_name}

    @staticmethod
    def _update_manifest(run_dir: Path, **values) -> None:
        path = run_dir / "manifest.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data.update(values)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── ④ 切り替え ────────────────────────────────────────────

    async def activate(self, run: str = "", runtime_model: str = "", backend: str = "",
                       prompt_mode: str = "") -> dict:
        """学習済みモデルを配信に使う（LLMを作り直す。再起動は不要）"""
        p = self._cfg.persona
        if run and not runtime_model:
            runtime_model = next((r["runtime_model"] for r in self.list_runs()
                                  if r["run"] == run), "")
        if not runtime_model:
            raise ValueError("モデル名がありません。先にOllamaへ登録するか、モデル名を入力してください")
        backend = backend or p.backend
        if backend not in self._cfg.llm.backends:
            raise ValueError(f"LLMバックエンド「{backend}」が config.yaml にありません")
        p.enabled = True
        p.adapter = run or p.adapter
        p.runtime_model = runtime_model
        p.backend = backend
        if prompt_mode in ("slim", "full"):
            p.prompt_mode = prompt_mode
        save_config(self._cfg)
        name = await self._orch.rebuild_llm()
        await self._hub.send_log("info", f"人格モデルに切替: {runtime_model}（{backend}）")
        return {"ok": True, "llm_backend": name, "runtime_model": runtime_model}

    async def deactivate(self) -> dict:
        self._cfg.persona.enabled = False
        save_config(self._cfg)
        name = await self._orch.rebuild_llm()
        await self._hub.send_log("info", "人格モデルを解除しました（通常のLLMに戻しました）")
        return {"ok": True, "llm_backend": name}

    def update_settings(self, values: dict) -> dict:
        """学習の既定値などを保存する（未知のキーは無視）"""
        p = self._cfg.persona
        fields = type(p).model_fields
        for key, value in values.items():
            if key not in fields or key == "enabled":
                continue   # 有効化は activate() 経由（LLMの作り直しが要るため）
            current = getattr(p, key)
            if isinstance(current, bool):
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
            else:
                value = type(current)(value)
            setattr(p, key, value)
        save_config(self._cfg)
        return p.model_dump()

    async def shutdown(self) -> None:
        await self.cancel()