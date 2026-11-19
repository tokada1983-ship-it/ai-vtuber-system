"""設定ローダー (config.yaml → Pydanticモデル)"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── サブ設定 ────────────────────────────────────────────

class YouTubeAPIConfig(BaseModel):
    api_key_env: str = "YOUTUBE_API_KEY"
    polling_interval_ms: int = 5000

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class TwitcastingConfig(BaseModel):
    access_token_env: str = "TWITCASTING_ACCESS_TOKEN"
    access_token: str = ""
    user_id: str = ""
    polling_interval_ms: int = 2000

    def get_token(self) -> str:
        import os
        return self.access_token or os.getenv(self.access_token_env, "")


class TwitchConfig(BaseModel):
    token_env: str = "TWITCH_OAUTH_TOKEN"
    token: str = ""
    nick: str = ""
    channel: str = ""

    def get_token(self) -> str:
        import os
        return self.token or os.getenv(self.token_env, "")


class YouTubeConfig(BaseModel):
    video_id: str = ""
    primary_source: str = "pytchat"
    youtube_api: YouTubeAPIConfig = Field(default_factory=YouTubeAPIConfig)


class LLMBackendConfig(BaseModel):
    base_url: str = ""
    model: str = ""
    temperature: float = 0.8
    max_tokens: int = 200
    api_key_env: str = ""   # 認証が要るOpenAI互換サーバ用（空ならバックエンド既定）
    extra_payload: dict = Field(default_factory=dict)
    # ↑ API固有の追加パラメータ（Gemini の思考オフ指定など）を、モデルが
    #   対応しない項目はAPIの文法に合わせてここに書く。既存バックエンド（LM Studio等）は空。


class HybridLLMConfig(BaseModel):
    """
    ハイブリッドLLM振り分け：待機系はローカル、即応が要る応答はクラウド。

    待機中（独り言・トピック・掛け合い）は反応速度もコストも気にしなくてよいので
    ローカルLLMで回し、視聴者コメントへの返答やマスターの呼びかけ（音声入力・
    ローカルチャット）のように「待たせたくない」応答だけクラウドAPIに寄せる。

    振り分けは Orchestrator が応答の種別（route）を見て行う:
      - route="idle"    → idle_backend（独り言ループ・掛け合い・記憶整理・内省）
      - route="respond" → fast_backend（コメント返答・ローカルチャット・音声入力への応答）
    route を渡さない呼び出し（旧コード経由）は fast 側に倒す。

    どちらかのバックエンド名が空なら、その種別は従来どおり agent.llm を使う
    （＝ハイブリッド無効時は何も変わらない）。
    クラウド側が不調（is_healthy=False）なら自動でローカルに戻る。
    """
    enabled: bool = False
    idle_backend: str = ""    # llm.backends のキー（例: lmstudio）
    fast_backend: str = ""    # llm.backends のキー（例: gemini）


class LLMConfig(BaseModel):
    primary: str = "lmstudio"
    fallback: str = "ollama"
    # 出力プロトコル: "json"（従来のJSON方式）| "tag"（タグ方式・Phase 5）。
    # プロンプトのフォーマット節・パーサ・計測が一体で切り替わる。
    # "json" に戻せば従来動作に完全復帰する（ロールバック経路）
    protocol: str = "json"
    # num_ctx 適応制御（Ollama のみ。2026-11-19 問題2対策の恒久化）:
    # プロンプトが num_ctx に達して生成枠が潰れる事故を受け、固定値では
    # なく生成ごとの prompt_eval_count を見て段（context_tiers）を自動で
    # 上げ下げする。上げる条件は「現段の context_raise_ratio 超」または
    # done_reason=length、下げる条件は「現段の context_lower_ratio 未満が
    # 持続（context_lower_turns 回連続 or context_lower_window_sec 秒）」。
    # 段の適用はターン境界のみで、適用後は context_cooldown_sec の間は
    # 再変更しない。adaptive_ctx=False で従来動作（各バックエンドの
    # extra_payload に書かれた num_ctx 固定）に戻る。
    adaptive_ctx: bool = True
    context_tiers: list[int] = Field(
        default_factory=lambda: [4096, 6144, 8192, 12288, 16384])
    context_initial: int = 8192
    context_raise_ratio: float = 0.85
    context_lower_ratio: float = 0.45
    context_cooldown_sec: float = 300.0
    context_lower_turns: int = 50
    context_lower_window_sec: float = 1800.0
    backends: dict[str, LLMBackendConfig] = Field(default_factory=dict)
    hybrid: HybridLLMConfig = Field(default_factory=HybridLLMConfig)

    def get_backend(self, name: str) -> LLMBackendConfig | None:
        return self.backends.get(name)


class OllamaServiceConfig(BaseModel):
    """Ollama の自動起動・モデルの事前ロード"""
    auto_start: bool = True          # 起動していなければ ollama serve を立ち上げる
    preload_on_start: bool = True    # サーバ起動時に使うモデルをVRAMへ読み込んでおく
    executable: str = ""             # ollama.exe のパス（空なら自動検出）
    keep_alive: str = "30m"          # 最後の応答からモデルをVRAMに残しておく時間（-1で無期限）
    auto_pull: bool = False          # 未取得のモデルを自動でダウンロードする（数GB）
    startup_timeout_sec: float = 30.0
    stop_on_exit: bool = True        # 自分で起動した ollama serve は終了時に止める


class TTSBackendConfig(BaseModel):
    base_url: str = "http://localhost:50021"
    speaker_id: int = 3
    speed: float = 1.1
    pitch: float = 0.0
    # IrodoriTTS 拡張フィールド（VOICEVOXでは無視される）
    model: str = "irodori-tts"
    voice: str = "default"
    use_emotion_voice: bool = False
    emotion_voice_prefix: str = ""
    response_format: str = "wav"
    api_key: str = "not-used" 


class IrodoriServerConfig(BaseModel):
    """Irodori-TTS サーバの自動起動（モデル・声などはサーバ側の .env で設定）"""
    auto_start: bool = True                       # 応答しなければ起動する
    dir: str = "Irodori-TTS/Irodori-TTS-Server"  # Irodori-TTS-Server のフォルダ
    command: str = ""                             # 空なら .venv の python -m irodori_openai_tts
    startup_timeout_sec: float = 300.0            # 初回はモデルのダウンロード・ロードで時間がかかる
    stop_on_exit: bool = True                     # 自分で起動したサーバは終了時に止める
    # サーバに渡す環境変数（例: HF_HUB_CACHE でモデルの保存先を空きのあるドライブにする）
    env: dict[str, str] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    primary: str = "voicevox"
    # 1文の合成にこれ以上かかったら諦めて字幕のみで進む。
    # ローカルTTSサーバがハングしても配信全体が止まらないようにするための上限。
    timeout_sec: float = 30.0
    backends: dict[str, TTSBackendConfig] = Field(default_factory=dict)
    irodori_server: IrodoriServerConfig = Field(default_factory=IrodoriServerConfig)


class CharacterVoiceConfig(BaseModel):
    """キャラクターごとの声（空・None の項目は tts.backends の値をそのまま使う）"""
    backend: str = ""                # tts.backends のキー（空なら tts.primary）
    voice: str = ""                  # Irodori-TTS などの話者名（voices/<名前>.wav）
    speaker_id: Optional[int] = None  # VOICEVOX の話者ID
    speed: Optional[float] = None
    pitch: Optional[float] = None


class CharacterLLMConfig(BaseModel):
    """キャラクターごとのLLMの差し替え（空なら llm.primary / fallback をそのまま使う）"""
    backend: str = ""                # llm.backends のキー
    model: str = ""                  # そのバックエンドでのモデル名（キャラ専用モデルなど）


class EffectEntryConfig(BaseModel):
    """エフェクトカタログの1エントリ"""
    file: str = ""    # cutin / se 用のファイル（作業フォルダ相対。cutin は models/ 以下に置くと /models で配信される）
    text: str = ""    # flip 用の定型文
    label: str = ""   # プロンプト・画面に出す短い説明


class SystemEffectsConfig(BaseModel):
    """システム発火エフェクト（無音・材料なし時間帯の自動演出）の設定

    モデルの自発発火とは別経路だが、全体クールダウン（EffectsConfig.
    cooldown_sec）は自発発火と共有し、連発しない。さらにシステム発火
    どうしの間隔（cooldown_sec）も空ける。
    """
    enabled: bool = True
    cooldown_sec: float = 300.0          # システム発火どうしの最小間隔
    silence_threshold_sec: float = 180.0  # 無音がこの秒数を超えたら発火を試みる
    cutin_probability: float = 0.15      # カットインを混ぜる確率（残りはSE/フリップ）


class EffectsConfig(BaseModel):
    """演出エフェクト（カットイン動画・効果音・フリップ表示）の設定"""
    enabled: bool = True
    cooldown_sec: float = 90.0      # 自発エフェクト全体の最小間隔（手動発火は無視）
    flip_duration_sec: float = 5.0  # フリップの表示時間
    flip_max_chars: int = 30        # フリップ自由文の上限文字数
    cutin: dict[str, EffectEntryConfig] = Field(default_factory=dict)
    se: dict[str, EffectEntryConfig] = Field(default_factory=dict)
    flip: dict[str, EffectEntryConfig] = Field(default_factory=dict)
    system: SystemEffectsConfig = Field(default_factory=SystemEffectsConfig)


class CharacterEffectsConfig(BaseModel):
    """キャラクターごとのエフェクト上書き（None・空の項目は全体設定を使う）"""
    enabled: Optional[bool] = None
    cooldown_sec: Optional[float] = None
    flip_duration_sec: Optional[float] = None
    flip_max_chars: Optional[int] = None
    cutin: dict[str, EffectEntryConfig] = Field(default_factory=dict)
    se: dict[str, EffectEntryConfig] = Field(default_factory=dict)
    flip: dict[str, EffectEntryConfig] = Field(default_factory=dict)


class ThoughtsConfig(BaseModel):
    """内心（inner thoughts）の設定。声には出さず、画面の小さい吹き出しにだけ出す"""
    enabled: bool = True
    max_chars: int = 30          # 内心の上限文字数（検証で切り詰める）
    cooldown_sec: float = 20.0   # 内心表示の最小間隔（thought専用。effectとは別時計）


class CharacterThoughtsConfig(BaseModel):
    """キャラクターごとの内心設定の上書き（Noneの項目は全体設定を使う）"""
    enabled: Optional[bool] = None
    max_chars: Optional[int] = None
    cooldown_sec: Optional[float] = None


class ElythMemoryConfig(BaseModel):
    """ELYTH会話エピソードの記憶取り込み（PC側）の設定。

    VM側 cron が Google Drive `Muse連携/elyth-episodes/` へ置く日次
    エピソード（ELYTH-EP-*.md）を、アプリ起動時＋定期的に走査して
    character（既定 Character1）の MemoryStore へだけ取り込む。
    Character2は ELYTH に参加していないため対象外（個体分離）。
    episodes_dir が空、または enabled=false で取り込みは止まる。
    """
    enabled: bool = True
    episodes_dir: str = ""            # Drive ローカルコピーの elyth-episodes フォルダ
    character: str = "Character1"         # 取り込み先キャラの id
    scan_interval_sec: float = 3600.0  # 定期走査の間隔（起動時にも1回走る）


class CharacterConfig(BaseModel):
    """
    出演キャラクター1体ぶんの設定。

    掛け合い配信では config.yaml の characters に複数並べる。
    声・アバター・LLM はキャラごとに上書きでき、書かなければ全体設定を使う。
    """
    name: str = "Character2"
    # アバター画面の ?char= や WebSocket のメッセージで使う識別子（空なら name）
    id: str = ""
    enabled: bool = True
    # 空なら <data_dir>/characters/<name>/character.yaml
    prompt_file: str = "backend/llm/prompts/character.yaml"
    stream_theme: str = ""
    # キャラクターごとの記憶・会話ログ・DBを置く場所（characters/<名前>/ の親）
    data_dir: str = "backend/llm/prompts"
    # コメントで呼ばれたと判定する別名（例: Character1ちゃん, character1）。name は自動で含まれる
    aliases: list[str] = Field(default_factory=list)
    voice: CharacterVoiceConfig = Field(default_factory=CharacterVoiceConfig)
    llm: CharacterLLMConfig = Field(default_factory=CharacterLLMConfig)
    # キャラ専用のアバター（None なら全体の avatar 設定）
    avatar: Optional["AvatarConfig"] = None
    # 生成が反復ブロック等で詰まったときにそのまま喋る定型の繋ぎ台詞。
    # 話題に依存しない汎用文だけを置く（事実の主張・数値は入れない）
    fallback_lines: list[str] = Field(default_factory=list)
    effects: CharacterEffectsConfig = Field(default_factory=CharacterEffectsConfig)
    thoughts: CharacterThoughtsConfig = Field(default_factory=CharacterThoughtsConfig)

    @property
    def key(self) -> str:
        return (self.id or self.name).strip()

    def resolved_prompt_file(self) -> Path:
        if self.prompt_file:
            return Path(self.prompt_file)
        return Path(self.data_dir) / "characters" / self.name / "character.yaml"

    def call_names(self) -> list[str]:
        """コメント中にあれば「このキャラへの呼びかけ」とみなす名前"""
        names = [self.name, *self.aliases]
        return [n for n in dict.fromkeys(x.strip() for x in names) if n]


class EnsembleConfig(BaseModel):
    """
    複数キャラクターの掛け合いの設定（characters が2体以上のときだけ効く）

    コメントへの返答は1体が担当し、そのあと相方が一言かぶせることがある。
    コメントが無いときは交互に話題を振り、相方が受けて会話を続ける。
    """
    # 誰がコメントに答えるか。balance=呼ばれた方→しばらく喋っていない方 / lead=基本は先頭のキャラ
    responder: str = "balance"
    banter_probability: float = 0.5        # コメント返答のあと相方が反応する確率
    idle_banter_probability: float = 0.85  # 独り言・話題振りのあと相方が反応する確率
    max_banter_turns: int = 2              # 1つの話題でキャラ同士が続けて話す最大回数
    turn_gap_sec: float = 0.4              # 話者が交代するときの間
    yield_to_comments: bool = True         # コメントが待っていたら掛け合いを切り上げる


class PersonaConfig(BaseModel):
    """
    キャラクター人格のQLoRA化（性格・口調をモデルの重みに焼き込む）の設定。

    学習済みアダプタを有効にすると、そのキャラ専用に作ったモデル
    （Ollama等に登録した名前）で応答し、システムプロンプトからは
    長いキャラ設定を省いて軽量化する（prompt_mode: slim）。
    """
    enabled: bool = False
    adapter: str = ""                # 使うアダプタ名（models/persona/<キャラ>/<名前>）
    backend: str = "ollama"          # 学習済みモデルを載せるLLMバックエンド（llm.backendsのキー）
    runtime_model: str = ""          # そのバックエンド上でのモデル名（例: Character1-persona）
    prompt_mode: str = "slim"        # slim=キャラ設定を短縮 / full=従来どおり全文
    fallback_to_base: bool = True    # 専用モデルが落ちていたら通常のLLMに戻す
    # ── 学習の既定値（ダッシュボードから上書きできる）───────────
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "models/persona"
    epochs: float = 3.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 8
    # ── データセット ────────────────────────────────────────
    # ── Ollamaへの登録（学習結果をGGUFにして載せる）─────────────
    llama_cpp_dir: str = "tools/llama.cpp"   # convert_hf_to_gguf.py がある場所
    gguf_outtype: str = "q8_0"               # q8_0 / f16 / bf16（q8_0で品質はほぼ劣化なし）
    synthetic_samples: int = 200     # 今のLLMに作らせる会話例の数（0で生成しない）
    # 会話例を作る先生役（空なら普段の llm.primary とそのモデルを使う）
    teacher_backend: str = ""        # llm.backends のキー（例: lmstudio）
    teacher_model: str = ""          # そのバックエンドでのモデル名（例: qwen_qwen3.6-35b-a3b）
    include_chat_history: bool = True  # ローカルチャットの実会話も教材にする


class VRMConfig(BaseModel):
    model_path: str = "models/default.vrm"
    idle_motion: str = "breathe"


class WorldConfig(BaseModel):
    enabled: bool = False
    model_path: str = "models/studio_world.glb"
    spawn_point: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class Live2DConfig(BaseModel):
    """Live2D（Cubism）アバターの設定

    モデルは models/live2d/<名前>/ にフォルダごと置き、その中の *.model3.json を指す。
    model_path が空なら models/ 以下で最初に見つかったモデルを使う。
    """
    model_path: str = ""            # 例: models/live2d/hiyori/hiyori.model3.json
    scale: float = 0.9              # 画面の高さに対するモデルの高さの比率
    position_x: float = 0.5         # 画面幅に対する中心位置（0=左端 1=右端）
    position_y: float = 0.5         # 画面高さに対する中心位置（0=上端 1=下端）
    idle_motion_group: str = "Idle"  # 何もしていないときに流すモーショングループ
    lipsync_gain: float = 1.2       # 口の開き具合の倍率（大きいほど口が開く）
    mouth_param: str = ""           # 口のパラメータID（空なら自動判定）
    # AI側の感情/モーション名 → モデル側の表情名・モーショングループ名
    expression_map: dict[str, str] = Field(default_factory=dict)
    motion_map: dict[str, str] = Field(default_factory=dict)


class VideoAvatarConfig(BaseModel):
    """動画アバター（透過WebMなどのモーション動画を再生する描画方式）の設定

    モーション一式は models/ 以下に置き、カタログJSONで
    待機モーション・モーション対応表・感情対応表を定義する。

    カタログの形式:
      {
        "character": "Character1",
        "idle": ["idle_main.webm", "idle_02.webm", ...],
        "motion_map": {"nod": "custom_nod.webm", "dance": ["dance_01.webm", ...]},
        "emotion_map": {"happy": ["custom_tailwag.webm", ...]}
      }
    ファイル名はカタログのあるフォルダからの相対名。"/" 始まりや http(s) はそのまま使う。
    """
    catalog_path: str = ""          # 例: models/motions/Character1/catalog.json


class AvatarConfig(BaseModel):
    type: str = "vrm"               # vrm / live2d / video
    vrm: VRMConfig = Field(default_factory=VRMConfig)
    live2d: Live2DConfig = Field(default_factory=Live2DConfig)
    video: VideoAvatarConfig = Field(default_factory=VideoAvatarConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)


class AudioOutputConfig(BaseModel):
    device: Optional[str] = None


class OrchestratorConfig(BaseModel):
    idle_threshold_sec: int = 30
    response_history_turns: int = 20
    consolidation_trigger_turns: int = 50  # N ターンごとに長期記憶整理を発火
    comment_dedupe_window_sec: int = 60
    max_comment_length: int = 200
    max_queue_size: int = 50       # コメントキューの上限（超えた分は破棄）
    comment_ttl_sec: int = 120     # キュー内でこの秒数を超えたコメントは破棄
    # ストリーミング発話（1文目を喋りながら2文目を合成して待ち時間を減らす）
    streaming_tts: bool = True
    # 先行生成（P1-C）: 現在の発話の再生・待機中に、次の自律発話を
    # 先に生成しておく（生成の失敗・回復リトライも再生中に済ませる）
    prefetch_idle: bool = True
    stream_min_chunk_chars: int = 8    # これ未満の断片は次の文とまとめる
    stream_max_chunk_chars: int = 60   # 文末が来なくてもこの長さで区切る
    # エピソード記憶（自動で貯まる記憶）
    max_memory_entries: int = 300      # 保持する上限。超えたらスコアの低い順に忘れる
    memory_half_life_days: float = 30.0  # 参照されない記憶の重みが半分になる日数
    memory_top_k: int = 3              # 1回の応答で参照する記憶の件数
    memory_min_score: float = 0.25     # これ未満の関連度の記憶はプロンプトに入れない
    idle_core_memory_chars: int = 1200  # 独り言プロンプトに入れるコア記憶の上限
    idle_episodic_entries: int = 8      # 独り言プロンプトに混ぜるエピソード記憶の件数
    # 長期記憶・人格維持（A: リスナー関係性、B: 自己内省日記、C: 人格死守）
    enable_viewer_profiles: bool = True     # 視聴者ごとの親密度・関係性管理
    enable_reflection: bool = True          # 配信終了時・定期内省日記の生成
    max_diary_entries: int = 5              # コア記憶に保持する内省日記の件数
    enable_persona_validation: bool = True  # 発話前の一人称・口調の自動補正


class SafetyConfig(BaseModel):
    """応答後フィルタの設定"""
    enabled: bool = True
    ng_words: list[str] = Field(default_factory=list)


class CommentFilterConfig(BaseModel):
    """コメント選別（トリアージ）の設定。

    配信コメントのうち「応答するもの」だけをキューへ進める段の設定。
    記録（StreamArchiver）と視聴者プロフィールは選別結果に依らず更新される。
    enabled=false でこの段は素通りし、従来動作（全件キュー投入）に戻る。
    必ず拾う条件（スパチャ／キャラ名の呼びかけ／質問形／初コメ）と、
    落とす条件（NGワード／重複／連打／リアクションのみ）はコード側の
    ルールで、ここでは確率・閾値・NGワードだけを調整できるようにする。
    """
    enabled: bool = True
    ng_words: list[str] = Field(default_factory=list)  # 含むコメントは応答しない
    # 混雑度ごとの「通常コメントを拾う確率」（キュー滞留で下がる）
    base_pickup_probability: float = 1.0        # 通常時
    congested_pickup_probability: float = 0.5   # 滞留時
    saturated_pickup_probability: float = 0.2   # 飽和時
    congested_queue_size: int = 5    # キューがこの件数以上で「滞留」
    saturated_queue_size: int = 15   # キューがこの件数以上で「飽和」
    saturated_min_score: float = 3.0  # 飽和時、これ未満のスコアは確率判定なしで落とす
    duplicate_window_sec: int = 60   # 同一ユーザーの同一文を重複とみなす間隔（秒）
    spam_repeat_min_length: int = 6  # 連打判定を始める最小文字数
    spam_repeat_ratio: float = 0.8   # 最頻文字がこの割合以上なら連打とみなす


class ViewerPointsConfig(BaseModel):
    """視聴者ポイント（Phase 4 / 組み込み検討書 §5 第1段）の設定。

    視聴者ごとのポイント台帳と、累積ポイントによる親密度レベル移行。
    加算は基本点＋レート制限つきルールで、同一イベントの連打では
    稼げないようにする。enabled=false でポイント加算は止まり、
    レベルは従来の来訪回数ベースに戻る。
    """
    enabled: bool = True
    # 基本点
    visit_points: int = 10            # 来訪（1配信セッションにつき1回だけ）
    comment_points: int = 2           # 実質のあるコメント1件
    reaction_points: int = 1          # リアクションのみのコメント1件
    picked_points: int = 3            # コメントに応答してもらえたときの上乗せ
    superchat_base_points: int = 10   # スパチャ基本点
    superchat_per_1000_points: int = 1  # スパチャ1,000円ごとの上乗せ
    superchat_bonus_cap: int = 10     # スパチャ金額上乗せの上限
    attendance_milestone_every: int = 5    # 参加配信数がこの倍数に達するたび
    attendance_milestone_points: int = 20  # 継続参加ボーナス
    # レート制限（直近1時間の加算回数上限）
    comment_max_per_hour: int = 10
    reaction_max_per_hour: int = 5
    picked_max_per_hour: int = 20
    session_point_cap: int = 150      # 1セッションで稼げる上限（スパチャ分は除く）
    # レベル閾値（Lv1..Lv5 の開始ポイント）
    level_thresholds: list[int] = Field(default_factory=lambda: [0, 50, 200, 500, 1000])


class WebSearchConfig(BaseModel):
    """Web検索（知識の仕入れ）の設定"""
    provider: str = "auto"          # auto / google_cse / brave / duckduckgo / google_news
    api_key_env: str = "GOOGLE_SEARCH_API_KEY"
    cse_id_env: str = "GOOGLE_CSE_ID"
    brave_api_key_env: str = "BRAVE_SEARCH_API_KEY"
    max_results: int = 3
    timeout_sec: float = 8.0
    cache_ttl_sec: int = 900        # 同じクエリはこの秒数だけ使い回す（API節約）
    language: str = "ja"
    region: str = "JP"
    # レート制限対策（Phase 3）: 検索リクエスト全体の最小間隔と、
    # 429 を出したプロバイダへの指数バックオフ（クールダウン）
    min_request_interval_sec: float = 3.0   # プロバイダを問わず空ける間隔
    cooldown_base_sec: float = 60.0         # 429 の初回クールダウン
    cooldown_max_sec: float = 900.0         # クールダウンの上限


class NewsConfig(BaseModel):
    """ニュース取得（RSS）の設定"""
    enabled: bool = True
    feeds: list[str] = Field(default_factory=lambda: [
        "https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja",
    ])
    refresh_interval_sec: int = 900   # フィードを取り直す間隔
    max_items: int = 30
    max_age_hours: float = 24.0       # これより古い記事は話題にしない
    keywords: list[str] = Field(default_factory=list)  # 指定すると該当記事だけに絞る
    retry_base_sec: float = 120.0     # 取得失敗時の再試行の初回間隔（指数バックオフ、Phase 3）


class AutonomousTopicConfig(BaseModel):
    """自律トーク（コメントが無いときに自分から話題を振る）の設定"""
    enabled: bool = True
    # 話題の出どころの重み。memory=記憶ネタ / news=最新ニュース / curiosity=Web検索
    weights: dict[str, float] = Field(default_factory=lambda: {
        "memory": 3.0, "news": 2.0, "curiosity": 1.0,
    })
    min_web_interval_sec: int = 120   # Webネタはこの間隔以上あける（API節約・話題の偏り防止）
    max_documents: int = 3            # プロンプトに入れる記事数の上限
    remember: bool = True             # 話したネタをエピソード記憶に残す
    # Phase 1（組み込み検討書 §3-C1/C2）: プランナーの穴埋め
    memory_share_cap: float = 0.45    # 直近ウィンドウで memory 話題がこの占有率以上なら、他候補がある間は memory を選ばない
    memory_share_window: int = 20     # 占有率を測る直近の話題数
    filler_enabled: bool = False      # 材料が全て尽きたとき、定番小噺へフォールバックする（本番は config.yaml で有効化）
    # Phase 3（次発言プランニング / STUDY §6-(a1)）: ルール版プランナー
    planner_enabled: bool = True      # 次発言の意図決定と弧の完遂判定（無効で Phase 2 までの動作）
    arc_max_speeches: int = 4         # 1つの話題の弧で許す発話数の上限（導入を含む）
    arc_max_continuations: int = 2    # 弧の継続回数の上限（Phase 1 までの固定値2を設定化）
    arc_max_sec: float = 900.0        # 弧の開始からの上限時間（秒）。超えたら弧を解放する
    arc_ask_probability: float = 0.25 # 展開の発話で「問いかけ」の意図を選ぶ確率


class AnswerWithSearchConfig(BaseModel):
    """コメントで聞かれたことをその場で検索して答える設定"""
    enabled: bool = True
    trigger_words: list[str] = Field(default_factory=lambda: [
        "調べて", "検索して", "ググって", "最新", "ニュース", "って何", "とは", "教えて",
    ])
    max_results: int = 3
    min_interval_sec: int = 20        # 連続検索の下限間隔（コメント連投で叩きすぎない）
    # 事実質問の判定（Phase 3）: trigger_words に加えて「疑問形 × 事実カテゴリ」
    # でも拾う。トリガーワードを含まない事実質問（例:「昨日の日経平均株価の
    # 終値って、いくらだったか分かりますか？」）が無回答に落ちた実測への対処。
    fact_categories: list[str] = Field(default_factory=lambda: [
        "株価", "終値", "始値", "高値", "安値", "日経平均", "為替", "円相場",
        "金利", "天気", "気温", "降水確率", "価格", "値段", "発売日", "発売",
        "開催", "結果", "順位", "スコア", "得点", "営業時間", "住所",
        "電話番号", "人口", "視聴率", "売上", "時価総額", "配当", "送料", "在庫",
    ])
    question_words: list[str] = Field(default_factory=lambda: [
        "いくら", "何円", "何位", "何人", "何時", "何分", "何日", "何年",
        "何月", "何％", "何パーセント", "いつ", "どこ", "誰", "どれ",
    ])
    date_words: list[str] = Field(default_factory=lambda: [
        "昨日", "今日", "明日", "明後日", "今週", "先週", "来週",
        "今月", "先月", "今年", "去年", "現在",
    ])
    # 意見マーカーがあれば事実質問にしない（意見ルールと一致させる）
    opinion_markers: list[str] = Field(default_factory=lambda: [
        "どう思う", "感想", "好き", "派？", "派か", "おすすめ", "推し", "応援",
    ])
    # 保留回答キュー（Phase 3）: 同期検索が空振り／間隔ゲートで見送った
    # 質問を裏で調べ直し、答えを自律トークとして差し込む
    answer_retry_delays_sec: list[float] = Field(
        default_factory=lambda: [30.0, 120.0, 300.0])  # ワーカーの再試行間隔
    pending_ttl_sec: int = 600        # 保留の有効期限（過ぎたら破棄）


class KnowledgeConfig(BaseModel):
    """Web検索・ニュース・自律トークをまとめた設定"""
    enabled: bool = True
    blocked_domains: list[str] = Field(default_factory=list)
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    autonomous_topic: AutonomousTopicConfig = Field(default_factory=AutonomousTopicConfig)
    answer_with_search: AnswerWithSearchConfig = Field(default_factory=AnswerWithSearchConfig)


class LocalChatConfig(BaseModel):
    """ローカルテストモード（ローカルユーザーと1対1でチャットする）の設定"""
    user_name: str = "マスター"     # チャット画面の発言者名
    priority: float = 10.0          # 常に最優先で応答する（1対1なので取りこぼさない）
    idle_talk: bool = False         # 会話の合間に独り言を言うか（既定は言わない）
    autostart: bool = True          # 未起動のままチャット送信したら自動でローカルモードを開始
    greeting: str = ""              # 開始時に一度だけ話させたいことがあれば（空なら何もしない）
    # ── 記憶の残るチャット ──────────────────────────────────
    persist_history: bool = True    # 会話をキャラクターごとのファイルに残す
    history_max_entries: int = 1000  # 残す発言数の上限（超えたら古い順に捨てる）
    restore_context_turns: int = 8  # 起動時にLLMへ読み戻す過去ターン数
    consolidate_every_turns: int = 6  # このターン数ごとにエピソード記憶へ整理（0で無効）


class STTConfig(BaseModel):
    """音声認識（マイクで話しかける機能）の設定"""
    enabled: bool = True
    backend: str = "faster_whisper"
    model: str = "small"          # tiny / base / small / medium / large-v3
    device: str = "auto"          # auto / cuda / cpu
    compute_type: str = ""        # 空ならdeviceに応じて自動（cuda:float16 / cpu:int8）
    language: str = "ja"
    speaker_name: str = "マスター"  # 認識結果をコメントとして扱うときの発話者名
    priority: float = 9.0          # 視聴者コメントより優先させる
    min_duration_ms: int = 400     # これより短い音声は無視（物音対策）
    warmup_on_start: bool = True   # 起動時にモデルをロードしておく


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000


# ── ルート設定 ─────────────────────────────────────────

class AppConfig(BaseModel):
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    twitcasting: TwitcastingConfig = Field(default_factory=TwitcastingConfig)
    twitch: TwitchConfig = Field(default_factory=TwitchConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    # 出演キャラクター（先頭がリーダー。人格QLoRA・プリセット・記憶編集の対象）
    characters: list[CharacterConfig] = Field(default_factory=lambda: [CharacterConfig()])
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    audio_output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    comment_filter: CommentFilterConfig = Field(default_factory=CommentFilterConfig)
    viewer_points: ViewerPointsConfig = Field(default_factory=ViewerPointsConfig)
    effects: EffectsConfig = Field(default_factory=EffectsConfig)
    thoughts: ThoughtsConfig = Field(default_factory=ThoughtsConfig)
    elyth_memory: ElythMemoryConfig = Field(default_factory=ElythMemoryConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    local_chat: LocalChatConfig = Field(default_factory=LocalChatConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    ollama: OllamaServiceConfig = Field(default_factory=OllamaServiceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def character(self) -> CharacterConfig:
        """リーダー（先頭）のキャラクター。1体運用だった頃の config.character と同じもの"""
        if not self.characters:
            self.characters.append(CharacterConfig())
        return self.characters[0]

    @property
    def cast(self) -> list[CharacterConfig]:
        """いま出演しているキャラクター（enabled のもの。リーダーは常に含む）"""
        lead = self.character
        return [lead] + [c for c in self.characters[1:] if c.enabled]


CharacterConfig.model_rebuild()


def load_dotenv(path: str | Path = ".env") -> int:
    """
    .env からAPIキーを読み込む（python-dotenv不要の最小実装）。

    既に環境変数が設定されている場合はそちらを優先する。
    KEY=VALUE 形式のみを見て、# 始まりの行と空行は無視する。
    戻り値は読み込んだ件数。
    """
    path = Path(path)
    if not path.exists():
        return 0
    loaded = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or os.getenv(key):
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 古いconfig.yamlとの後方互換：不足キーをデフォルト値で補完
    raw.setdefault("twitcasting", {})
    raw.setdefault("twitch", {})
    # 1体運用の頃の character: を characters: の先頭に移す
    legacy = raw.pop("character", None)
    if not raw.get("characters"):
        raw["characters"] = [legacy or {}]
    for ch in raw["characters"]:
        ch.setdefault("stream_theme", "")
    raw.setdefault("orchestrator", {})
    raw["orchestrator"].setdefault("consolidation_trigger_turns", 50)
    raw["orchestrator"].setdefault("response_history_turns", 20)
    # ハイブリッド振り分け（旧 config には無いので既定で補う）
    raw.setdefault("llm", {})
    if isinstance(raw.get("llm"), dict):
        raw["llm"].setdefault("hybrid", {})
    raw.setdefault("avatar", {})
    raw["avatar"].setdefault("world", {})
    raw["avatar"].setdefault("live2d", {})
    raw["avatar"].setdefault("video", {})
    raw.setdefault("safety", {})
    raw.setdefault("knowledge", {})
    raw.setdefault("local_chat", {})
    raw.setdefault("stt", {})

    try:
        return AppConfig.model_validate(raw)
    except Exception as e:
        import logging
        logging.warning(f"config.yaml parse error: {e}, using defaults")
        return AppConfig()


def save_config(cfg: AppConfig, path: str | Path = "config.yaml") -> None:
    """
    AppConfigをconfig.yamlに書き戻す。

    以前はセクションごとに手書きでdictを組んでいたため、設定項目を足すたびに
    ここへの追記漏れで「画面で変えた値が保存されない」不具合が起きていた。
    モデルをそのまま吐き出すので、項目を足してもここは変更不要。
    """
    data = cfg.model_dump(mode="json")
    # 書き込み途中で落ちても config.yaml が壊れないよう、一時ファイル経由で置き換える
    tmp = Path(path).with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)
