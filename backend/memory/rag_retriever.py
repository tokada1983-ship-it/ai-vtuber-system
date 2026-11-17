"""RAGRetriever: 長期記憶に対するRAG（検索拡張生成）

動作モード:
  1. sentence-transformers が利用可能な場合 → 多言語ベクトル検索（高品質）
  2. 利用不可の場合 → TF-IDF + コサイン類似度（軽量フォールバック）

インストール（推奨）:
  pip install sentence-transformers
  モデル: paraphrase-multilingual-MiniLM-L12-v2 （約120MB、日本語対応）
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.logger import get_logger

logger = get_logger(__name__)

# 使用するSentenceTransformerモデル（日本語対応・軽量）
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# これより長い連なりは語トークンとして扱わない（日本語の文がまるごと1トークンになるのを防ぐ）
_MAX_WORD_TOKEN_LEN = 8


def _is_cjk(ch: str) -> bool:
    """漢字・ひらがな・カタカナかどうか"""
    return (
        "぀" <= ch <= "ゟ"   # ひらがな
        or "゠" <= ch <= "ヿ"  # カタカナ
        or "一" <= ch <= "鿿"  # 漢字
    )


@dataclass
class MemoryChunk:
    text: str               # チャンクのテキスト
    source_line: int        # 元ファイルの行番号
    section: str            # 所属セクション（見出し）
    score: float = 0.0      # 検索スコア


class RAGRetriever:
    """
    長期記憶MDをチャンク化してベクトルインデックスを構築し、
    クエリに関連するチャンクを検索する。

    使い方:
        retriever = RAGRetriever()
        retriever.index(memory_text)
        results = retriever.search("ゲームについて", top_k=3)
        for chunk in results:
            print(chunk.text, chunk.score)
    """

    def __init__(self):
        self._chunks: list[MemoryChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._tfidf_matrix_np: Optional[np.ndarray] = None
        self._vocab: Optional[list[str]] = None
        self._use_st = False
        self._model = None
        self._indexed_text = ""
        self._model_tried = False   # モデルは初回インデックス時に遅延ロードする

    # ── 重複判定しきい値 ──────────────────────────────────────

    # 埋め込みモデルとTF-IDFではスコアの分布がまったく違うため、同じ数値を
    # 使い回すと片方で機能しない。実測（同一事実の言い換え / 別の事実）:
    #
    #   sentence-transformers : 0.87〜1.00 / 0.18〜0.71  → 0.82で綺麗に分離できる
    #   TF-IDF（文字N-gram）  : 0.58〜0.99 / 0.50〜0.67  → 分離不能（逆転もする）
    #
    # TF-IDFでは「タナカさんはゲームが好き」と「タナカさんは猫が好き」のように
    # 文型が同じで中身が違う組が高スコアになり、言い換えより上に来ることがある。
    # 誤って別の事実を1つにまとめると情報が消えるので、TF-IDF側は
    # ほぼ完全一致だけを弾く高いしきい値にしておく。
    # 言い換えの取りこぼしで記憶が増えるぶんは、MemoryStoreの件数上限が受け持つ。
    _DEDUPE_THRESHOLD_ST = 0.82
    _DEDUPE_THRESHOLD_TFIDF = 0.85

    @property
    def dedupe_threshold(self) -> float:
        """現在の検索モードに応じた「同じ内容とみなす」しきい値"""
        self._ensure_model()
        return self._DEDUPE_THRESHOLD_ST if self._use_st else self._DEDUPE_THRESHOLD_TFIDF

    @property
    def uses_embeddings(self) -> bool:
        """意味の近さで判定できるか（Falseなら文字列一致で代用する必要がある）"""
        self._ensure_model()
        return self._use_st

    def _ensure_model(self) -> None:
        """
        sentence-transformers を初回利用時にロードする。
        起動時にロードするとサーバの立ち上がりが数秒遅くなるため遅延させている。
        """
        if self._model_tried:
            return
        self._model_tried = True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_ST_MODEL_NAME)
            self._use_st = True
            logger.info("RAG: using sentence-transformers", model=_ST_MODEL_NAME)
        except ImportError:
            logger.info("RAG: sentence-transformers not found, using TF-IDF fallback")
        except Exception as e:
            logger.warning("RAG: sentence-transformers init failed, using TF-IDF", error=str(e))

    # ── チャンク化 ─────────────────────────────────────────────

    def _chunk(self, text: str) -> list[MemoryChunk]:
        """
        MDテキストをセクション付きチャンクに分割する。
        - 見出し行（## 〜）でセクションを更新
        - 箇条書き行（- 〜）を1チャンクとして扱う
        - 通常の文は句点で分割
        - 短すぎるチャンク（5文字未満）は除外
        """
        chunks: list[MemoryChunk] = []
        current_section = "一般"

        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue

            # 見出し
            if re.match(r"^#{1,3}\s*", line):
                current_section = re.sub(r"^#+\s*", "", line).strip()
                continue

            # 箇条書き → そのまま1チャンク
            if re.match(r"^[-*]\s*", line):
                body = re.sub(r"^[-*]\s*", "", line).strip()
                if len(body) >= 5:
                    chunks.append(MemoryChunk(
                        text=body,
                        source_line=i,
                        section=current_section,
                    ))
                continue

            # 通常テキスト → 句点で分割
            sentences = re.split(r"[。！？\n]", line)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) >= 5:
                    chunks.append(MemoryChunk(
                        text=sent,
                        source_line=i,
                        section=current_section,
                    ))

        return chunks

    # ── インデックス構築 ──────────────────────────────────────

    def index_texts(self, texts: list[str]) -> None:
        """
        1件=1チャンクとしてそのまま索引する（MemoryStore用）。

        index() はMarkdownを見出し・箇条書きで刻むが、エピソード記憶は
        既に1件ずつ独立しているので刻んではいけない。検索結果の
        source_line に元リストの添字が入るので、呼び出し側はそれで
        エントリを引ける。
        """
        self._reset_index()
        self._indexed_text = "\x00".join(texts)  # index() との相互キャッシュ汚染を防ぐ
        self._chunks = [
            MemoryChunk(text=t.strip(), source_line=i, section="episodic")
            for i, t in enumerate(texts) if t.strip()
        ]
        if not self._chunks:
            return
        self._ensure_model()
        if self._use_st and self._model:
            self._build_st_index()
        else:
            self._build_tfidf_index()

    def _reset_index(self) -> None:
        self._chunks = []
        self._embeddings = None
        self._tfidf_matrix_np = None
        self._vocab = None

    def index(self, memory_text: str) -> None:
        """長期記憶テキストからインデックスを構築する"""
        if not memory_text.strip():
            self._reset_index()
            self._indexed_text = ""
            return

        if memory_text == self._indexed_text:
            return  # 変更なしならスキップ

        self._indexed_text = memory_text
        self._chunks = self._chunk(memory_text)

        if not self._chunks:
            return

        self._ensure_model()
        if self._use_st and self._model:
            self._build_st_index()
        else:
            self._build_tfidf_index()

        logger.info("RAG index built", chunks=len(self._chunks), mode="st" if self._use_st else "tfidf")

    def _build_st_index(self) -> None:
        texts = [c.text for c in self._chunks]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)

    def _build_tfidf_index(self) -> None:
        """TF-IDF ベクトルを構築する（日本語は文字N-gramで近似）"""
        docs = [self._tokenize(c.text) for c in self._chunks]
        # 語彙構築
        vocab_set: set[str] = set()
        for doc in docs:
            vocab_set.update(doc.keys())
        self._vocab = sorted(vocab_set)
        vocab_idx = {w: i for i, w in enumerate(self._vocab)}

        # DF（文書頻度）計算
        n = len(docs)
        df = Counter()
        for doc in docs:
            for w in doc:
                df[w] += 1

        # TF-IDF行列
        matrix = np.zeros((n, len(self._vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            total = sum(doc.values())
            for w, cnt in doc.items():
                if w in vocab_idx:
                    tf = cnt / total
                    idf = math.log((n + 1) / (df[w] + 1)) + 1
                    matrix[i, vocab_idx[w]] = tf * idf

        # L2正規化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._tfidf_matrix_np = matrix / norms

    def _tokenize(self, text: str) -> Counter:
        """日本語テキストをbigramで近似トークナイズ"""
        tokens: list[str] = []
        for w in re.findall(r"[ぁ-んァ-ン一-龥ａ-ｚＡ-Ｚa-zA-Z0-9]+", text):
            # 短い語はそのまま1トークンにする（英単語や「猫」などに効く）。
            # 日本語は分かち書きされないので、長い連なりをそのまま入れても
            # 完全一致以外では二度とヒットせず、重みを食うだけになる。
            if len(w) <= _MAX_WORD_TOKEN_LEN:
                tokens.append(w)
            # 文字バイグラム
            for j in range(len(w) - 1):
                tokens.append(w[j:j + 2])
            # 文字ユニグラム。これが無いと「猫」1文字の検索が
            # 「猫を3匹飼っている」に一切マッチしなかった
            if len(w) > 1:
                tokens.extend(ch for ch in w if _is_cjk(ch))
        return Counter(tokens)

    # ── 検索 ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[MemoryChunk]:
        """クエリに関連するチャンクをtop_k件返す"""
        if not self._chunks or not query.strip():
            return []

        if self._use_st and self._model and self._embeddings is not None:
            scores = self._search_st(query)
        elif self._tfidf_matrix_np is not None:
            scores = self._search_tfidf(query)
        else:
            return []

        # スコア付きでソート
        results = []
        for i, score in enumerate(scores):
            chunk = MemoryChunk(
                text=self._chunks[i].text,
                source_line=self._chunks[i].source_line,
                section=self._chunks[i].section,
                score=float(score),
            )
            results.append(chunk)

        results.sort(key=lambda c: c.score, reverse=True)
        # スコアが低すぎるものは除外（閾値0.1）
        results = [c for c in results if c.score >= 0.1]
        return results[:top_k]

    def _search_st(self, query: str) -> np.ndarray:
        q_emb = self._model.encode([query], normalize_embeddings=True)
        return (self._embeddings @ q_emb.T).flatten()

    def _search_tfidf(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        vocab_idx = {w: i for i, w in enumerate(self._vocab)}
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        total = sum(q_tokens.values())
        for w, cnt in q_tokens.items():
            if w in vocab_idx:
                q_vec[vocab_idx[w]] = cnt / total
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return np.zeros(len(self._chunks))
        q_vec /= norm
        return self._tfidf_matrix_np @ q_vec

    # ── プロンプト生成 ────────────────────────────────────────

    def build_rag_prompt(self, query: str, top_k: int = 3) -> str:
        """
        検索結果をプロンプト用テキストに変換する。
        関連チャンクが見つからない場合は空文字を返す。
        """
        chunks = self.search(query, top_k=top_k)
        if not chunks:
            return ""

        lines = ["## 関連する記憶（RAG検索結果）"]
        lines.append("以下はこの会話に関連するキャラクターの記憶です：")
        lines.append("")

        # セクションでグループ化
        by_section: dict[str, list[str]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk.section, []).append(chunk.text)

        for section, texts in by_section.items():
            lines.append(f"**{section}**")
            for t in texts:
                lines.append(f"- {t}")
        lines.append("")

        return "\n".join(lines)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def mode(self) -> str:
        return "sentence-transformers" if self._use_st else "tfidf"
