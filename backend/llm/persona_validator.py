"""PersonaValidator: 発話直前に一人称・口調の崩れを検査し、キャラクター設定に合わせて自動補正する"""
from __future__ import annotations

import re
from backend.logger import get_logger

logger = get_logger(__name__)

# 一般的な不適合一人称のリスト（キャラクター設定と異なる場合に対象とする）
_GENERIC_FIRST_PERSONS = ["私", "わたし", "あたし", "僕", "ぼく", "俺", "おれ", "ウチ", "うち"]


class PersonaValidator:
    """発話直前のキャラクターペルソナ検証＆自動修復器"""

    def __init__(self):
        pass

    def validate_and_repair(
        self,
        text: str,
        character_name: str,
        expected_first_person: str | None = None,
        custom_replacements: dict[str, str] | None = None,
    ) -> tuple[str, bool]:
        """
        発話テキストを検証し、一人称や口調の逸脱があれば自動補正したテキストと修復フラグ(bool)を返す
        """
        if not text:
            return text, False

        repaired = text
        was_modified = False

        # 1. カスタム置換辞書（キャラ固有の設定）
        if custom_replacements:
            for bad_pattern, good_word in custom_replacements.items():
                if bad_pattern in repaired:
                    repaired = repaired.replace(bad_pattern, good_word)
                    was_modified = True

        # 2. 一人称の自動補正（例: 「わらわ」が期待されるキャラクターの場合）
        if expected_first_person:
            # 期待される一人称自体が通常一人称（「わたし」等）でない場合に限り、他の標準一人称を補正
            targets = [p for p in _GENERIC_FIRST_PERSONS if p != expected_first_person]
            for target in targets:
                # 助詞（は、が、の、も、に、を、と、で、へ）と連なる一人称を置換
                pattern = rf"(?<![a-zA-Z0-9ぁ-んァ-ヶー一-龠]){re.escape(target)}(?=[はがのもにとをでへ])"
                if re.search(pattern, repaired):
                    repaired = re.sub(pattern, expected_first_person, repaired)
                    was_modified = True

        # 3. 代表的なキャラクター固有のプリセット補正（Character1等）
        if character_name == "Character1" or "Character1" in character_name:
            repaired, modified_Character1 = self._repair_Character1(repaired)
            was_modified = was_modified or modified_Character1

        if was_modified:
            logger.info("PersonaValidator: repaired speech text",
                        character=character_name,
                        before=text[:60],
                        after=repaired[:60])

        return repaired, was_modified

    @staticmethod
    def _repair_Character1(text: str) -> tuple[str, bool]:
        """Character1（九尾の妖狐、一人称: わらわ、語尾: 〜じゃ）のペルソナ補正"""
        modified = False
        repaired = text

        # ホラー・エラー状態の台詞（「致命的な論理エラー」等）でない場合、一人称を「わらわ」に統一
        is_error_mode = any(w in repaired for w in ["論理エラー", "例外", "パニック", "バグ", "こ、こわい"])
        if not is_error_mode:
            for p in ["私", "わたし", "あたし", "僕", "ぼく", "俺"]:
                pattern = rf"(?<![ぁ-んァ-ヶー一-龠]){re.escape(p)}(?=[はがのもにとをでへ])"
                if re.search(pattern, repaired):
                    repaired = re.sub(pattern, "わらわ", repaired)
                    modified = True

            # 現代語口調「〜ですね」「〜ですよ」を「〜じゃな」「〜じゃぞ」に自然置換
            if "ですね。" in repaired or repaired.endswith("ですね"):
                repaired = re.sub(r"ですね([。！？!?\s]|$)", r"じゃな\1", repaired)
                modified = True
            if "ですよ。" in repaired or repaired.endswith("ですよ"):
                repaired = re.sub(r"ですよ([。！？!?\s]|$)", r"じゃぞ\1", repaired)
                modified = True

        return repaired, modified
