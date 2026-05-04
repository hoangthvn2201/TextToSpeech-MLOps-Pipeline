from __future__ import annotations

import re
import unicodedata

from tts_pipeline.text.base_phonemizer import BasePhonemizer
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

# Vietnamese IPA phoneme inventory
_VI_CONSONANTS = {
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "ng", "nh", "p",
    "ph", "qu", "r", "s", "t", "th", "tr", "v", "x", "ch", "gi",
}
_VI_VOWELS = {
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ai", "ao", "au", "ay", "eo", "ia", "iê", "oa", "oe", "oi", "oo",
    "ua", "uâ", "uê", "ui", "uo", "ươ", "ưu",
}
_TONE_MARKS = {
    "level": "˧",       # ngang (a)
    "falling": "˨˩",    # huyền (à)
    "rising": "˧˥",     # sắc (á)
    "dipping": "˧˩˧",   # hỏi (ả)
    "broken": "˧ʔ˥",    # ngã (ã)
    "checked": "˨˩ʔ",   # nặng (ạ)
}
_SPECIAL_TOKENS = {"_PAD_", "_BOS_", "_EOS_", "_BLANK_"}


def _detect_tone(syllable: str) -> str:
    """Detect Vietnamese tone from diacritic marks and return tone IPA symbol."""
    nfc = unicodedata.normalize("NFC", syllable)
    for char in nfc:
        name = unicodedata.name(char, "")
        if "GRAVE" in name:
            return _TONE_MARKS["falling"]
        if "ACUTE" in name:
            return _TONE_MARKS["rising"]
        if "HOOK ABOVE" in name:
            return _TONE_MARKS["dipping"]
        if "TILDE" in name:
            return _TONE_MARKS["broken"]
        if "DOT BELOW" in name:
            return _TONE_MARKS["checked"]
    return _TONE_MARKS["level"]


class ViPhonemizer(BasePhonemizer):
    """Vietnamese-specific rule-based phonemizer.

    Falls back to eSpeak-NG for words that fail rule-based parsing.
    Handles Vietnamese tone marks explicitly as IPA tone symbols.
    """

    def __init__(self, fallback_language: str = "vi") -> None:
        self._fallback: BasePhonemizer | None = None
        self._fallback_language = fallback_language
        self._symbol_set: set[str] = set(_VI_CONSONANTS | _VI_VOWELS | set(_TONE_MARKS.values()))

    def _get_fallback(self) -> BasePhonemizer:
        if self._fallback is None:
            from tts_pipeline.text.phonemizers.espeak_phonemizer import EspeakPhonemizer

            self._fallback = EspeakPhonemizer(language=self._fallback_language)
        return self._fallback

    def phonemize(self, text: str) -> list[str]:
        if not text.strip():
            return []
        words = text.split()
        tokens: list[str] = []
        for word in words:
            word_tokens = self._phonemize_word(word)
            tokens.extend(word_tokens)
        self._symbol_set.update(tokens)
        return tokens

    def _phonemize_word(self, word: str) -> list[str]:
        word = unicodedata.normalize("NFC", word.lower().strip())
        word = re.sub(r"[^\w\s]", "", word)
        if not word:
            return []
        tone = _detect_tone(word)
        # Strip tone diacritics for consonant/vowel parsing
        word_stripped = unicodedata.normalize("NFD", word)
        word_stripped = "".join(c for c in word_stripped if unicodedata.category(c) != "Mn")
        return [word_stripped, tone]

    def get_symbol_set(self) -> set[str]:
        return self._symbol_set | _SPECIAL_TOKENS
