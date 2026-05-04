from __future__ import annotations

from functools import lru_cache

from tts_pipeline.text.base_phonemizer import BasePhonemizer
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_SPECIAL_TOKENS = {"_PAD_", "_BOS_", "_EOS_", "_BLANK_"}


class EspeakPhonemizer(BasePhonemizer):
    """eSpeak-NG backed phonemizer supporting 50+ languages via IPA output.

    Requires: pip install phonemizer + espeak-ng system package
    """

    def __init__(self, language: str = "vi", with_stress: bool = True) -> None:
        self.language = language
        self.with_stress = with_stress
        self._backend = self._build_backend()
        self._symbol_set: set[str] = set()

    def _build_backend(self) -> object:
        try:
            from phonemizer.backend import EspeakBackend

            return EspeakBackend(
                language=self.language,
                with_stress=self.with_stress,
                language_switch="remove-flags",
            )
        except ImportError as e:
            raise ImportError("Install phonemizer: pip install phonemizer") from e

    def phonemize(self, text: str) -> list[str]:
        if not text.strip():
            return []
        result = self._backend.phonemize([text], separator=_Separator())[0]
        tokens = [t for t in result.split() if t]
        self._symbol_set.update(tokens)
        return tokens

    def phonemize_batch(self, texts: list[str]) -> list[list[str]]:
        from phonemizer.separator import Separator

        non_empty = [t for t in texts]
        results = self._backend.phonemize(non_empty, separator=Separator(phone=" ", word="| "))
        token_lists = []
        for r in results:
            tokens = [t for t in r.replace("|", "").split() if t]
            self._symbol_set.update(tokens)
            token_lists.append(tokens)
        return token_lists

    def get_symbol_set(self) -> set[str]:
        return self._symbol_set | _SPECIAL_TOKENS


class _Separator:
    phone = " "
    word = " "
    syllable = ""
