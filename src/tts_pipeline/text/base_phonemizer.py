from __future__ import annotations

from abc import ABC, abstractmethod


class BasePhonemizer(ABC):
    """Language-agnostic G2P interface.

    All language-specific phonemizers must implement this contract.
    The rest of the system (feature extraction, training, serving) depends
    only on this interface — swapping a language requires only a new subclass.
    """

    @abstractmethod
    def phonemize(self, text: str) -> list[str]:
        """Convert normalized text to a list of phoneme tokens.

        Args:
            text: Clean, normalized input text (post transcript_cleaner).

        Returns:
            Ordered list of phoneme strings from this phonemizer's symbol set.
        """
        ...

    @abstractmethod
    def get_symbol_set(self) -> set[str]:
        """Return the complete phoneme vocabulary for this language.

        Must include any special tokens (_PAD_, _BOS_, _EOS_, _BLANK_) that
        this phonemizer emits.
        """
        ...

    def phonemize_batch(self, texts: list[str]) -> list[list[str]]:
        """Phonemize a list of texts. Override for batched efficiency."""
        return [self.phonemize(t) for t in texts]

    def validate_coverage(self, phoneme_lists: list[list[str]], min_count: int = 10) -> dict[str, int]:
        """Return phonemes with fewer than min_count occurrences in the corpus."""
        from collections import Counter

        counts: Counter[str] = Counter(p for plist in phoneme_lists for p in plist)
        return {p: counts[p] for p in self.get_symbol_set() if counts.get(p, 0) < min_count}
