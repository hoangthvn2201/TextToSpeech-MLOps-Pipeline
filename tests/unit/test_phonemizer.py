from __future__ import annotations

import pytest

from tts_pipeline.text.base_phonemizer import BasePhonemizer


class _StubPhonemizer(BasePhonemizer):
    def phonemize(self, text: str) -> list[str]:
        return text.split()

    def get_symbol_set(self) -> set[str]:
        return {"a", "b", "c", "_PAD_", "_BOS_", "_EOS_", "_BLANK_"}


def test_phonemize_empty() -> None:
    p = _StubPhonemizer()
    assert p.phonemize("") == []
    assert p.phonemize("   ") == []


def test_phonemize_batch_consistent() -> None:
    p = _StubPhonemizer()
    texts = ["hello world", "foo bar baz"]
    batch = p.phonemize_batch(texts)
    for t, b in zip(texts, batch):
        assert b == p.phonemize(t)


def test_validate_coverage_detects_sparse() -> None:
    p = _StubPhonemizer()
    # Only 'a' and 'b' appear, 'c' never appears
    lists = [["a", "b"], ["a", "a", "b"]]
    sparse = p.validate_coverage(lists, min_count=2)
    assert "c" in sparse
    assert sparse["c"] == 0
    assert "a" not in sparse  # 'a' appears 3 times
