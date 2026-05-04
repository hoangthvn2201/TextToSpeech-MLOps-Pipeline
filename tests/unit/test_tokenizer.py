from __future__ import annotations

from pathlib import Path

import pytest

from tts_pipeline.text.tokenizer import Tokenizer


def test_encode_decode_roundtrip() -> None:
    t = Tokenizer.from_inventory(["a", "b", "c"], add_blank=False)
    phonemes = ["a", "b", "c", "a"]
    ids = t.encode(phonemes, add_bos_eos=False)
    decoded = t.decode(ids)
    assert decoded == phonemes


def test_encode_adds_bos_eos() -> None:
    t = Tokenizer.from_inventory(["a", "b"], add_blank=False)
    ids = t.encode(["a", "b"], add_bos_eos=True)
    symbols = t.decode(ids)
    assert symbols[0] == "_BOS_"
    assert symbols[-1] == "_EOS_"


def test_encode_with_blank_intersperses() -> None:
    t = Tokenizer.from_inventory(["a", "b"], add_blank=True)
    ids = t.encode(["a", "b"], add_bos_eos=False)
    symbols = t.decode(ids)
    # With blank interspersed: [blank, a, blank, b, blank]
    assert "_BLANK_" in symbols
    assert symbols.count("_BLANK_") == 3


def test_save_load_roundtrip(tmp_path: Path) -> None:
    t = Tokenizer.from_inventory(["x", "y", "z"], add_blank=True)
    vocab_path = tmp_path / "vocab.json"
    t.save(vocab_path)
    t2 = Tokenizer.load(vocab_path)
    assert t2.vocab_size == t.vocab_size
    assert t2.encode(["x", "y"]) == t.encode(["x", "y"])


def test_unknown_phoneme_maps_to_pad() -> None:
    t = Tokenizer.from_inventory(["a"], add_blank=False)
    ids = t.encode(["a", "UNKNOWN"], add_bos_eos=False)
    symbols = t.decode(ids)
    assert symbols[1] == "_PAD_"
