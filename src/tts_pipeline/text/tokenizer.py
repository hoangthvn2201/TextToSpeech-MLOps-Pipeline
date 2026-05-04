from __future__ import annotations

import json
from pathlib import Path

from tts_pipeline.text.symbols import BLANK, BOS, EOS, PAD, build_symbol_list


class Tokenizer:
    """Maps phoneme tokens ↔ integer IDs.

    The vocabulary is built from the language phoneme inventory and optionally
    loaded from / saved to a JSON file (the phoneme_vocab.json in the model bundle).
    """

    def __init__(self, symbols: list[str], add_blank: bool = True) -> None:
        self._symbols = symbols
        self._add_blank = add_blank
        self._sym2id = {s: i for i, s in enumerate(symbols)}
        self._id2sym = {i: s for i, s in enumerate(symbols)}

    # ── Vocabulary properties ─────────────────────────────────────────────────
    @property
    def pad_id(self) -> int:
        return self._sym2id[PAD]

    @property
    def bos_id(self) -> int:
        return self._sym2id[BOS]

    @property
    def eos_id(self) -> int:
        return self._sym2id[EOS]

    @property
    def blank_id(self) -> int:
        return self._sym2id[BLANK]

    @property
    def vocab_size(self) -> int:
        return len(self._symbols)

    # ── Encode / decode ───────────────────────────────────────────────────────
    def encode(self, phonemes: list[str], add_bos_eos: bool = True) -> list[int]:
        ids = [self._sym2id.get(p, self.pad_id) for p in phonemes]
        if self._add_blank:
            ids = self._intersperse(ids, self.blank_id)
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id2sym.get(i, PAD) for i in ids]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"symbols": self._symbols, "add_blank": self._add_blank}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(symbols=data["symbols"], add_blank=data.get("add_blank", True))

    @classmethod
    def from_inventory(cls, phoneme_inventory: list[str], add_blank: bool = True) -> "Tokenizer":
        symbols = build_symbol_list(phoneme_inventory, add_blank=add_blank)
        return cls(symbols=symbols, add_blank=add_blank)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _intersperse(ids: list[int], blank_id: int) -> list[int]:
        result = [blank_id] * (2 * len(ids) + 1)
        result[1::2] = ids
        return result
