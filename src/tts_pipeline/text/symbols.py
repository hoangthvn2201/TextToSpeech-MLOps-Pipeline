from __future__ import annotations

# Special tokens always present in every language's vocabulary
PAD = "_PAD_"
BOS = "_BOS_"
EOS = "_EOS_"
BLANK = "_BLANK_"  # used between phonemes when add_blank=True

SPECIAL_TOKENS = [PAD, BOS, EOS, BLANK]


def build_symbol_list(phoneme_inventory: list[str], add_blank: bool = True) -> list[str]:
    """Build the ordered symbol list: specials first, then sorted phonemes."""
    phonemes = sorted(set(phoneme_inventory) - set(SPECIAL_TOKENS))
    symbols = list(SPECIAL_TOKENS) + phonemes
    if add_blank and BLANK not in symbols:
        symbols.append(BLANK)
    return symbols
