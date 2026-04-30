from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))


def normalize_punctuation(text: str) -> str:
    text = text.replace("—", ", ").replace("–", ", ")  # em/en dash → comma
    text = text.replace("“", '"').replace("”", '"')    # smart quotes
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[!?]+", ".", text)                           # normalize sentence endings
    text = re.sub(r"\.{2,}", ".", text)                          # collapse ellipsis
    return text


def expand_numbers(text: str, language: str = "vi") -> str:
    """Expand digits to words. Uses inflect for English; basic rules for others."""
    if re.search(r"\d", text):
        try:
            import inflect

            _engine = inflect.engine()

            def replace_num(m: re.Match[str]) -> str:
                return _engine.number_to_words(int(m.group()))

            text = re.sub(r"\b\d+\b", replace_num, text)
        except Exception:
            pass  # keep digits if expansion fails
    return text


def clean_text(text: str, language: str = "vi") -> str:
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = normalize_punctuation(text)
    text = expand_numbers(text, language=language)
    text = collapse_whitespace(text)
    return text
