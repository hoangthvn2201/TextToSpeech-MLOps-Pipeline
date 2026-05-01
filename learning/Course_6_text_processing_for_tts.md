# Course 6: Text Processing for TTS

**Prerequisites:** Course 1 (Python for ML Engineers) — regex, ABC, lru_cache, optional imports.

**What this course covers:** The text pipeline converts raw transcript strings into integer sequences the model can consume. By the end you will understand every transformation from `"Tôi có 3 con mèo"` to `[1, 4, 0, 17, 0, 8, ...]`, why each step exists, and what breaks if you skip one.

**Project anchors:**
- [src/tts_pipeline/text/cleaners.py](../src/tts_pipeline/text/cleaners.py) — `clean_text` and all sub-cleaners
- [src/tts_pipeline/text/base_phonemizer.py](../src/tts_pipeline/text/base_phonemizer.py) — `BasePhonemizer` ABC
- [src/tts_pipeline/text/phonemizers/espeak_phonemizer.py](../src/tts_pipeline/text/phonemizers/espeak_phonemizer.py) — `EspeakPhonemizer`
- [src/tts_pipeline/text/phonemizers/vi_phonemizer.py](../src/tts_pipeline/text/phonemizers/vi_phonemizer.py) — `ViPhonemizer`
- [src/tts_pipeline/text/symbols.py](../src/tts_pipeline/text/symbols.py) — `build_symbol_list`, special tokens
- [src/tts_pipeline/text/tokenizer.py](../src/tts_pipeline/text/tokenizer.py) — `Tokenizer`
- [src/tts_pipeline/data/transcript_cleaner.py](../src/tts_pipeline/data/transcript_cleaner.py) — `clean_metadata`

---

## Module 1 — The Full Pipeline

Before diving into individual pieces, trace the full journey of one sentence:

```
Raw text:     "Tôi có 3 con mèo."
      ↓ clean_text()
Cleaned:      "Tôi có ba con mèo."       ← digits expanded, punctuation normalized
      ↓ phonemizer.phonemize()
Phonemes:     ["toi", "˦˥", "ko", "˨˩˦", "ba", ...]    ← G2P: graphemes → phoneme tokens
      ↓ tokenizer.encode()
IDs:          [1, 4, 0, 17, 0, 8, 0, 22, ...]           ← BLANK interspersed, BOS/EOS added
      ↓ Embedding lookup
Tensor:       shape (T_text, encoder_dim=192)             ← model input
```

Each arrow is a stage. This course covers stages 1–3.

### Where each file lives in this pipeline

```
transcript_cleaner.py → clean_metadata()
    └─ calls cleaners.py → clean_text()
            └─ normalize_unicode()
            └─ remove_control_characters()
            └─ normalize_punctuation()
            └─ expand_numbers()
            └─ collapse_whitespace()

phonemizers/
    base_phonemizer.py   → BasePhonemizer (ABC)
    espeak_phonemizer.py → EspeakPhonemizer (IPA via eSpeak-NG)
    vi_phonemizer.py     → ViPhonemizer (rule-based + fallback)

symbols.py   → SPECIAL_TOKENS, build_symbol_list()
tokenizer.py → Tokenizer.encode() / decode() / save() / load()
```

---

## Module 2 — Unicode Fundamentals

Every cleaner in this project depends on `unicodedata`. Understanding Unicode normalization is prerequisite to understanding why the cleaners are written the way they are.

### What Unicode normalization means

A single visible character can be encoded in multiple byte sequences. The letter `ô` (o with circumflex) exists as:

| Form | Codepoints | Bytes (UTF-8) |
|---|---|---|
| Precomposed (NFC) | U+00F4 (one code point) | `0xC3 0xB4` |
| Decomposed (NFD) | U+006F + U+0302 (base + combining) | `0x6F 0xCC 0x82` |

Both render identically. But `"ô" == "ô"` is **False** if one is NFC and the other is NFD. String comparison, hashing, and regex all break.

### The four Unicode normal forms

| Form | What it does | Use when |
|---|---|---|
| NFC | Compose characters into single code points where possible | Storage, display, string comparison |
| NFD | Decompose to base + combining marks | Manipulating diacritics individually |
| NFKC | NFC + compatibility mapping (ligatures, fullwidth → ASCII) | Search/matching across character variants |
| NFKD | NFD + compatibility mapping | Stripping diacritics completely |

Our project uses:
- `NFC` in `normalize_unicode()` — canonical form for storage/comparison
- `NFC` in `ViPhonemizer._phonemize_word()` before parsing
- `NFD` in `ViPhonemizer._phonemize_word()` before stripping tone marks (decompose so marks become separate code points that can be filtered)

### `unicodedata.category()`

Every Unicode code point has a category:

| Category code | Name | Examples |
|---|---|---|
| `Lu` | Uppercase Letter | A, B, Ô |
| `Ll` | Lowercase Letter | a, b, ô |
| `Nd` | Decimal digit | 0–9 |
| `Cc` | Control character | `\n`, `\t`, `\x00` |
| `Cf` | Format character | Zero-width space, BOM |
| `Mn` | Non-spacing mark (combining) | Diacritics in NFD form |
| `Po` | Punctuation, other | `.`, `,`, `!` |

```python
import unicodedata
unicodedata.category("\n")          # "Cc" — control
unicodedata.category("​")      # "Cf" — zero-width space
unicodedata.category("̂")      # "Mn" — combining circumflex (NFD)
unicodedata.category("a")           # "Ll" — lowercase letter
```

### How `remove_control_characters` uses this

```python
def remove_control_characters(text: str) -> str:     # cleaners.py:15
    return "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))
```

This removes invisible characters that a speaker would never pronounce: carriage returns, zero-width spaces, BOM markers. Keeping them would produce silent frames or garbled phonemes.

### How `ViPhonemizer` uses NFD to strip tone marks

```python
word_stripped = unicodedata.normalize("NFD", word)                    # vi_phonemizer.py:87
word_stripped = "".join(c for c in word_stripped if unicodedata.category(c) != "Mn")
```

`"mèo"` in NFD decomposes to `"me" + combining grave + "o"`. Filtering `Mn` yields `"meo"`. The tone (`˨˩`) was already extracted by `_detect_tone()` and stored separately. Result: `["meo", "˨˩"]`.

---

## Module 3 — Text Cleaning

### The pipeline: `clean_text()`

```python
def clean_text(text: str, language: str = "vi") -> str:   # cleaners.py:45
    text = normalize_unicode(text)         # 1. NFC normalization
    text = remove_control_characters(text) # 2. strip Cc/Cf
    text = normalize_punctuation(text)     # 3. canonical punctuation
    text = expand_numbers(text, language)  # 4. digits → words
    text = collapse_whitespace(text)       # 5. merge runs of whitespace
    return text
```

Order matters. Unicode must be normalized before punctuation matching (otherwise smart quotes in NFD form won't match the string literals). Digits must be expanded before whitespace collapse (expansion can introduce spaces).

### `normalize_punctuation()`

```python
def normalize_punctuation(text: str) -> str:             # cleaners.py:19
    text = text.replace("—", ", ").replace("–", ", ")   # em/en dash → comma pause
    text = text.replace("“", '"').replace("”", '"')  # smart quotes
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[!?]+", ".", text)     # !! or ?? → single period
    text = re.sub(r"\.{2,}", ".", text)    # ... → .
    return text
```

Why collapse `!` and `?` to `.`? Because our phonemizer doesn't model prosody from punctuation — the acoustic model learns rhythm from the mel spectrogram, not from sentence-ending marks. Normalizing them prevents one-off symbols from inflating the vocabulary.

### `expand_numbers()`

```python
def expand_numbers(text: str, language: str = "vi") -> str:   # cleaners.py:28
    if re.search(r"\d", text):
        try:
            import inflect
            _engine = inflect.engine()
            def replace_num(m):
                return _engine.number_to_words(int(m.group()))
            text = re.sub(r"\b\d+\b", replace_num, text)
        except Exception:
            pass  # keep digits if expansion fails
    return text
```

Key points:
- `re.search(r"\d", text)` short-circuits: if no digits, skip the import entirely
- `inflect` is an optional dependency (guarded by try/except)
- The lambda inside `re.sub` is called once per match — `m.group()` is the matched digit string
- `\b\d+\b` matches whole numbers only (`123` in `abc123` won't match because `\b` checks for word boundary)
- On failure the digits are silently kept — better than crashing a batch job over one malformed number

### `collapse_whitespace()`

```python
def collapse_whitespace(text: str) -> str:    # cleaners.py:11
    return re.sub(r"\s+", " ", text).strip()
```

`\s+` matches any run of whitespace characters (space, tab, newline, carriage return). This fixes transcripts scraped from HTML or PDFs that contain `\t`, `\r\n`, or multiple consecutive spaces.

---

## Module 4 — Grapheme-to-Phoneme (G2P)

### What G2P is and why TTS needs it

A grapheme is a written character. A phoneme is a unit of sound. G2P is the mapping from spelling to pronunciation.

The acoustic model processes **sound units** (phonemes), not letters. The word "read" has one spelling but two pronunciations ("reed" vs "red" depending on tense). "gh" in "ghost", "through", "cough", "hiccough" maps to four different sounds. Without G2P, the model would need to learn these rules from scratch — and would need vastly more training data.

For Vietnamese, every syllable has exactly one tone and one consonant-vowel combination. G2P is more deterministic than English but still requires handling:
- Tone diacritics (6 tones)
- Compound vowels (ươ, iê, uâ)
- Multi-character consonants (ng, nh, ch, tr, ph)

### IPA: the phoneme representation we use

IPA (International Phonetic Alphabet) is a standardized symbol set where each symbol maps to exactly one sound. Our project uses IPA as the phoneme format (`use_ipa: true` in `params.yaml`). Example:

| Word | Spelling | IPA |
|---|---|---|
| "cat" | c-a-t | k æ t |
| "phone" | p-h-o-n-e | f oʊ n |
| "mèo" | m-è-o | mɛw˨˩ |

---

## Module 5 — `BasePhonemizer` ABC

### The interface contract

```python
class BasePhonemizer(ABC):              # base_phonemizer.py:6

    @abstractmethod
    def phonemize(self, text: str) -> list[str]:
        ...

    @abstractmethod
    def get_symbol_set(self) -> set[str]:
        ...

    def phonemize_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.phonemize(t) for t in texts]    # base_phonemizer.py:37

    def validate_coverage(self, phoneme_lists, min_count=10) -> dict[str, int]:
        from collections import Counter
        counts = Counter(p for plist in phoneme_lists for p in plist)
        return {p: counts[p] for p in self.get_symbol_set() if counts.get(p, 0) < min_count}
```

This ABC enforces that every phonemizer in the system:
1. Converts a single string to a list of phoneme tokens (`phonemize`)
2. Exposes its complete vocabulary (`get_symbol_set`) — needed to build the `Tokenizer`

The concrete methods `phonemize_batch` and `validate_coverage` provide default implementations that work for any subclass. `phonemize_batch` can be overridden for batched efficiency (as `EspeakPhonemizer` does).

### Why `get_symbol_set()` is necessary

The `Tokenizer` must be built from a **complete** phoneme vocabulary before training starts. If a symbol not in the vocabulary appears at inference time, it maps to `PAD` (ID 0) silently — the model ignores it. Missing symbols that occur frequently means the model was never trained to handle them.

`get_symbol_set()` is the contract that lets the Tokenizer ask "what phonemes can you produce?" and build a closed vocabulary.

---

## Module 6 — `EspeakPhonemizer`

### eSpeak-NG: what it is

eSpeak-NG is a compact open-source text-to-speech engine that includes a built-in G2P system for 50+ languages. We use only its G2P component — not its synthesis — via the `phonemizer` Python library.

### Construction and backend

```python
class EspeakPhonemizer(BasePhonemizer):        # espeak_phonemizer.py:13

    def __init__(self, language: str = "vi", with_stress: bool = True) -> None:
        self.language = language
        self.with_stress = with_stress
        self._backend = self._build_backend()   # eagerly built (not lazy)
        self._symbol_set: set[str] = set()

    def _build_backend(self) -> object:
        try:
            from phonemizer.backend import EspeakBackend
            return EspeakBackend(
                language=self.language,
                with_stress=self.with_stress,
                language_switch="remove-flags",  # ← removes language-switch markers
            )
        except ImportError as e:
            raise ImportError("Install phonemizer: pip install phonemizer") from e
```

`language_switch="remove-flags"` controls what eSpeak does when it encounters words from other languages: "remove-flags" strips the `(en)` markers it would otherwise emit. This keeps our phoneme stream clean.

### Single-text phonemization

```python
def phonemize(self, text: str) -> list[str]:         # espeak_phonemizer.py:37
    if not text.strip():
        return []
    result = self._backend.phonemize([text], separator=_Separator())[0]
    tokens = [t for t in result.split() if t]
    self._symbol_set.update(tokens)                  # grow vocabulary as we see new symbols
    return tokens
```

The `_Separator` class controls how eSpeak delimits phones and words in its output:

```python
class _Separator:           # espeak_phonemizer.py:61
    phone = " "             # separate each phone with a space
    word = " "              # separate words with a space
    syllable = ""           # no syllable delimiter
```

This ensures tokens split cleanly on whitespace. Example:

```
Input:  "con mèo"
Output: "k ɔ n m ɛ w"   → ["k", "ɔ", "n", "m", "ɛ", "w"]
```

### Batched phonemization

```python
def phonemize_batch(self, texts: list[str]) -> list[list[str]]:    # espeak_phonemizer.py:45
    from phonemizer.separator import Separator
    results = self._backend.phonemize(
        texts,
        separator=Separator(phone=" ", word="| ")   # word boundary marked with |
    )
    token_lists = []
    for r in results:
        tokens = [t for t in r.replace("|", "").split() if t]
        self._symbol_set.update(tokens)
        token_lists.append(tokens)
    return token_lists
```

Calling `phonemize()` N times starts eSpeak N times. `phonemize_batch()` passes all texts in one call — much faster for large corpora. The `|` word-boundary markers are stripped before splitting.

---

## Module 7 — `ViPhonemizer`

### Vietnamese phonology basics

Vietnamese is a tonal, monosyllabic language. Each syllable has:
- An optional initial consonant (b, c, d, đ, g, h, k, l, m, n, ng, nh, ...)
- A vowel nucleus (a, ă, â, e, ê, i, o, ô, ơ, u, ư, y, or diphthongs)
- An optional final consonant (n, ng, m, p, t, k, ...)
- A tone (6 tones encoded as diacritics on the vowel)

### The 6 Vietnamese tones

```python
_TONE_MARKS = {                          # vi_phonemizer.py:22
    "level":   "˧",       # ngang  — flat mid tone:    a
    "falling": "˨˩",      # huyền  — low falling:      à
    "rising":  "˧˥",      # sắc    — high rising:      á
    "dipping": "˧˩˧",     # hỏi    — mid dipping:      ả
    "broken":  "˧ʔ˥",     # ngã    — high broken:      ã
    "checked": "˨˩ʔ",     # nặng   — low checked:      ạ
}
```

The IPA tone symbols (˧, ˥, ˨, ʔ) use the Chao tone letter system — numbers 1–5 representing low→high pitch.

### Tone detection from diacritics

```python
def _detect_tone(syllable: str) -> str:         # vi_phonemizer.py:32
    nfc = unicodedata.normalize("NFC", syllable)
    for char in nfc:
        name = unicodedata.name(char, "")
        if "GRAVE" in name:   return _TONE_MARKS["falling"]  # à → ˨˩
        if "ACUTE" in name:   return _TONE_MARKS["rising"]   # á → ˧˥
        if "HOOK ABOVE" in name: return _TONE_MARKS["dipping"]  # ả → ˧˩˧
        if "TILDE" in name:   return _TONE_MARKS["broken"]   # ã → ˧ʔ˥
        if "DOT BELOW" in name:  return _TONE_MARKS["checked"]  # ạ → ˨˩ʔ
    return _TONE_MARKS["level"]   # no mark → ngang (flat)
```

`unicodedata.name()` returns the official Unicode name of a code point:
```python
unicodedata.name("à")   # "LATIN SMALL LETTER A WITH GRAVE"
unicodedata.name("á")   # "LATIN SMALL LETTER A WITH ACUTE"
```

In NFC form, composed characters like `à` carry the tone mark in their name. The function scans each character in the syllable looking for the first marked vowel.

### Word phonemization

```python
def _phonemize_word(self, word: str) -> list[str]:    # vi_phonemizer.py:80
    word = unicodedata.normalize("NFC", word.lower().strip())
    word = re.sub(r"[^\w\s]", "", word)    # strip punctuation
    if not word:
        return []
    tone = _detect_tone(word)              # extract tone first (NFC form)
    # Then strip tone diacritics to get the base consonant-vowel form
    word_stripped = unicodedata.normalize("NFD", word)
    word_stripped = "".join(c for c in word_stripped if unicodedata.category(c) != "Mn")
    return [word_stripped, tone]
```

Trace through `"mèo"`:
```
NFC form: "mèo"          (m + è + o, precomposed)
_detect_tone("mèo"):
  scan "m" → name = "LATIN SMALL LETTER M" → no tone mark
  scan "è" → name = "LATIN SMALL LETTER E WITH GRAVE" → GRAVE found → "˨˩"
  return "˨˩"

NFD form: "mèo"    (m + e + combining grave + o)
filter Mn: drop ̀    → "meo"

Result: ["meo", "˨˩"]
```

### Lazy fallback to eSpeak

```python
def _get_fallback(self) -> BasePhonemizer:        # vi_phonemizer.py:62
    if self._fallback is None:
        from tts_pipeline.text.phonemizers.espeak_phonemizer import EspeakPhonemizer
        self._fallback = EspeakPhonemizer(language=self._fallback_language)
    return self._fallback
```

The fallback is only instantiated when needed (lazy init pattern from Course 1). This avoids importing `phonemizer` and starting eSpeak-NG if the rule-based path handles all words.

---

## Module 8 — Special Tokens and Symbol Lists

### The four special tokens

```python
PAD   = "_PAD_"    # symbols.py:4  — padding, ID=0 by convention
BOS   = "_BOS_"    #               — beginning of sequence
EOS   = "_EOS_"    #               — end of sequence
BLANK = "_BLANK_"  #               — CTC blank / inter-phoneme separator
```

These tokens appear in the phoneme ID sequence but never as actual phonemes from the G2P system. Their names use `_` delimiters to avoid colliding with real phoneme symbols.

### `build_symbol_list()`

```python
def build_symbol_list(phoneme_inventory: list[str], add_blank: bool = True) -> list[str]:
    # symbols.py:12
    phonemes = sorted(set(phoneme_inventory) - set(SPECIAL_TOKENS))
    symbols = list(SPECIAL_TOKENS) + phonemes
    if add_blank and BLANK not in symbols:
        symbols.append(BLANK)
    return symbols
```

The ordering is: `[_PAD_, _BOS_, _EOS_, _BLANK_] + sorted(phonemes)`.

Why sorted? Determinism. Two runs with the same phoneme inventory must produce the same symbol→ID mapping. If the input list order depended on dict iteration or set order, the mapping would differ between Python versions.

Why specials first? `PAD` at ID 0 is a convention from NLP: padding with zeros is common, and `padding_idx=0` in `nn.Embedding` (Course 4) zeros out the gradient for padded positions.

### Building the vocabulary in practice

```python
# During data processing:
phonemizer = ViPhonemizer()
all_phonemes = []
for text in corpus:
    all_phonemes.extend(phonemizer.phonemize(text))

tokenizer = Tokenizer.from_inventory(phonemizer.get_symbol_set(), add_blank=True)
tokenizer.save(Path("model_bundle/phoneme_vocab.json"))
```

The vocabulary is built once from the training corpus, saved to JSON, and loaded at every subsequent run (training, evaluation, serving). If the vocabulary changes, the model must be retrained from scratch.

---

## Module 9 — The Tokenizer

### `encode()` in detail

```python
def encode(self, phonemes: list[str], add_bos_eos: bool = True) -> list[int]:
    # tokenizer.py:44
    ids = [self._sym2id.get(p, self.pad_id) for p in phonemes]   # 1. symbol → ID
    if self._add_blank:
        ids = self._intersperse(ids, self.blank_id)               # 2. insert BLANK
    if add_bos_eos:
        ids = [self.bos_id] + ids + [self.eos_id]                 # 3. wrap with BOS/EOS
    return ids
```

Step 1 — Unknown symbols map to `pad_id` (0). This silently handles out-of-vocabulary phonemes at runtime rather than crashing.

Step 2 — `_intersperse`:

```python
@staticmethod
def _intersperse(ids: list[int], blank_id: int) -> list[int]:   # tokenizer.py:74
    result = [blank_id] * (2 * len(ids) + 1)
    result[1::2] = ids
    return result
```

For `ids = [4, 17, 8]` and `blank_id = 3`:
```
result initialized: [3, 3, 3, 3, 3, 3, 3]   (length = 2*3+1 = 7)
result[1::2] = [4, 17, 8]
result final:       [3, 4, 3, 17, 3, 8, 3]
```

Every phoneme is surrounded by BLANK tokens. This is the CTC-style blank interspersion: it allows the duration predictor to learn how long each phoneme lasts (including pauses between phonemes).

Step 3 — After interspersion, wrap with BOS and EOS:
```
[BOS, BLANK, p1, BLANK, p2, BLANK, p3, BLANK, EOS]
```

### Full trace: `"mèo"` to IDs

```
text: "mèo"
  ↓ clean_text()
cleaned: "mèo"          (no changes needed)
  ↓ ViPhonemizer.phonemize()
phonemes: ["meo", "˨˩"]
  ↓ Tokenizer.encode()
  sym2id: {"_PAD_": 0, "_BOS_": 1, "_EOS_": 2, "_BLANK_": 3, "˨˩": 4, "meo": 17, ...}
  raw ids: [17, 4]
  interspersed: [3, 17, 3, 4, 3]
  with BOS/EOS: [1, 3, 17, 3, 4, 3, 2]
final IDs: [1, 3, 17, 3, 4, 3, 2]
```

### `decode()`

```python
def decode(self, ids: list[int]) -> list[str]:    # tokenizer.py:53
    return [self._id2sym.get(i, PAD) for i in ids]
```

The inverse map. Unknown IDs return `PAD`. Used for debugging — when you see a model output you can decode the IDs back to phoneme strings.

### Persistence: `save()` / `load()`

```python
def save(self, path: Path) -> None:               # tokenizer.py:56
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"symbols": self._symbols, "add_blank": self._add_blank}, f, ensure_ascii=False)

@classmethod
def load(cls, path: Path) -> "Tokenizer":         # tokenizer.py:61
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return cls(symbols=data["symbols"], add_blank=data.get("add_blank", True))
```

`ensure_ascii=False` is critical — without it, `json.dump` would escape all non-ASCII characters (`"˨˩"` → `"˨˩"`). The saved file would still deserialize correctly, but it would be unreadable for debugging.

`data.get("add_blank", True)` uses a default to maintain backward compatibility with vocabulary files that predate the `add_blank` field.

---

## Module 10 — Transcript Cleaning at the Dataset Level

### `clean_metadata()`

```python
def clean_metadata(
    df: pd.DataFrame,
    language: str = "vi",
    min_chars: int = 2,
    max_chars: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:          # transcript_cleaner.py:12
    df = df.copy()
    df["text_clean"] = df["text"].apply(lambda t: clean_text(str(t), language=language))

    mask_short = df["text_clean"].str.len() < min_chars
    mask_long  = df["text_clean"].str.len() > max_chars
    mask_empty = df["text_clean"].str.strip() == ""

    rejected_mask = mask_short | mask_long | mask_empty
    return df[~rejected_mask].copy(), df[rejected_mask].copy()
```

This is where `clean_text()` is applied to an entire metadata DataFrame. The function returns two DataFrames: accepted and rejected — so the caller can log or inspect what was dropped without losing information.

Why `df.copy()` at the start? Pandas `apply` can mutate the original DataFrame if it returns a view. The copy prevents the caller's DataFrame from being modified (a common Pandas gotcha).

Why `str(t)`? If a cell is `NaN` or `float`, `clean_text` would fail. `str(NaN)` → `"nan"` which is then rejected by `min_chars`.

### Length thresholds

| Threshold | Value | Rationale |
|---|---|---|
| `min_chars=2` | 2 | Single-character utterances are noise (e.g., column headers) |
| `max_chars=300` | 300 | Very long utterances don't fit in GPU memory during training without truncation |

These thresholds interact with `params.yaml`'s `min_duration=0.5` and `max_duration=15.0`. The audio-based filters in `audio_processor.py` handle duration; `clean_metadata` handles text length. Both must pass.

---

## Module 11 — Common Mistakes

### Mistake 1: Normalizing after G2P

```python
# BAD — phonemize first, then normalize?
phonemes = phonemizer.phonemize(raw_text)
clean = clean_text(raw_text)   # too late

# GOOD — always clean before phonemize
clean = clean_text(raw_text)
phonemes = phonemizer.phonemize(clean)
```

If raw text contains `"3 con mèo"`, the phonemizer sees the digit `3` and either crashes or produces undefined output. The cleaner expands it to `"ba"` first.

### Mistake 2: Building a new Tokenizer at inference time

```python
# BAD — new tokenizer at serving time has different symbol ordering
phonemizer = EspeakPhonemizer()
phonemizer.phonemize(text)
tokenizer = Tokenizer.from_inventory(phonemizer.get_symbol_set())  # wrong order!

# GOOD — load the saved tokenizer from the model bundle
tokenizer = Tokenizer.load(Path("model_bundle/phoneme_vocab.json"))
```

The model was trained with specific integer IDs. If the tokenizer is rebuilt, the ordering of `sorted(phoneme_inventory)` might differ slightly (e.g., from a different corpus size), producing wrong IDs.

### Mistake 3: Forgetting `ensure_ascii=False`

```python
# BAD — IPA symbols become escape sequences in the file
json.dump({"symbols": symbols}, f)
# File: {"symbols": ["_PAD_", "_BOS_", ..., "˨˩", ...]}

# GOOD — save readable IPA symbols directly
json.dump({"symbols": symbols}, f, ensure_ascii=False)
# File: {"symbols": ["_PAD_", "_BOS_", ..., "˨˩", ...]}
```

Both serialize/deserialize correctly. The second is debuggable by a human.

### Mistake 4: Not stripping whitespace before phonemize

```python
# BAD — leading/trailing spaces produce empty tokens
phonemizer.phonemize("  mèo  ")
# eSpeak may emit: ["", "m", "ɛ", "w", ""]

# GOOD — cleaners handle this
clean = clean_text("  mèo  ")   # "mèo"
phonemizer.phonemize(clean)
```

### Mistake 5: Vocabulary mismatch between train and serve

The `Tokenizer` must be loaded from the same JSON that was used during training. A common mistake is to create a language-specific default vocabulary hardcoded in the serving code, then train with a corpus that adds extra phonemes. The model's embedding matrix has `vocab_size` rows — if the serving tokenizer has more or fewer symbols, ID assignments drift.

---

## Module 12 — Mini Practice Project: `phonokit`

An independent mini-project that builds the complete text→ID pipeline for English — a custom regex-based G2P, a symbol list, and a tokenizer. No TTS pipeline code.

**Goal:** Given English text with numbers and punctuation, produce integer IDs suitable for a character-level or phoneme-level model. The focus is on the pipeline architecture, not phonetic accuracy.

### Project structure

```
phonokit/
├── cleaners.py     # clean_text() for English
├── g2p.py          # SimpleG2P — rule-based grapheme-to-phoneme
├── tokenizer.py    # Tokenizer (same design as project's Tokenizer)
└── pipeline.py     # end-to-end: text → IDs → back to text
```

### `cleaners.py`

```python
import re
import unicodedata


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))


def expand_numbers(text: str) -> str:
    """Expand single-digit numbers to English words."""
    digits = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    return re.sub(r"\b(\d)\b", lambda m: digits[m.group()], text)


def normalize_punctuation(text: str) -> str:
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"[!?]+", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_punctuation(text)
    text = expand_numbers(text)
    return re.sub(r"\s+", " ", text).strip()
```

### `g2p.py`

```python
import re

# Simplified English consonant digraphs and their phoneme tokens
DIGRAPH_MAP = {
    "ch": "tʃ", "sh": "ʃ", "th": "θ", "ph": "f",
    "wh": "w", "gh": "g", "ng": "ŋ", "ck": "k",
}

# Silent-e rule: CVCe → long vowel (very simplified)
LONG_VOWELS = {"a": "eɪ", "e": "iː", "i": "aɪ", "o": "oʊ", "u": "juː"}
SHORT_VOWELS = {"a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɒ", "u": "ʌ"}

# Consonant → phoneme (1:1 where possible)
CONSONANT_MAP = {
    "b": "b", "c": "k", "d": "d", "f": "f", "g": "g", "h": "h",
    "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
    "r": "r", "s": "s", "t": "t", "v": "v", "w": "w", "x": "ks",
    "y": "j", "z": "z",
}


class SimpleG2P:
    """Rule-based G2P for English. Simplified — not linguistically complete."""

    def phonemize(self, text: str) -> list[str]:
        tokens = []
        words = text.lower().split()
        for word in words:
            tokens.extend(self._word_to_phonemes(word))
        return tokens

    def _word_to_phonemes(self, word: str) -> list[str]:
        # Strip non-alpha
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return []
        
        phonemes = []
        i = 0
        silent_e = word.endswith("e") and len(word) > 2
        
        while i < len(word):
            # Check digraph first
            if i + 1 < len(word) and word[i:i+2] in DIGRAPH_MAP:
                phonemes.append(DIGRAPH_MAP[word[i:i+2]])
                i += 2
            elif word[i] in "aeiou":
                vowel = word[i]
                # Use long vowel if silent-e applies and this is the last vowel
                if silent_e and i == len(word) - 2:
                    phonemes.append(LONG_VOWELS[vowel])
                else:
                    phonemes.append(SHORT_VOWELS[vowel])
                i += 1
            elif word[i] in CONSONANT_MAP:
                phonemes.append(CONSONANT_MAP[word[i]])
                i += 1
            else:
                i += 1  # skip unknown characters
        
        return phonemes

    def get_symbol_set(self) -> set[str]:
        return set(DIGRAPH_MAP.values()) | set(LONG_VOWELS.values()) | \
               set(SHORT_VOWELS.values()) | set(CONSONANT_MAP.values())
```

### `tokenizer.py`

```python
import json
from pathlib import Path

PAD   = "_PAD_"
BOS   = "_BOS_"
EOS   = "_EOS_"
BLANK = "_BLANK_"
SPECIAL_TOKENS = [PAD, BOS, EOS, BLANK]


def build_symbol_list(phoneme_inventory: set[str]) -> list[str]:
    phonemes = sorted(phoneme_inventory - set(SPECIAL_TOKENS))
    return SPECIAL_TOKENS + phonemes


class Tokenizer:
    def __init__(self, symbols: list[str], add_blank: bool = True):
        self._symbols = symbols
        self._add_blank = add_blank
        self._sym2id = {s: i for i, s in enumerate(symbols)}
        self._id2sym = {i: s for i, s in enumerate(symbols)}

    @property
    def vocab_size(self) -> int:
        return len(self._symbols)

    def encode(self, phonemes: list[str]) -> list[int]:
        ids = [self._sym2id.get(p, 0) for p in phonemes]    # 0 = PAD for unknowns
        if self._add_blank:
            blank_id = self._sym2id[BLANK]
            result = [blank_id] * (2 * len(ids) + 1)
            result[1::2] = ids
            ids = result
        return [self._sym2id[BOS]] + ids + [self._sym2id[EOS]]

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id2sym.get(i, PAD) for i in ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"symbols": self._symbols, "add_blank": self._add_blank}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(symbols=data["symbols"], add_blank=data.get("add_blank", True))
```

### `pipeline.py`

```python
from pathlib import Path
from cleaners import clean_text
from g2p import SimpleG2P
from tokenizer import Tokenizer, build_symbol_list


def build_tokenizer(corpus: list[str]) -> Tokenizer:
    """Build tokenizer by phonemizing the full corpus first."""
    g2p = SimpleG2P()
    all_phonemes = set()
    for text in corpus:
        all_phonemes.update(g2p.phonemize(clean_text(text)))
    symbols = build_symbol_list(all_phonemes)
    return Tokenizer(symbols=symbols, add_blank=True)


def text_to_ids(text: str, g2p: SimpleG2P, tokenizer: Tokenizer) -> list[int]:
    clean = clean_text(text)
    phonemes = g2p.phonemize(clean)
    return tokenizer.encode(phonemes)


def main():
    corpus = [
        "The cat sat on the mat.",
        "She sells seashells by the seashore.",
        "I have 3 apples and 1 orange.",
        "The phone rang at midnight.",
    ]

    # Build and save tokenizer
    tokenizer = build_tokenizer(corpus)
    tokenizer.save(Path("vocab.json"))
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Symbols: {tokenizer._symbols[:10]}...")

    # Process one sentence
    g2p = SimpleG2P()
    text = "The cat sat."
    ids = text_to_ids(text, g2p, tokenizer)
    decoded = tokenizer.decode(ids)
    
    print(f"\nInput:   {text!r}")
    print(f"Cleaned: {clean_text(text)!r}")
    print(f"Phonemes:{g2p.phonemize(clean_text(text))}")
    print(f"IDs:     {ids}")
    print(f"Decoded: {decoded}")

    # Round-trip test
    reloaded = Tokenizer.load(Path("vocab.json"))
    ids2 = text_to_ids(text, g2p, reloaded)
    assert ids == ids2, "Round-trip mismatch!"
    print("\nRound-trip: PASS")


if __name__ == "__main__":
    main()
```

### Running it

```bash
pip install phonokit  # just pure stdlib + json, no extra deps
cd phonokit
python pipeline.py
```

Expected output:

```
Vocabulary size: 47
Symbols: ['_PAD_', '_BOS_', '_EOS_', '_BLANK_', 'æ', 'b', 'd', ...]...

Input:   'The cat sat.'
Cleaned: 'The cat sat.'
Phonemes: ['θ', 'ɛ', 'k', 'æ', 't', 's', 'æ', 't']
IDs:     [1, 3, 22, 3, 4, 3, 7, 3, 5, 3, 22, 3, 19, 3, 5, 3, 22, 3, 19, 3, 2]
Decoded: ['_BOS_', '_BLANK_', 'θ', '_BLANK_', 'ɛ', '_BLANK_', ...]

Round-trip: PASS
```

### Exercises

1. **Add multi-digit number expansion** — `"42"` should expand to `"forty two"`. Implement `expand_numbers` using a loop that decomposes the number digit-by-digit and handles tens (10-19 as irregular).
2. **Test `ensure_ascii`** — save with `ensure_ascii=True` and `ensure_ascii=False`. Open both files in a text editor. Verify both load correctly with `Tokenizer.load()`.
3. **Add a new G2P rule** — `"qu"` in English sounds like `"kw"`. Add it to `DIGRAPH_MAP` and verify `"queen"` → `["kw", "iː", "n"]`.
4. **Measure vocabulary coverage** — build the tokenizer from 3 of the 4 corpus sentences, then phonemize the 4th. Count how many phonemes fall back to PAD (ID 0). Observe how vocabulary coverage degrades on unseen data.
5. **Intersperse by hand** — without running the code, compute `_intersperse([5, 12, 8], blank_id=3)`. Then verify with Python.

---

## Key Takeaways

- `clean_text()` always runs before G2P: digits must be expanded, diacritics normalized, control characters removed before phonemization.
- NFC normalization is the canonical form for string comparison; NFD is used to manipulate diacritics as separate combining marks that can be filtered by category `Mn`.
- `BasePhonemizer` is an ABC with two abstract methods: `phonemize()` (text→phoneme list) and `get_symbol_set()` (the closed vocabulary needed to build the Tokenizer).
- `_intersperse` inserts BLANK between every phoneme: `[p1, p2, p3]` → `[B, p1, B, p2, B, p3, B]`. This gives the duration predictor space to model inter-phoneme pauses.
- The Tokenizer vocabulary must be built once from the full training corpus and saved. Loading a different tokenizer at serving time produces wrong IDs and broken output.
- `json.dump(..., ensure_ascii=False)` preserves IPA symbols as human-readable characters in the saved vocabulary file.