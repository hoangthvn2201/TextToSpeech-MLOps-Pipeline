# Course 1: Python for ML Engineers

**Prerequisite:** Basic Python (functions, classes, loops, imports). No ML knowledge required.

**Goal:** After this course you can read every `.py` file in this project without hitting unfamiliar Python syntax.

## Module 1 — Type Hints & Static Typing

### 1.1 Why type hints exist in ML code

ML codebases break in silent, expensive ways: a tensor of shape `(batch, time, mel)` gets passed where `(time, mel)` is expected. You find out at hour 3 of a training run. Type hints let `mypy` catch this at edit time, before you run anything.

This project runs `mypy --strict` on every commit (see `pyproject.toml:56-60`).

### 1.2 Basic annotations

```python
# Bad — reader has no idea what these are
def clean(text, language):
    ...

# Good — immediately obvious
def clean(text: str, language: str = "vi") -> str:
    ...
```

### 1.3 `Optional` vs `X | None`

```python
from __future__ import annotations   # enables new syntax on Python 3.10+

# Old style (still common in libraries)
from typing import Optional

def load(path: Optional[str] = None) -> str: ...

# Modern style — what this project uses (Python 3.10+)
def load(path: str | None = None) -> str: ...
```

In this project:

```python
# config.py:53
resume_from_checkpoint: str | None = None
```

### 1.4 `list`, `dict`, `tuple` — built-in generics

Since Python 3.9 you use lowercase directly (no `from typing import List`):

```python
def phonemize_batch(self, texts: list[str]) -> list[list[str]]: ...

def compute_stats(mels: dict[str, np.ndarray]) -> dict[str, float]: ...

def get_split(stem: str) -> tuple[str, float]: ...
```

In this project: `audio_processor.py` — `process_batch` returns `list[Path]`.

### 1.5 `TypeVar` — generics for your own functions

Use when a function returns the same type it receives:

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

first([1, 2, 3])        # inferred as int
first(["a", "b"])       # inferred as str
```

### 1.6 `Protocol` — structural typing (duck typing with type safety)

`Protocol` says "anything with these methods works" without requiring inheritance. This is how `BasePhonemizer` could have been written if it needed more flexibility:

```python
from typing import Protocol

class Phonemizer(Protocol):
    def phonemize(self, text: str) -> list[str]: ...
    def get_symbol_set(self) -> set[str]: ...

def build_vocab(p: Phonemizer) -> set[str]:
    return p.get_symbol_set()
```

Any class with those methods passes, even if it doesn't inherit `Phonemizer`. Contrast this with `ABC` (Module 2) which requires explicit subclassing.

### 1.7 `Literal` — restrict values to a fixed set

```python
from typing import Literal

Split = Literal["train", "val", "test"]

def assign_split(stem: str) -> Split:
    ...
```

`mypy` will error if you pass `"validation"` where `Split` is expected.

### 1.8 `TypedDict` — typed dictionaries

When a dict has a fixed known schema:

```python
from typing import TypedDict

class SNRStats(TypedDict):
    mean: float
    median: float
    p10: float
    p90: float
```

In this project: the scorecard dict in `validator.py` is a plain dict — using `TypedDict` here would make it cleaner.

### 1.9 `from __future__ import annotations`

Every file in this project starts with this line. It makes all annotations lazy (strings), so you can use forward references and newer syntax without Python version issues:

```python
from __future__ import annotations   # must be line 1

class Node:
    def clone(self) -> Node: ...   # works even if Node isn't fully defined yet
```

## Module 2 — Abstract Base Classes (`abc`)

### 2.1 The problem ABC solves

The project supports multiple languages (Vietnamese, English, etc.). Each needs a different phonemizer. The rest of the system — feature extraction, training, serving — should not care which language it is using.

The solution: define a contract that all phonemizers must fulfill. `ABC` enforces this at instantiation time.

### 2.2 How ABC works

```python
from abc import ABC, abstractmethod

class BasePhonemizer(ABC):

    @abstractmethod
    def phonemize(self, text: str) -> list[str]:
        ...

    @abstractmethod
    def get_symbol_set(self) -> set[str]:
        ...
```

If you try to instantiate `BasePhonemizer()` directly — Python raises `TypeError`. If you subclass it but forget `phonemize` — also `TypeError`. The error happens early, not buried inside a training loop.

In this project: `base_phonemizer.py`

### 2.3 Concrete implementations

```python
class EspeakPhonemizer(BasePhonemizer):
    def phonemize(self, text: str) -> list[str]:
        # real implementation
        ...

    def get_symbol_set(self) -> set[str]:
        return self._symbol_set | _SPECIAL_TOKENS


class ViPhonemizer(BasePhonemizer):
    def phonemize(self, text: str) -> list[str]:
        # Vietnamese-specific rules
        ...

    def get_symbol_set(self) -> set[str]:
        return self._symbol_set | _SPECIAL_TOKENS
```

In this project: `espeak_phonemizer.py`, `vi_phonemizer.py`

### 2.4 Concrete methods on an abstract class

`ABC` allows non-abstract methods. Use them for shared logic that all subclasses inherit for free:

```python
class BasePhonemizer(ABC):

    @abstractmethod
    def phonemize(self, text: str) -> list[str]: ...

    # Not abstract — default implementation all subclasses get
    def phonemize_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.phonemize(t) for t in texts]

    def validate_coverage(self, phoneme_lists: list[list[str]], min_count: int = 10) -> dict[str, int]:
        from collections import Counter
        counts: Counter[str] = Counter(p for plist in phoneme_lists for p in plist)
        return {p: counts[p] for p in self.get_symbol_set() if counts.get(p, 0) < min_count}
```

`EspeakPhonemizer` overrides `phonemize_batch` with a more efficient batched call. `ViPhonemizer` uses the default. Callers never know or care.

### 2.5 ABC vs Protocol — when to use which

| Feature                        | ABC |         Protocol |
| ------------------------------ | --: | ---------------: |
| Requires subclassing           | Yes |               No |
| Enforces at `__init__`         | Yes | No (only `mypy`) |
| Allows shared code             | Yes |               No |
| Works with third-party classes |  No |              Yes |

Rule of thumb: If you own both the base and the implementations, use `ABC`. If you want to accept third-party objects that happen to have the right methods, use `Protocol`.

## Module 3 — Pydantic v2

### 3.1 What Pydantic does

Pydantic takes a plain Python class and turns it into a validated, typed data container. You put in a dict from a YAML file or an API request, you get back a structured object with guaranteed types and ranges. Invalid input raises a descriptive error immediately.

### 3.2 Basic model

```python
from pydantic import BaseModel

class DataConfig(BaseModel):
    language: str = "vi"
    sample_rate: int = 22050
    n_mels: int = 80
    min_duration: float = 0.5
    max_duration: float = 15.0
```

Creating an instance validates types automatically:

```python
cfg = DataConfig(sample_rate="22050")  # "22050" → coerced to int 22050
cfg = DataConfig(sample_rate="hello")   # ValidationError raised immediately
```

In this project: `config.py:7-37`

### 3.3 `field_validator` — custom validation logic

```python
from pydantic import BaseModel, field_validator

class DataConfig(BaseModel):
    dataset_grade_threshold: str = "C"

    @field_validator("dataset_grade_threshold")
    @classmethod
    def valid_grade(cls, v: str) -> str:
        if v not in ("A", "B", "C", "D", "F"):
            raise ValueError(f"Invalid grade threshold: {v}")
        return v
```

If you pass `"Z"`, you get:

```text
pydantic_core.ValidationError: 1 validation error for DataConfig
dataset_grade_threshold
  Value error, Invalid grade threshold: Z
```

In this project: `config.py:27-31`

### 3.4 Nested models

```python
class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    text: TextConfig = TextConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
```

The entire pipeline is one object. You access `cfg.training.learning_rate`, `cfg.data.sample_rate`. Nested validation: if `TrainingConfig` is invalid, `PipelineConfig` fails too.

In this project: `config.py:87-93`

### 3.5 `dict[str, float]` as a field type

```python
class EvaluationConfig(BaseModel):
    absolute_floor: dict[str, float] = {
        "utmos_mean": 3.0,
        "cer": 0.20,
        "stoi_mean": 0.60,
        "rtf_p95": 0.20,
    }
```

Pydantic validates every value in the dict is a float. This is the evaluation gate config — wrong types here would silently pass wrong thresholds into the gating logic.

In this project: `config.py:62-78`

### 3.6 Constructing from a dict (common pattern with YAML)

```python
import yaml
from pathlib import Path

raw = yaml.safe_load(Path("params.yaml").read_text())
cfg = DataConfig(**raw["data"])  # unpacks dict into keyword args
```

In this project: `config.py:96-106` — the `from_hydra()` and `load_params()` functions do exactly this.

## Module 4 — `pathlib.Path`

### 4.1 Why `pathlib` instead of `os.path`

`os.path` works with strings. `pathlib.Path` is an object — it knows it's a path, it has methods, and it composes cleanly:

```python
# os.path — string manipulation, easy to get wrong
import os
out = os.path.join(base_dir, "data", "processed", stem + ".npy")

# pathlib — clean, readable, cross-platform
from pathlib import Path
out = Path(base_dir) / "data" / "processed" / f"{stem}.npy"
```

### 4.2 Common operations used in this project

```python
path = Path("data/processed/features/hello.npy")

path.stem          # "hello"
path.suffix        # ".npy"
path.name          # "hello.npy"
path.parent        # Path("data/processed/features")

path.exists()      # bool
path.is_file()     # bool
path.is_dir()      # bool

path.mkdir(parents=True, exist_ok=True)   # create dir tree safely
path.parent.mkdir(parents=True, exist_ok=True)

# Reading / writing
text = path.read_text(encoding="utf-8")
path.write_text("content", encoding="utf-8")

# Globbing
audio_files = list(Path("data/raw").rglob("*.wav"))
```

In this project: `ingestion.py` — `audio_dir.rglob("*")`, `transcript_files[stem].read_text(encoding="utf-8")`.

### 4.3 `Path` in function signatures

Always annotate with `Path`, not `str`. If a caller passes a string, tell them to convert it. This prevents mixing path types deep in the call stack:

```python
# Bad
def extract_mel(audio_path: str) -> np.ndarray: ...

# Good
def extract_mel(audio_path: Path) -> np.ndarray: ...

# Accepting both (use sparingly, prefer Path)
from os import PathLike
def extract_mel(audio_path: Path | str) -> np.ndarray:
    audio_path = Path(audio_path)
    ...
```

## Module 5 — `concurrent.futures` — Parallel Processing

### 5.1 The problem: single-threaded data pipelines are too slow

Processing 10,000 audio files one at a time on one CPU core: slow. Each file is independent. This is a textbook case for parallelism.

### 5.2 `ProcessPoolExecutor` — the right tool for CPU-bound ML work

Python's GIL (Global Interpreter Lock) means threads can't run CPU-bound code in parallel. `ProcessPoolExecutor` spawns real OS processes — each with its own Python interpreter and its own GIL — so they truly run in parallel:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_batch(
    src_dst_pairs: list[tuple[Path, Path]],
    n_workers: int = 4,
) -> list[Path]:
    results: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(normalize_audio, src, dst): (src, dst)
            for src, dst in src_dst_pairs
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Failed to process %s: %s", src, exc)
    return results
```

In this project: `audio_processor.py:49-66`, `feature_extractor.py:81-97`

### 5.3 Anatomy of the pattern

| Part                                 | What it does                                                 |
| ------------------------------------ | ------------------------------------------------------------ |
| `ProcessPoolExecutor(max_workers=n)` | Creates a pool of `n` worker processes                       |
| `executor.submit(fn, *args)`         | Schedules `fn(*args)` to run in a worker, returns a `Future` |
| `futures = {future: metadata}`       | Dict maps `Future` → original args, for error reporting      |
| `as_completed(futures)`              | Yields futures in completion order (fastest first)           |
| `future.result()`                    | Gets return value — raises if the function raised            |
| `with` block                         | Calls `executor.shutdown(wait=True)` on exit — waits for all |

### 5.4 What gets pickled

`ProcessPoolExecutor` sends work to other processes via pickle. This means:

* Function must be importable (defined at module level, not a lambda)
* Arguments must be picklable (`Path` ✓, `numpy.ndarray` ✓, open file handles ✗)

```python
# Won't work — lambda can't be pickled
executor.submit(lambda x: x * 2, data)

# Works — module-level function
def double(x: int) -> int:
    return x * 2

executor.submit(double, data)
```

### 5.5 `ThreadPoolExecutor` — when to use it instead

Use `ThreadPoolExecutor` only for I/O-bound work (network calls, reading small files). Audio normalization and feature extraction are CPU-bound (FFT, convolution) — always `ProcessPoolExecutor` for those.

### 5.6 Error handling pattern

Notice the `try/except` inside `as_completed`. Without it, one bad file would leave the rest of the futures uncollected (the `with` block would wait for them). The pattern here: log the error, continue, report how many succeeded at the end.

## Module 6 — `pyproject.toml` and Packaging

### 6.1 What `pyproject.toml` replaces

Old Python projects had `setup.py`, `setup.cfg`, `requirements.txt`, `tox.ini`, `.flake8`, `mypy.ini` — all separate files. `pyproject.toml` is one file for all of it.

### 6.2 The `[project]` section

```toml
[project]
name = "tts-pipeline"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "pydantic>=2.7.0",
    ...
]
```

`dependencies` is what always gets installed with `pip install tts-pipeline`.

In this project: `pyproject.toml:1-30`

### 6.3 Optional dependency groups

```toml
[project.optional-dependencies]
train = [
    "lightning>=2.3.0",
    "torchcrepe>=0.0.23",
]
serve = [
    "fastapi>=0.111.0",
    "onnxruntime>=1.18.0",
]
dev = [
    "pytest>=8.2.0",
    "ruff>=0.4.7",
]
```

Install groups selectively:

```bash
pip install "tts-pipeline[train]"    # base + training deps
pip install "tts-pipeline[serve]"    # base + serving deps only (no GPU)
pip install "tts-pipeline[dev]"      # base + dev tools
pip install "tts-pipeline[train,dev]" # combine groups
```

The serving Docker image installs only `[serve]` — that's why it's 300 MB instead of 5 GB.

In this project: `pyproject.toml:32-50`

### 6.4 Entry points (`[project.scripts]`)

```toml
[project.scripts]
tts-data  = "tts_pipeline.data.ingestion:cli"
tts-train = "tts_pipeline.training.trainer:cli"
tts-eval  = "tts_pipeline.evaluation.reporter:cli"
tts-serve = "tts_pipeline.serving.app:cli"
```

After `pip install`, these become terminal commands. `tts-train` calls `cli()` in `tts_pipeline/training/trainer.py`. No more `python -m tts_pipeline.training.trainer` — just `tts-train`.

In this project: `pyproject.toml:52-55`

### 6.5 Tool configuration sections

All linter and test configs live in the same file:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]     # E501 = line too long (Black handles this)

[tool.mypy]
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["slow", "gpu"]
```

In this project: `pyproject.toml:62-90`

### 6.6 `setuptools.packages.find`

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

This tells setuptools: your importable packages live under `src/`. So `import tts_pipeline` resolves to `src/tts_pipeline/`, not the root. This prevents the project root from accidentally becoming importable.

## Module 7 — `lru_cache` and Lazy Initialization

### 7.1 Expensive objects that shouldn't be rebuilt repeatedly

The eSpeak backend takes ~500 ms to initialize. If `phonemize()` is called thousands of times, you don't want 500 ms overhead each time.

`lru_cache` is Python's built-in memoization decorator:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def get_espeak_backend(language: str) -> EspeakBackend:
    return EspeakBackend(language=language, with_stress=True)
```

First call for `"vi"`: builds the backend, caches it. Every subsequent call with `"vi"`: returns the cached instance instantly.

In this project: `espeak_phonemizer.py` — the `_build_backend` approach is used here; `lru_cache` is noted as an alternative.

### 7.2 Lazy initialization pattern (without `lru_cache`)

The `ViPhonemizer` uses a different pattern — initializing the fallback only when it's first needed:

```python
class ViPhonemizer(BasePhonemizer):
    def __init__(self) -> None:
        self._fallback: BasePhonemizer | None = None   # not built yet

    def _get_fallback(self) -> BasePhonemizer:
        if self._fallback is None:
            from tts_pipeline.text.phonemizers.espeak_phonemizer import EspeakPhonemizer
            self._fallback = EspeakPhonemizer(language="vi")
        return self._fallback
```

The import is also deferred inside the method — if eSpeak isn't installed, you won't get an `ImportError` just by importing `ViPhonemizer`.

In this project: `vi_phonemizer.py:38-44`

## Module 8 — Handling Missing Dependencies Gracefully

### 8.1 Optional imports — the pattern

Some features need heavy packages that aren't always installed. Use `try/except ImportError` to detect and give a clear error:

```python
def extract_f0(audio_path: Path, sample_rate: int = 22050) -> np.ndarray:
    try:
        import torchcrepe

        f0, _ = torchcrepe.predict(...)
        return f0

    except ImportError:
        # torchcrepe not installed — fall back to librosa
        import librosa
        f0, _, _ = librosa.pyin(...)
        return f0
```

In this project: `feature_extractor.py:39-57` — exact pattern used for CREPE vs PYIN fallback.

### 8.2 Raising with context — `raise X from Y`

When re-raising a caught exception with more context:

```python
def _build_backend(self) -> object:
    try:
        from phonemizer.backend import EspeakBackend
        return EspeakBackend(language=self.language)
    except ImportError as e:
        raise ImportError("Install phonemizer: pip install phonemizer") from e
```

The `from e` preserves the original traceback. Without it, the original `ImportError` is lost.

In this project: `espeak_phonemizer.py`

## Module 9 — Hash-Based Determinism

This is a pattern specific to ML pipelines that's worth its own treatment.

### 9.1 The problem with random train/val/test splits

If you split randomly, re-running the pipeline produces a different split. A sample that was in val is now in train. Evaluation numbers are incomparable. DVC checksums don't help if the split itself is non-deterministic.

### 9.2 Hash-based split assignment

```python
import hashlib

def _assign_split(file_id: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> str:
    # MD5 of the file ID → a hex string → integer 0..999
    h = int(hashlib.md5(file_id.encode()).hexdigest(), 16) % 1000

    if h < test_ratio * 1000:       # 0..49   → test
        return "test"
    if h < (test_ratio + val_ratio) * 1000:  # 50..99  → val
        return "val"
    return "train"                            # 100..999 → train
```

Given the same `file_id`, this always returns the same split — forever, on any machine, regardless of order files were discovered. Add 1000 new files tomorrow: the existing 10,000 files' splits don't change.

In this project: `ingestion.py:18-24`

### 9.3 SHA256 for deduplication

```python
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

Two audio files with the same content but different names → same SHA256 → deduplicated. The `iter(lambda: f.read(65536), b"")` pattern reads in 64 KB chunks to avoid loading large files into memory.

In this project: `ingestion.py:10-15`

## Exercises

These exercises use the actual project files as your workspace.

### Exercise 1

Add a `field_validator` to `ModelConfig` in `config.py` that rejects `ode_solver` values other than `"dopri5"` and `"euler"`.

### Exercise 2

In `base_phonemizer.py`, the `phonemize_batch` method iterates serially. Rewrite it using `ProcessPoolExecutor` so it parallelizes across texts. Think about what needs to be picklable.

### Exercise 3

Add a new config class `ServingConfig` (it doesn't exist yet) with fields: `host: str`, `port: int`, `n_workers: int`, `log_level: Literal["debug", "info", "warning"]`. Add it to `PipelineConfig`. Write a `field_validator` that rejects ports below 1024.

### Exercise 4

Read `ingestion.py` and annotate the `rows` list on line 36 with a proper `TypedDict` instead of `list[dict]`.

### Exercise 5

In `vi_phonemizer.py`, the `_get_fallback` lazy init pattern is used. Refactor it to use `@lru_cache` as a module-level factory function instead.

## Reference: Key Files for This Course

| Concept                     | File                   | Key lines |
| --------------------------- | ---------------------- | --------- |
| Pydantic v2 models          | `config.py`            | 7–106     |
| Abstract base class         | `base_phonemizer.py`   | all       |
| Concrete ABC impl           | `espeak_phonemizer.py` | all       |
| Lazy init + deferred import | `vi_phonemizer.py`     | 38–44     |
| ProcessPoolExecutor         | `audio_processor.py`   | 49–66     |
| ProcessPoolExecutor         | `feature_extractor.py` | 81–97     |
| Hash-based split + SHA256   | `ingestion.py`         | 10–24     |
| `pyproject.toml`            | `pyproject.toml`       | all       |
| Optional import fallback    | `feature_extractor.py` | 39–57     |

---

## Mini Practice Project: `logcrunch` — A Parallel Log File Analyzer

This is a self-contained project you build from scratch. It has nothing to do with TTS. Its only purpose is to force you to use every concept from this course in a realistic way.

**What it does:** `logcrunch` is a CLI tool that scans a directory of `.log` files, analyzes each file using pluggable analyzers (word count, error rate, line hash deduplication), processes files in parallel, and writes a JSON report. Config is validated with Pydantic.

**What it practices:**

| Course concept          | Where you use it in `logcrunch`                          |
| ----------------------- | -------------------------------------------------------- |
| Type hints + mypy       | Every function signature, strict mode                    |
| `ABC`                   | `BaseAnalyzer` interface, multiple concrete analyzers    |
| Pydantic v2             | `CrunchConfig` with validators                           |
| `pathlib.Path`          | All file I/O, directory scanning                         |
| `ProcessPoolExecutor`   | Parallel analysis of log files                           |
| `pyproject.toml`        | The project's own packaging and `logcrunch` CLI command  |
| `lru_cache`             | Caching compiled regex patterns                          |
| Optional imports        | Optional `rich` for pretty terminal output               |
| Hash-based dedup        | SHA256 to skip duplicate log files                       |

**Time estimate:** 3–5 hours if you write it yourself. 1 hour if you use this guide as a walkthrough.

---

### Step 0 — Project structure

Create this layout from scratch:

```
logcrunch/
├── pyproject.toml
├── src/
│   └── logcrunch/
│       ├── __init__.py
│       ├── analyzers.py      # ABC + concrete analyzers
│       ├── config.py         # Pydantic config
│       ├── runner.py         # ProcessPoolExecutor orchestration
│       ├── hasher.py         # SHA256 deduplication
│       └── cli.py            # entry point
└── sample_logs/              # you'll generate these
    ├── app_2024_01.log
    ├── app_2024_02.log
    └── ...
```

---

### Step 1 — `pyproject.toml`

Write this yourself before writing any Python. The packaging configuration is the contract the rest of the project must honor.

First, make sure your terminal is inside the mini-project directory:

```bash
cd logcrunch
```

If you stay at the parent repository root, `pip install -e ".[dev]"` installs the parent
`tts-pipeline` project instead of this mini project, so the `logcrunch` command will not be
created. From the parent root, use `pip install -e "./logcrunch[dev]"` instead.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "logcrunch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7.0",
]

[project.optional-dependencies]
pretty = [
    "rich>=13.0.0",
]
dev = [
    "mypy>=1.10.0",
    "ruff>=0.4.7",
    "pytest>=8.2.0",
]

[project.scripts]
logcrunch = "logcrunch.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "RUF"]
```

Because `[project.scripts]` points to `logcrunch.cli:main`, create a tiny temporary `cli.py`
before installing. You will replace it with the real CLI in Step 6.

```python
from __future__ import annotations


def main() -> None:
    print("logcrunch CLI is installed")
```

Install it in editable mode so `logcrunch` becomes a real terminal command:

```bash
pip install -e ".[dev]"
```

**Checkpoint:** `logcrunch` should run and print `logcrunch CLI is installed`.
After Step 6, `logcrunch --help` should show the real argparse help text.
If you get `command not found`, check that you ran `pip install -e ".[dev]"` from the
`logcrunch/` directory, not from the parent repository root.

---

### Step 2 — `config.py` (Pydantic v2)

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class CrunchConfig(BaseModel):
    log_dir: Path
    output_file: Path = Path("report.json")
    n_workers: int = 4
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    min_file_size_bytes: int = 0
    analyzers: list[str] = ["word_count", "error_rate"]

    @field_validator("n_workers")
    @classmethod
    def positive_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_workers must be >= 1, got {v}")
        return v

    @field_validator("analyzers")
    @classmethod
    def known_analyzers(cls, v: list[str]) -> list[str]:
        allowed = {"word_count", "error_rate", "line_dedup"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown analyzers: {unknown}. Allowed: {allowed}")
        return v
```

**Try to break it.** Open a Python REPL and run:

```python
from logcrunch.config import CrunchConfig
from pathlib import Path

# This should work
cfg = CrunchConfig(log_dir=Path("/tmp/logs"))
print(cfg.n_workers)   # 4

# This should raise ValidationError
CrunchConfig(log_dir=Path("/tmp"), n_workers=0)

# This should raise ValidationError
CrunchConfig(log_dir=Path("/tmp"), analyzers=["word_count", "made_up"])
```

You have successfully used Pydantic when you understand *why* each of these outcomes happens.

---

### Step 3 — `analyzers.py` (ABC + concrete implementations)

```python
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any


class AnalysisResult:
    def __init__(self, analyzer_name: str, data: dict[str, Any]) -> None:
        self.analyzer_name = analyzer_name
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return {"analyzer": self.analyzer_name, **self.data}


class BaseAnalyzer(ABC):
    """Contract every analyzer must fulfill."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in config and output."""
        ...

    @abstractmethod
    def analyze(self, text: str) -> AnalysisResult:
        """Analyze file content and return a result."""
        ...


# -- Concrete implementations --------------------------------------------------

class WordCountAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "word_count"

    def analyze(self, text: str) -> AnalysisResult:
        words = text.split()
        lines = text.splitlines()
        return AnalysisResult(
            self.name,
            {
                "total_words": len(words),
                "total_lines": len(lines),
                "avg_words_per_line": len(words) / max(len(lines), 1),
            },
        )


class ErrorRateAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "error_rate"

    def analyze(self, text: str) -> AnalysisResult:
        lines = text.splitlines()
        error_lines = [l for l in lines if _error_pattern().search(l)]
        return AnalysisResult(
            self.name,
            {
                "total_lines": len(lines),
                "error_lines": len(error_lines),
                "error_rate": len(error_lines) / max(len(lines), 1),
            },
        )


class LineDedupAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "line_dedup"

    def analyze(self, text: str) -> AnalysisResult:
        lines = text.splitlines()
        unique = set(lines)
        return AnalysisResult(
            self.name,
            {
                "total_lines": len(lines),
                "unique_lines": len(unique),
                "duplicate_lines": len(lines) - len(unique),
            },
        )


# -- lru_cache on a module-level factory: compiled regex is expensive ----------

@lru_cache(maxsize=None)
def _error_pattern() -> re.Pattern[str]:
    return re.compile(r"\b(ERROR|CRITICAL|FATAL|Exception|Traceback)\b", re.IGNORECASE)


# -- Registry: maps config string → analyzer instance -------------------------

def build_analyzers(names: list[str]) -> list[BaseAnalyzer]:
    registry: dict[str, BaseAnalyzer] = {
        "word_count": WordCountAnalyzer(),
        "error_rate": ErrorRateAnalyzer(),
        "line_dedup": LineDedupAnalyzer(),
    }
    return [registry[n] for n in names]
```

**What `lru_cache` is doing here:** `re.compile()` is not expensive for a single call, but if you're analyzing 10,000 files and `_error_pattern()` is called once per file, even small overhead multiplies. The cached version compiles once and returns the same `re.Pattern` object every time. Observe this:

```python
from logcrunch.analyzers import _error_pattern

p1 = _error_pattern()
p2 = _error_pattern()
print(p1 is p2)   # True — same object, not recompiled
```

**Try to violate the ABC contract.** Write a class that inherits `BaseAnalyzer` but forgets to implement `analyze`. Try to instantiate it. You should get `TypeError: Can't instantiate abstract class`. This is the contract enforcing itself.

---

### Step 4 — `hasher.py` (SHA256 deduplication + `pathlib`)

```python
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate(paths: list[Path]) -> tuple[list[Path], int]:
    """Return unique paths (first seen) and count of duplicates removed."""
    seen: dict[str, Path] = {}
    for p in paths:
        digest = sha256(p)
        if digest not in seen:
            seen[digest] = p
    duplicates_removed = len(paths) - len(seen)
    return list(seen.values()), duplicates_removed


def scan_log_dir(log_dir: Path, min_size_bytes: int = 0) -> list[Path]:
    """Find all .log files in a directory, filter by minimum size."""
    all_files = list(log_dir.rglob("*.log"))
    return [p for p in all_files if p.stat().st_size >= min_size_bytes]
```

**Why `pathlib` over `os.path` here:** notice `log_dir.rglob("*.log")` — one clean method call. The equivalent with `os.walk` is 6 lines. And `p.stat().st_size` reads naturally as "get the file stat, then its size in bytes" — no searching for the right `os.path` function.

---

### Step 5 — `runner.py` (`ProcessPoolExecutor` orchestration)

This is the most important file in the project. Read every comment.

```python
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from logcrunch.analyzers import BaseAnalyzer, build_analyzers
from logcrunch.config import CrunchConfig


# IMPORTANT: this function must be at module level (not a nested function or lambda)
# because ProcessPoolExecutor pickles it to send to worker processes.
def _analyze_file(
    file_path: Path,
    analyzer_names: list[str],
) -> dict[str, Any]:
    """Worker function: runs in a subprocess, analyzes one file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    analyzers = build_analyzers(analyzer_names)   # each subprocess builds its own

    results: list[dict[str, Any]] = []
    for analyzer in analyzers:
        result = analyzer.analyze(text)
        results.append(result.to_dict())

    return {
        "file": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "results": results,
    }


def run(
    paths: list[Path],
    cfg: CrunchConfig,
) -> list[dict[str, Any]]:
    """Analyze all files in parallel and return list of per-file reports."""
    reports: list[dict[str, Any]] = []
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
        # Submit all files at once; track which future = which file
        futures = {
            executor.submit(_analyze_file, p, cfg.analyzers): p
            for p in paths
        }

        # as_completed yields each future as it finishes (fastest first)
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:
                failed.append(f"{file_path}: {exc}")

    if failed:
        print(f"[warn] {len(failed)} file(s) failed:")
        for msg in failed:
            print(f"  {msg}")

    return reports


def write_report(reports: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"total_files": len(reports), "files": reports}, f, indent=2)
```

**A critical rule to internalize:** `_analyze_file` is defined at module level, not inside `run()`. If you move it inside `run()`, Python cannot pickle it (it becomes a closure), and `ProcessPoolExecutor` will fail with a cryptic `PicklingError`. This is the #1 mistake people make with `ProcessPoolExecutor`.

**Experiment:** Move `_analyze_file` inside `run()` and observe the error. Then move it back. Now you'll never forget.

---

### Step 6 — `cli.py` (entry point + optional import)

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from logcrunch.config import CrunchConfig
from logcrunch.hasher import deduplicate, scan_log_dir
from logcrunch.runner import run, write_report


def _print_summary(n_files: int, n_deduped: int, output: Path) -> None:
    """Print summary, using `rich` if available, plain text otherwise."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="logcrunch summary")
        table.add_column("Metric")
        table.add_column("Value", style="cyan")
        table.add_row("Files analyzed", str(n_files))
        table.add_row("Duplicates removed", str(n_deduped))
        table.add_row("Report written to", str(output))
        console.print(table)

    except ImportError:
        # rich not installed — plain text fallback, no crash
        print(f"Files analyzed:     {n_files}")
        print(f"Duplicates removed: {n_deduped}")
        print(f"Report written to:  {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel log file analyzer")
    parser.add_argument("log_dir", type=Path, help="Directory containing .log files")
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=0, help="Min file size in bytes")
    parser.add_argument(
        "--analyzers",
        nargs="+",
        default=["word_count", "error_rate"],
        help="Analyzers to run",
    )
    args = parser.parse_args()

    # Pydantic validates everything — wrong types raise here with a clear message
    try:
        cfg = CrunchConfig(
            log_dir=args.log_dir,
            output_file=args.output,
            n_workers=args.workers,
            min_file_size_bytes=args.min_size,
            analyzers=args.analyzers,
        )
    except Exception as exc:
        print(f"[error] Invalid config: {exc}", file=sys.stderr)
        sys.exit(1)

    # Scan and deduplicate
    all_paths = scan_log_dir(cfg.log_dir, min_size_bytes=cfg.min_file_size_bytes)
    unique_paths, n_dupes = deduplicate(all_paths)

    if not unique_paths:
        print("[warn] No .log files found.")
        sys.exit(0)

    # Analyze in parallel
    reports = run(unique_paths, cfg)

    # Write report
    write_report(reports, cfg.output_file)

    _print_summary(len(reports), n_dupes, cfg.output_file)


if __name__ == "__main__":
    main()
```

---

### Step 7 — Generate sample logs and run it

Create a small script `make_sample_logs.py` at the project root to generate test data:

```python
import random
from pathlib import Path

LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
MESSAGES = [
    "Connection established",
    "Request processed",
    "Cache miss",
    "Retrying operation",
    "Exception in thread main",
    "Traceback (most recent call last)",
    "NullPointerException",
    "Disk usage at 90%",
    "User login successful",
    "Timeout after 30s",
]

def make_log(path: Path, n_lines: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_lines):
            level = random.choice(LEVELS)
            msg = random.choice(MESSAGES)
            f.write(f"2024-01-01 00:{i:02d}:00 [{level}] {msg}\n")

log_dir = Path("sample_logs")
for month in range(1, 7):
    make_log(log_dir / f"app_2024_{month:02d}.log", n_lines=300)

# Add a duplicate to test deduplication
import shutil
shutil.copy(log_dir / "app_2024_01.log", log_dir / "app_2024_01_backup.log")
print(f"Created {len(list(log_dir.rglob('*.log')))} log files in {log_dir}/")
```

Run it:
```bash
python make_sample_logs.py
logcrunch sample_logs/ --output report.json --workers 4 --analyzers word_count error_rate line_dedup
```

Expected output (plain text):
```
Files analyzed:     6
Duplicates removed: 1
Report written to:  report.json
```

Open `report.json` and verify each file has results from all three analyzers.

---

### Step 8 — Add `mypy` and fix every error

```bash
mypy src/logcrunch/ --strict
```

You will likely see errors on your first run. Common ones and what they mean:

| Error | What went wrong | Fix |
|---|---|---|
| `error: Function is missing a return type annotation` | You forgot `-> None` on a function | Add it |
| `error: Returning Any from function declared to return ...` | `json.load()` returns `Any` | Narrow it with a cast or `TypedDict` |
| `error: Item "None" of "X \| None" has no attribute "Y"` | You forgot to check for `None` before using | Add `if x is not None:` |
| `error: Argument 1 to "submit" has incompatible type` | Wrong type passed to `executor.submit` | Fix the type of the argument |

**Do not suppress errors with `# type: ignore` until you understand them.** Looking up what each error means is 80% of the learning value of mypy.

---

### Step 9 — Add one more analyzer (the real test)

Add a `TopWordsAnalyzer` that finds the 10 most common words in a log file. Requirements:

1. It must subclass `BaseAnalyzer` — if you forget an abstract method, Python tells you at instantiation time
2. Add `"top_words"` to the `allowed` set in `CrunchConfig.known_analyzers`
3. Add it to the `registry` in `build_analyzers()`
4. Run `logcrunch sample_logs/ --analyzers top_words` and verify the output

If you can do this without looking at the existing analyzer code, you have internalized the ABC pattern.

---

### Completion Checklist

Mark each when you can do it without referring to notes:

- [ ] Explain why `from __future__ import annotations` is on line 1 of every file
- [ ] Write a Pydantic model with a nested model and a `field_validator` from memory
- [ ] Explain why `_analyze_file` must be at module level, not inside `run()`
- [ ] Add a fourth analyzer end-to-end without breaking any existing code
- [ ] Run `mypy --strict` with zero errors
- [ ] Explain the difference between `ProcessPoolExecutor` and `ThreadPoolExecutor` and when to use each
- [ ] Explain what `lru_cache` prevents in `_error_pattern()`
- [ ] Show what happens when you try to instantiate `BaseAnalyzer()` directly
- [ ] Run `logcrunch` as a terminal command (not `python -m ...`) and explain how entry points work
