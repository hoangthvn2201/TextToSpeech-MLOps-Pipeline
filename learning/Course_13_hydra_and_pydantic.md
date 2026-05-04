# Course 13: Hydra + Pydantic — Configuration Management

**Prerequisites:** Course 1 (Python for ML Engineers) — Pydantic v2 BaseModel, field_validator. Course 10 (DVC) — `params.yaml` as single source of truth.

**What this course covers:** Hydra composes multiple YAML files into one config object at runtime and lets you override any value from the CLI. Pydantic validates and type-coerces that config the moment it enters Python. Together they give the pipeline a typed, composable, CLI-overridable configuration system with zero boilerplate. By the end you will understand every line in `configs/` and `utils/config.py`.

**Project anchors:**
- [configs/config.yaml](../configs/config.yaml) — root config, defaults list, interpolation
- [configs/data/base.yaml](../configs/data/base.yaml) — data config group
- [configs/data/vi.yaml](../configs/data/vi.yaml) — Vietnamese overrides
- [configs/training/base.yaml](../configs/training/base.yaml) — training config group
- [configs/training/finetune.yaml](../configs/training/finetune.yaml) — fine-tune variant
- [configs/model/matcha_tts.yaml](../configs/model/matcha_tts.yaml) — acoustic model config
- [configs/model/hifigan.yaml](../configs/model/hifigan.yaml) — vocoder config group
- [src/tts_pipeline/utils/config.py](../src/tts_pipeline/utils/config.py) — Pydantic models + `from_hydra()`
- [scripts/run_training.py](../scripts/run_training.py) — `@hydra.main` entry point

---

## Module 1 — The Config Management Problem

### Why plain `argparse` or a single YAML file breaks down

A TTS pipeline has dozens of parameters across four concerns:

```
data:      sample_rate, n_mels, hop_length, min_snr_db, ...   (18 keys)
model:     encoder_dim, decoder_channels, sigma_min, ...       (8 keys)
training:  learning_rate, warmup_steps, precision, ...         (14 keys)
evaluation: absolute_floor, relative_delta, whisper_model, ... (8 keys)
```

Problems with a single big config file:
- Switching to a fine-tuning run means manually editing `learning_rate`, `warmup_steps`, `max_epochs`, `resume_from_checkpoint` — and remembering to undo them afterward
- Switching languages (vi → en) means editing `data.language`, `text.phonemizer`, `text.language` — scattered across the file
- Running a hyperparameter sweep requires a shell loop that edits the file, runs, edits again

Problems with `argparse`:
- Every new parameter needs a new `parser.add_argument()` call
- Nested structures (`model.encoder_dim`) require awkward flat naming
- No config file — reproducibility requires saving the full `argparse.Namespace` manually

### Hydra's solution: composable config groups

Hydra lets you define **config groups** — directories of alternative YAML files for the same config section:

```
configs/
├── config.yaml          ← root: composes everything
├── data/
│   ├── base.yaml        ← default data config
│   └── vi.yaml          ← Vietnamese-specific overrides
├── training/
│   ├── base.yaml        ← default training config
│   └── finetune.yaml    ← fine-tune override (different LR, epochs, checkpoint)
└── model/
    ├── matcha_tts.yaml  ← acoustic model
    └── hifigan.yaml     ← vocoder (used when training vocoder)
```

Switching to fine-tuning is one CLI flag: `training=finetune`. All `finetune.yaml` values override the base; everything else stays the same. No file edits, no undo.

---

## Module 2 — The Root Config and the `defaults` List

### `configs/config.yaml`

```yaml
defaults:                          # config.yaml:5
  - data: base                     # load configs/data/base.yaml under key "data"
  - model: matcha_tts              # load configs/model/matcha_tts.yaml under "model"
  - training: base                 # load configs/training/base.yaml under "training"
  - evaluation: metrics            # load configs/evaluation/metrics.yaml under "evaluation"
  - serving: api                   # load configs/serving/api.yaml under "serving"
  - _self_                         # merge keys defined directly in config.yaml last

experiment:
  name: tts-${data.language}-matcha
  run_name: "${experiment.name}-${now:%Y%m%d_%H%M%S}"
  tags:
    language: ${data.language}
    model: matcha_tts

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: ${experiment.name}

paths:
  data_root: data
  artifacts: artifacts
  reports: reports/latest
  checkpoints: artifacts/checkpoints
  export: artifacts/export
```

### How `defaults` composition works

Hydra reads the `defaults` list top to bottom, loading each file and merging it into the config:

```
Step 1: load configs/data/base.yaml       → cfg.data.* keys
Step 2: load configs/model/matcha_tts.yaml → cfg.model.* keys
Step 3: load configs/training/base.yaml   → cfg.training.* keys
Step 4: load configs/evaluation/metrics.yaml → cfg.evaluation.* keys
Step 5: load configs/serving/api.yaml     → cfg.serving.* keys
Step 6: merge config.yaml body (_self_)   → cfg.experiment.*, cfg.mlflow.*, cfg.paths.*
```

The final `cfg` object is a single merged dictionary of all these files. `_self_` controls where the root config's own keys are inserted relative to the group configs — at position `last` (default), the root overrides any group keys with the same name.

### The `@package` directive

Each group config starts with `# @package <name>`:

```yaml
# @package data            ← data/base.yaml:1
language: vi
sample_rate: 22050
```

This tells Hydra to place this file's keys under the `data` key in the merged config. Without it, keys would go to the root level and collide with other groups. Our project's `@package data` line ensures `language: vi` becomes `cfg.data.language`, not `cfg.language`.

---

## Module 3 — Variable Interpolation

### `${key.path}` syntax

Hydra (via OmegaConf) supports interpolation — one config value can reference another:

```yaml
experiment:
  name: tts-${data.language}-matcha        # config.yaml:14
  run_name: "${experiment.name}-${now:%Y%m%d_%H%M%S}"
  tags:
    language: ${data.language}             # config.yaml:17
```

When `cfg.experiment.name` is accessed, OmegaConf resolves `${data.language}` to `"vi"` at that moment, producing `"tts-vi-matcha"`. If you change `data.language=en` at the CLI, `experiment.name` automatically becomes `"tts-en-matcha"` — you never update `experiment.name` manually.

### Special interpolation resolvers

| Syntax | Resolves to | Example |
|---|---|---|
| `${key.path}` | Another config value | `${data.language}` → `"vi"` |
| `${now:%Y%m%d_%H%M%S}` | Current timestamp | `"20240515_143022"` |
| `${hydra:runtime.cwd}` | Working directory at launch | `"/home/user/project"` |
| `${hydra:job.name}` | The job name (script name) | `"run_training"` |
| `${oc.env:MY_VAR}` | Environment variable | `${oc.env:MLFLOW_URI}` → `"http://..."` |

The `${now:...}` resolver is what makes every training run get a unique `run_name`:

```yaml
run_name: "${experiment.name}-${now:%Y%m%d_%H%M%S}"
# resolves to: "tts-vi-matcha-20240515_143022"
```

MLflow uses this as the run name — uniquely identifying every training run without a manual counter.

### `OmegaConf.to_container(resolve=True)`

Inside Python code, interpolations are resolved lazily. To get a plain Python dict with all interpolations pre-resolved:

```python
def from_hydra(cfg: DictConfig) -> PipelineConfig:           # config.py:102
    raw: dict = OmegaConf.to_container(cfg, resolve=True)
    return PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        ...
    )
```

`resolve=True` evaluates every `${...}` expression before converting to a dict. Without it, the dict contains raw strings like `"${data.language}"` instead of `"vi"`.

---

## Module 4 — `@hydra.main`: The Entry Point

### How `@hydra.main` works

```python
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:                           # run_training.py:8
    pipeline_cfg = from_hydra(cfg)
    ...
```

When Python calls `main()`:
1. Hydra locates `configs/config.yaml` (relative to `run_training.py` via `config_path`)
2. Reads the `defaults` list, loads and merges all group configs
3. Applies any CLI overrides (e.g., `training.learning_rate=5e-5`)
4. Injects the final merged `DictConfig` as `cfg`
5. Calls your function body

`config_name="config"` means Hydra looks for `config.yaml` (the `.yaml` is implicit).

`version_base="1.3"` pins the Hydra behavior version — prevents silent behavior changes when upgrading Hydra. Always specify it.

### Accessing config values

```python
# Dot access (OmegaConf DictConfig)
cfg.data.language          # "vi"
cfg.training.learning_rate # 1e-4
cfg.paths.checkpoints      # "artifacts/checkpoints"

# Dict-style access (equivalent)
cfg["data"]["language"]    # "vi"

# Missing key
cfg.data.nonexistent       # raises omegaconf.errors.ConfigAttributeError
cfg.data.get("nonexistent", "default")   # "default"
```

### What Hydra writes to disk

Every time `@hydra.main` runs, Hydra creates an **output directory** under `outputs/` (by default):

```
outputs/
└── 2024-05-15/
    └── 14-30-22/
        ├── run_training.log    ← captured stdout/stderr
        └── .hydra/
            ├── config.yaml     ← the fully merged config (resolved)
            ├── overrides.yaml  ← what CLI overrides were applied
            └── hydra.yaml      ← Hydra metadata
```

`config.yaml` in `.hydra/` is the exact config that ran — the definitive record of what parameters produced this run. Combined with `dvc.lock` and MLflow run tags, you have triple reproducibility.

---

## Module 5 — CLI Overrides

### Basic syntax

```bash
# Nested key override (dot notation)
python scripts/run_training.py training.learning_rate=5e-5

# Multiple overrides
python scripts/run_training.py training.learning_rate=5e-5 training.max_epochs=200

# String values (quote if needed)
python scripts/run_training.py data.language=en

# Boolean
python scripts/run_training.py training.use_compile=true

# List
python scripts/run_training.py model.upsample_rates=[8,8,4,2]
```

Overrides are applied after config composition — they take the highest priority.

### Switching config groups

```bash
# Use finetune.yaml instead of base.yaml for the training group
python scripts/run_training.py training=finetune

# Use hifigan.yaml for model group (train vocoder)
python scripts/run_training.py --config-name hifigan    # dvc.yaml:121
```

`training=finetune` replaces the entire `configs/training/base.yaml` with `configs/training/finetune.yaml`. The resulting `cfg.training` has `learning_rate=1e-5`, `max_epochs=200`, `warmup_steps=200`, `resume_from_checkpoint=artifacts/checkpoints/pretrained.ckpt` — all from `finetune.yaml`. Keys not in `finetune.yaml` fall back to `base.yaml`.

```yaml
# configs/training/finetune.yaml
# @package training
max_epochs: 200
learning_rate: 1.0e-5              # 10× smaller than base
warmup_steps: 200
early_stopping_patience: 20
freeze_encoder: false
resume_from_checkpoint: artifacts/checkpoints/pretrained.ckpt
```

### Adding new keys (`+` prefix)

```bash
# Add a key not in any config file
python scripts/run_training.py +training.debug_mode=true
```

Without `+`, overriding a key that doesn't exist raises an error. The `+` prefix creates a new key.

### Removing keys (`~` prefix)

```bash
# Remove a key from the composed config
python scripts/run_training.py ~training.resume_from_checkpoint
```

After `~`, `cfg.training.resume_from_checkpoint` raises `ConfigAttributeError` — the key is absent.

---

## Module 6 — Multirun Sweeps

### `--multirun` flag

```bash
# Run once for each combination of values (Cartesian product)
python scripts/run_training.py --multirun \
    training.learning_rate=1e-3,1e-4,5e-5 \
    model.encoder_layers=4,6
```

This launches **6 runs** (3 × 2), each with a different `(learning_rate, encoder_layers)` combination. Each run gets its own output directory under `multirun/`:

```
multirun/
└── 2024-05-15/
    └── 14-30-22/
        ├── 0/   ← lr=1e-3, layers=4
        ├── 1/   ← lr=1e-3, layers=6
        ├── 2/   ← lr=1e-4, layers=4
        ├── 3/   ← lr=1e-4, layers=6
        ├── 4/   ← lr=5e-5, layers=4
        └── 5/   ← lr=5e-5, layers=6
```

Each run has its own `.hydra/config.yaml` recording the exact parameters. Each run creates its own MLflow run (via the `run_name` interpolation, which includes the timestamp).

### Sweeper plugins

Hydra's default sweeper enumerates the Cartesian product. With `hydra-optuna-sweeper` plugin, you can do Bayesian optimization:

```bash
pip install hydra-optuna-sweeper

python scripts/run_training.py --multirun \
    hydra/sweeper=optuna \
    training.learning_rate="interval(1e-5, 1e-3)" \
    model.encoder_layers="choice(4,6,8)" \
    hydra.sweeper.n_trials=20
```

This runs 20 Optuna trials, each suggesting `learning_rate` from `[1e-5, 1e-3]` and `encoder_layers` from `{4, 6, 8}` based on previous results. The project doesn't use this yet — it's an extension path.

---

## Module 7 — `DictConfig` and OmegaConf

### What `DictConfig` is

`DictConfig` is OmegaConf's typed dictionary class. It is **not** a plain Python `dict`:

```python
cfg.data.language          # dot access — works
cfg["data"]["language"]    # bracket access — works
dict(cfg)                  # ← BAD: only top-level keys, nested DictConfigs not converted
cfg.data["nonexistent"]    # raises ConfigAttributeError, not KeyError
```

### Converting to plain Python

```python
# Convert the entire cfg to a plain nested dict (all interpolations resolved)
raw = OmegaConf.to_container(cfg, resolve=True)   # config.py:103
type(raw["data"])   # dict (not DictConfig)
type(raw["data"]["sample_rate"])  # int

# Convert a sub-config to YAML string (for logging)
print(OmegaConf.to_yaml(cfg.training))

# Merge two configs
merged = OmegaConf.merge(base_cfg, override_cfg)
```

### Structured configs

OmegaConf also supports `@dataclass`-based structured configs (type-checked at composition time):

```python
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class TrainingConf:
    learning_rate: float = 1e-4
    max_epochs: int = MISSING   # must be provided — no default
```

Our project doesn't use structured configs — it uses `DictConfig` (untyped YAML) at the Hydra layer and Pydantic (typed Python) immediately after. This is a valid architecture: Hydra handles composition and CLI, Pydantic handles validation.

---

## Module 8 — Pydantic v2 Models

### The Pydantic models in `config.py`

```python
class DataConfig(BaseModel):                      # config.py:10
    language: str = "vi"
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    trim_silence: bool = True
    peak_normalize_db: float = -3.0
    min_duration: float = 0.5
    max_duration: float = 15.0
    min_snr_db: float = 20.0
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    dataset_grade_threshold: str = "C"

    @field_validator("dataset_grade_threshold")
    @classmethod
    def valid_grade(cls, v: str) -> str:          # config.py:28
        if v not in ("A", "B", "C", "D", "F"):
            raise ValueError(f"Invalid grade threshold: {v}")
        return v
```

### What Pydantic does that Hydra/OmegaConf doesn't

| Feature | OmegaConf DictConfig | Pydantic BaseModel |
|---|---|---|
| Dot-access | ✅ | ✅ |
| Default values | ✅ | ✅ |
| Type annotation | ❌ (values are Python primitives but no class-level enforcement) | ✅ (coercion + error on wrong type) |
| Custom validators | ❌ | ✅ (`@field_validator`) |
| IDE type hints | ❌ (DictConfig is opaque to type checkers) | ✅ |
| `.model_dump()` | ❌ | ✅ |
| Nested model composition | Possible but manual | ✅ (fields can be other BaseModels) |

### Type coercion

Pydantic v2 coerces values to the declared type:

```python
DataConfig(sample_rate="22050")   # str → int: works, sample_rate=22050
DataConfig(sample_rate="abc")     # str → int: raises ValidationError
DataConfig(n_mels=80.0)           # float → int: works, n_mels=80
DataConfig(trim_silence="true")   # str → bool: works, trim_silence=True
DataConfig(trim_silence="yes")    # str → bool: works in v2, trim_silence=True
```

This is important because YAML sometimes loads values as unexpected types. `val_ratio: 0.05` in YAML is a float — but `val_ratio: 5e-2` might load as a string in some parsers. Pydantic's coercion handles these edge cases automatically.

### `@field_validator` — the grade threshold check

```python
@field_validator("dataset_grade_threshold")
@classmethod
def valid_grade(cls, v: str) -> str:
    if v not in ("A", "B", "C", "D", "F"):
        raise ValueError(f"Invalid grade threshold: {v}")
    return v
```

This runs when `DataConfig` is instantiated. If someone sets `dataset_grade_threshold: Z` in `params.yaml` or via CLI (`data.dataset_grade_threshold=Z`), Pydantic raises `ValidationError` at the start of the script — not silently during validation hours later.

The `@classmethod` requirement is Pydantic v2 syntax. In v1 it was an instance method. Always include it in v2.

### `model_dump()` — back to dict

```python
model_kw = pipeline_cfg.model.model_dump()    # trainer.py:41
model_kw["vocab_size"] = data_module.tokenizer.vocab_size

pl_model = MatchaTTSLightning.auto_resume(
    checkpoint_dir=checkpoint_dir,
    model_cfg=model_kw,                        # trainer.py:46
    training_cfg=pipeline_cfg.training.model_dump(),
)
```

`model_dump()` serializes the Pydantic model back to a plain dict. This is how the validated, typed config re-enters the codebase as keyword arguments to model constructors.

---

## Module 9 — The Full Config Pipeline

### Tracing a value from YAML to model

Let's follow `encoder_dim=192` from config file to model:

```
configs/model/matcha_tts.yaml:      encoder_dim: 192
        ↓ Hydra composition
cfg (DictConfig):                   cfg.model.encoder_dim = 192
        ↓ OmegaConf.to_container(resolve=True)
raw (dict):                         raw["model"]["encoder_dim"] = 192
        ↓ ModelConfig(**raw["model"])
pipeline_cfg.model (ModelConfig):   pipeline_cfg.model.encoder_dim = 192  (int, validated)
        ↓ pipeline_cfg.model.model_dump()
model_kw (dict):                    {"encoder_dim": 192, ...}
        ↓ MatchaTTS(**model_cfg)
MatchaTTS.__init__:                 self.encoder = PhonemeEncoder(dim=192, ...)
```

Every step is typed. If `encoder_dim: "big"` is accidentally placed in the YAML, the error appears at `ModelConfig(**raw["model"])` with a clear Pydantic message — not in PyTorch's internals.

### `from_hydra()` — the bridge

```python
def from_hydra(cfg: DictConfig) -> PipelineConfig:          # config.py:102
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    return PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        text=TextConfig(**raw.get("text", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
```

This is called at the top of both `run_training.py` and `run_evaluation.py`. After this call, all config access goes through typed Pydantic attributes — no more `cfg.data.language` (untyped string), but `pipeline_cfg.data.language` (validated `str` field).

### `load_params()` — DVC compatibility

```python
def load_params(params_path: Path = Path("params.yaml")) -> PipelineConfig:  # config.py:113
    cfg = OmegaConf.load(params_path)
    return from_hydra(cfg)
```

Some scripts (`audio_processor.py`, `feature_extractor.py`) run without Hydra — called directly by DVC. They use `load_params()` to load `params.yaml` via OmegaConf and get the same `PipelineConfig` object. Same validation, same types, no Hydra runtime needed. The project has one config format (`PipelineConfig`) regardless of how the script is launched.

---

## Module 10 — Common Patterns and Pitfalls

### Pattern: guarding lazy imports inside `@hydra.main`

```python
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from tts_pipeline.training.trainer import run_training    # run_training.py:12-14
    from tts_pipeline.data.dataset import TTSDataModule
    ...
```

All imports are inside the function body, not at module level. This is intentional: Hydra changes the working directory when it runs, so relative imports resolve correctly only after Hydra sets up the environment. Module-level imports run before Hydra's setup and may fail to find resources.

### Pattern: overriding for one-off experiments

```bash
# Quick ablation: halve the learning rate, run for fewer epochs
python scripts/run_training.py \
    training.learning_rate=5e-5 \
    training.max_epochs=100 \
    experiment.run_name="ablation-lr5e5"
```

The `experiment.run_name` override is important: without it, the auto-generated name includes a timestamp, but you want a memorable name for the ablation.

### Pitfall 1: Forgetting `resolve=True` in `to_container`

```python
# BAD — interpolations are not resolved
raw = OmegaConf.to_container(cfg)
raw["experiment"]["name"]   # "${data.language}-matcha" — literal string, not "vi-matcha"

# GOOD
raw = OmegaConf.to_container(cfg, resolve=True)
raw["experiment"]["name"]   # "tts-vi-matcha"
```

Pydantic receives `"${data.language}-matcha"` as a string and accepts it — there's no error. The bug manifests later when the MLflow experiment name is wrong.

### Pitfall 2: Overriding a group config with `=` instead of group syntax

```bash
# WRONG — tries to set a key named "training" to the string "finetune"
python scripts/run_training.py training="finetune"

# CORRECT — selects the finetune config group
python scripts/run_training.py training=finetune
```

Group overrides use `group=choice` syntax without quotes. Value overrides use `key=value`.

### Pitfall 3: Working directory changes

Hydra changes the CWD to the output directory by default:

```
Before hydra.main: CWD = /home/user/project
Inside main():     CWD = /home/user/project/outputs/2024-05-15/14-30-22/
```

This breaks any relative path used inside `main()`:

```python
# BAD — resolves to outputs/.../data/metadata.csv (doesn't exist)
df = pd.read_csv("data/metadata.csv")

# GOOD — use hydra's original CWD
import hydra
orig_cwd = hydra.utils.get_original_cwd()
df = pd.read_csv(Path(orig_cwd) / "data/metadata.csv")

# OR configure Hydra to not change CWD
# hydra:
#   job:
#     chdir: false    ← add to config.yaml's hydra: section
```

Our project uses `cfg.paths.data_root`, `cfg.paths.checkpoints` etc. — all relative paths are resolved through the config, which keeps them relative to the original CWD.

### Pitfall 4: `@field_validator` without `@classmethod`

```python
# WRONG — Pydantic v2 raises a warning/error at class definition
@field_validator("dataset_grade_threshold")
def valid_grade(cls, v):   # ← missing @classmethod
    ...

# CORRECT
@field_validator("dataset_grade_threshold")
@classmethod
def valid_grade(cls, v: str) -> str:
    ...
```

In Pydantic v1 `@classmethod` was optional (implicit). In v2 it is required.

### Pitfall 5: Calling `model_dump()` before validation runs

```python
# No validation yet — this is just a dict
raw = {"encoder_dim": "big"}

# BAD — manual dict access bypasses Pydantic
encoder_dim = raw["encoder_dim"]   # "big" (string) — no error until PyTorch complains

# GOOD — validation fires at construction
model_cfg = ModelConfig(**raw)     # raises ValidationError: encoder_dim must be int
encoder_dim = model_cfg.encoder_dim  # 192 (int) if valid
```

---

## Module 11 — Mini Practice Project: `configlab`

An independent mini-project that builds a Hydra-composed, Pydantic-validated config system for a tiny MLP experiment. No TTS pipeline code.

**Goal:** Run grid search over learning rate and hidden size from the CLI, with each run getting its own MLflow experiment name, using Pydantic to validate that hidden size is a power of 2.

### Project structure

```
configlab/
├── configs/
│   ├── config.yaml
│   ├── model/
│   │   ├── small.yaml
│   │   └── large.yaml
│   └── training/
│       ├── base.yaml
│       └── fast.yaml
├── schema.py    # Pydantic models
└── train.py     # @hydra.main entry point
```

### `configs/config.yaml`

```yaml
defaults:
  - model: small
  - training: base
  - _self_

experiment:
  name: "mlp-${model.hidden_size}-lr${training.learning_rate}"
  seed: 42
```

### `configs/model/small.yaml`

```yaml
# @package model
hidden_size: 64
n_layers: 2
dropout: 0.1
```

### `configs/model/large.yaml`

```yaml
# @package model
hidden_size: 256
n_layers: 4
dropout: 0.2
```

### `configs/training/base.yaml`

```yaml
# @package training
learning_rate: 1.0e-3
max_epochs: 50
batch_size: 32
```

### `configs/training/fast.yaml`

```yaml
# @package training
learning_rate: 5.0e-3
max_epochs: 10
batch_size: 64
```

### `schema.py` — Pydantic models

```python
import math
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1

    @field_validator("hidden_size")
    @classmethod
    def must_be_power_of_two(cls, v: int) -> int:
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"hidden_size={v} is not a power of 2")
        return v

    @field_validator("dropout")
    @classmethod
    def valid_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v


class TrainingConfig(BaseModel):
    learning_rate: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 32

    @field_validator("learning_rate")
    @classmethod
    def positive_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"learning_rate must be positive, got {v}")
        return v


class ExperimentConfig(BaseModel):
    name: str = "default"
    seed: int = 42


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    experiment: ExperimentConfig = ExperimentConfig()
```

### `train.py`

```python
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from schema import Config, ModelConfig, TrainingConfig, ExperimentConfig


def fake_train(cfg: Config) -> float:
    """Simulate training — returns a fake val_loss."""
    random.seed(cfg.experiment.seed)
    # Fake: smaller lr and bigger model → better (lower) loss
    base = 1.0 / (cfg.model.hidden_size ** 0.5)
    noise = random.gauss(0, 0.01)
    val_loss = base * (1 + cfg.training.learning_rate * 100) + noise
    return round(val_loss, 4)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 1. Show the resolved config
    print("=== Resolved config ===")
    print(OmegaConf.to_yaml(cfg))

    # 2. Bridge to Pydantic (validation fires here)
    raw = OmegaConf.to_container(cfg, resolve=True)
    try:
        pipeline_cfg = Config(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            experiment=ExperimentConfig(**raw.get("experiment", {})),
        )
    except Exception as e:
        print(f"Config validation failed: {e}")
        raise

    # 3. Train
    val_loss = fake_train(pipeline_cfg)
    print(f"\nExperiment: {pipeline_cfg.experiment.name}")
    print(f"  hidden_size={pipeline_cfg.model.hidden_size}")
    print(f"  learning_rate={pipeline_cfg.training.learning_rate}")
    print(f"  val_loss={val_loss}")


if __name__ == "__main__":
    main()
```

### Running the project

```bash
pip install hydra-core pydantic omegaconf

# Default run (small model, base training)
python train.py

# Switch to large model
python train.py model=large

# Override learning rate
python train.py training.learning_rate=5e-4

# Multirun grid search
python train.py --multirun \
    model=small,large \
    training.learning_rate=1e-3,1e-4,5e-5
# → 6 runs: all (model, lr) combinations
```

### Exercises

1. **Observe output directories** — after `python train.py`, inspect `outputs/`. Open `.hydra/config.yaml` and compare it to your `configs/config.yaml`. Notice that interpolations like `${model.hidden_size}` are resolved in the output copy.

2. **Trigger a Pydantic error** — run `python train.py model.hidden_size=100`. Observe the `ValidationError` from `must_be_power_of_two`. Then fix it: `model.hidden_size=128`.

3. **Add a new config group** — create `configs/training/debug.yaml` with `max_epochs: 2`, `batch_size: 4`. Run `python train.py training=debug`. Verify only the overridden keys change (not `learning_rate`).

4. **Add an environment variable interpolation** — add `api_key: ${oc.env:MY_API_KEY,"default_key"}` to `configs/config.yaml`. Run with `MY_API_KEY=secret python train.py` and verify the value is resolved. Run without the env var and verify the default is used.

5. **Multirun output structure** — run the grid search above. Inspect `multirun/` — find the `.hydra/overrides.yaml` for each run. Verify each run recorded a different `(model, learning_rate)` combination.

6. **`load_params` pattern** — write a standalone function (no `@hydra.main`) that loads `configs/config.yaml` using `OmegaConf.load()`, applies one override manually with `OmegaConf.merge()`, converts to `Config`, and prints the result. This is the `load_params()` equivalent for non-Hydra scripts.

---

## Key Takeaways

- Hydra composes multiple YAML files using the `defaults:` list. The `@package` directive controls where each file's keys appear in the merged config. `_self_` controls insertion order of the root config's own keys.
- `${key.path}` interpolation resolves lazily in `DictConfig` and eagerly in `OmegaConf.to_container(resolve=True)`. Always use `resolve=True` before passing to Pydantic.
- CLI overrides (`key=value`) take highest priority over all YAML files. Group overrides (`group=choice`) swap entire config files. `+key=value` adds new keys; `~key` removes them.
- `--multirun` with `key=a,b,c` runs the Cartesian product of all override combinations, each in its own output directory with its own `.hydra/overrides.yaml`.
- `@hydra.main` changes the working directory to the output directory. Use `hydra.utils.get_original_cwd()` or configure `hydra.job.chdir=false` to keep relative paths working.
- Pydantic v2 `BaseModel` is the typed Python boundary. `@field_validator` + `@classmethod` runs custom validation at construction time — bad configs fail fast with a clear message.
- `model_dump()` converts a Pydantic model back to a plain dict for passing as `**kwargs` to model constructors. `from_hydra()` is the bridge that converts `DictConfig → dict → PipelineConfig`.
- `load_params()` loads `params.yaml` via `OmegaConf.load()` and feeds it through the same `from_hydra()` path — giving DVC-called scripts the same typed config as Hydra-called scripts.