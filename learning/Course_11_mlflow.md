# Course 11: MLflow — Experiment Tracking and Model Registry

**Prerequisites:** Course 5 (PyTorch Lightning) — you should know `self.log()`, `MLFlowLogger`, and the `run_training()` function. Course 10 (DVC) — you should know how stages chain and how `dvc.lock` captures artifact hashes.

**What this course covers:** MLflow is the experiment tracking and model lifecycle layer of the pipeline. By the end you will understand how every `mlflow.*` call in the codebase works, what the Model Registry stages mean, and how the evaluation gate uses MLflow to promote or block a new model from reaching production.

**Project anchors:**
- [src/tts_pipeline/utils/registry.py](../src/tts_pipeline/utils/registry.py) — `register_model`, `transition_stage`, `load_model`
- [src/tts_pipeline/training/trainer.py](../src/tts_pipeline/training/trainer.py) — `MLFlowLogger`, `mlflow.start_run`, model logging
- [src/tts_pipeline/training/callbacks.py](../src/tts_pipeline/training/callbacks.py) — `MLflowMetricsCallback`, artifact logging
- [src/tts_pipeline/evaluation/reporter.py](../src/tts_pipeline/evaluation/reporter.py) — `generate_report`, `check_gates`, metric logging
- [src/tts_pipeline/evaluation/baseline.py](../src/tts_pipeline/evaluation/baseline.py) — `load_baseline` from Production stage
- [scripts/run_evaluation.py](../scripts/run_evaluation.py) — full evaluation → gate → stage transition flow

---

## Module 1 — What MLflow Is

MLflow is an open-source platform with four components. Our pipeline uses three of them:

| Component | What it does | Used in project? |
|---|---|---|
| **Tracking** | Log metrics, params, artifacts, tags for each run | Yes — every training run |
| **Model Registry** | Version models, manage lifecycle stages | Yes — Staging → Shadow → Production |
| **Projects** | Package code as reproducible experiments | No (DVC handles this) |
| **Models** | Standard model format with flavors | Yes — `mlflow.pytorch` flavor |

### The core data model

```
MLflow Server
└── Experiment  ("tts-vi-training")
    ├── Run  (run_id: "abc123")
    │   ├── params  {learning_rate: 1e-4, n_mels: 80, ...}
    │   ├── metrics {train/loss: [...], val/loss: [...]}
    │   ├── tags    {dvc_commit: "fd2d31f", language: "vi"}
    │   └── artifacts/
    │       ├── matcha_tts/   ← logged model
    │       ├── val_mels/     ← mel spectrograms per epoch
    │       └── evaluation_report.json
    └── Run  (run_id: "def456")
        └── ...

Model Registry
└── Registered Model  ("matcha-tts-vi")
    ├── Version 1  (run_id: "abc123", stage: "Production")
    ├── Version 2  (run_id: "def456", stage: "Staging")
    └── Version 3  (run_id: "ghi789", stage: "None")
```

A **Run** belongs to an **Experiment**. A **Registered Model** has multiple **Versions**, each pointing to a Run's artifacts. The Model Registry is separate from the Tracking server — you can have runs without registered models, but every registered model version must come from a run.

---

## Module 2 — MLflow Tracking: Runs

### Starting a run

```python
# Explicit context manager — most explicit form
with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_metric("val/loss", 0.42, step=1000)
    # run auto-ends when the with block exits
```

```python
# Fluent API without context manager
mlflow.start_run(run_name="experiment-1")
mlflow.log_param("n_mels", 80)
mlflow.end_run()   # must call manually — easy to forget
```

Our project uses the context manager form in `run_training()`:

```python
with mlflow.start_run(run_id=run_id):           # trainer.py:82
    mlflow.set_tag("dvc_commit", dvc_commit)
    mlflow.set_tag("language", cfg.data.language)
    trainer.fit(pl_model, datamodule=data_module)
    mlflow.pytorch.log_model(pl_model, artifact_path="matcha_tts")
```

### `run_id` reuse: why we pass it explicitly

The `MLFlowLogger` creates a run when instantiated:

```python
mlflow_logger = MLFlowLogger(                       # trainer.py:29
    experiment_name=cfg.experiment.name,
    run_name=cfg.experiment.run_name,
    tracking_uri=cfg.mlflow.tracking_uri,
    tags=dict(cfg.experiment.tags),
)
run_id = mlflow_logger.run_id                       # trainer.py:79
```

When we later call `mlflow.start_run(run_id=run_id)`, MLflow **resumes** that same run rather than creating a new one. This means:
- Metrics logged by `self.log()` (via `MLFlowLogger`) go into the same run
- Tags and artifacts logged by the manual `mlflow.*` calls go into the same run
- There is exactly one run in the UI, not two

Without `run_id=run_id`, you'd have two separate runs for the same training: one with metrics, one with the model artifact — impossible to connect them.

### `mlflow.active_run()`

```python
def on_validation_epoch_end(self, trainer, pl_module):  # callbacks.py:28
    if not mlflow.active_run():
        return
```

`mlflow.active_run()` returns the current active run object, or `None` if no run is active. Callbacks guard with this check so they work correctly in unit tests or when MLflow is disabled — without it, `mlflow.log_artifact()` would raise `MlflowException: No active run`.

---

## Module 3 — Logging: Metrics, Params, Tags, Artifacts

### Metrics

```python
# Log a scalar at a specific step
mlflow.log_metric("val/loss", 0.42, step=1000)

# Log multiple at once
mlflow.log_metrics({"train/loss": 0.31, "train/flow_loss": 0.28}, step=1000)
```

Metrics have a `step` dimension — MLflow stores a time series, not a single value. This is what powers the "Compare Runs" plot in the UI: you can overlay `val/loss` across multiple runs on the same axis.

`MLflowMetricsCallback` reads from `trainer.callback_metrics` at the end of each epoch:

```python
def on_train_epoch_end(self, trainer, pl_module):       # callbacks.py:49
    if mlflow.active_run():
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        mlflow.log_metrics(metrics, step=trainer.global_step)
```

`float(v)` is required — `trainer.callback_metrics` values are PyTorch tensors and `mlflow.log_metrics` expects Python floats.

### Parameters

```python
mlflow.log_param("learning_rate", 1e-4)     # single
mlflow.log_params({"n_mels": 80, "n_fft": 1024})  # multiple
```

Params are logged once per run (they're immutable after logging). Trying to log the same param twice with a different value raises `MlflowException`. Use tags for mutable metadata.

In our pipeline, params come from `save_hyperparameters()` via `MLFlowLogger` — Lightning automatically logs all `hparams` to MLflow when training starts.

### Tags

```python
mlflow.set_tag("dvc_commit", "fd2d31f")      # trainer.py:87
mlflow.set_tag("language", "vi")             # trainer.py:88
```

Tags are key-value strings. Unlike params, they can be updated after the run ends. Common uses:
- Linking to the Git/DVC commit that produced the run
- Marking runs by stage (`candidate`, `evaluated`, `promoted`)
- Adding CI job IDs or environment info

### Artifacts

```python
# Log a single file
mlflow.log_artifact("/tmp/mel.npy", artifact_path="val_mels/epoch_5")

# Log a directory (all files inside)
mlflow.log_artifacts("/tmp/report_dir/", artifact_path="evaluation")

# Log evaluation report
mlflow.log_artifact(str(json_path))    # reporter.py:86
mlflow.log_artifact(str(html_path))   # reporter.py:87
```

Artifacts are arbitrary files stored in the artifact store (S3, local filesystem, etc.). Unlike metrics and params (stored in a database), artifacts are stored as raw files. The `artifact_path` is a subdirectory within the run's artifact root.

---

## Module 4 — Logging PyTorch Models

### `mlflow.pytorch.log_model()`

```python
mlflow.pytorch.log_model(pl_model, artifact_path="matcha_tts")   # trainer.py:95
```

This does three things:
1. Serializes the model with `torch.save` (the underlying weights)
2. Saves a `MLmodel` metadata file describing the model's flavor and signature
3. Uploads both to the artifact store under `matcha_tts/`

The `MLmodel` file looks like:

```yaml
# MLmodel (auto-generated)
flavors:
  pytorch:
    model_data: data
    pytorch_version: 2.3.0
  python_function:
    loader_module: mlflow.pytorch
    python_version: 3.11.0
```

The **flavor** abstraction is what makes `mlflow.pytorch.load_model()` work: it reads the `MLmodel` file to know which loader to use. If the same model were logged as `mlflow.sklearn`, you'd use `mlflow.sklearn.load_model()`. The flavor system means you can write flavor-agnostic loading code with `mlflow.pyfunc.load_model()`.

### Loading a model back from a run

```python
model_uri = f"runs:/{run_id}/matcha_tts"
model = mlflow.pytorch.load_model(model_uri)
```

Or from the registry (Module 6):

```python
model_uri = "models:/matcha-tts-vi/Production"
model = mlflow.pytorch.load_model(model_uri)     # registry.py:62
```

---

## Module 5 — The Tracking Server and Tracking URI

### Where MLflow stores data

MLflow needs a **tracking server** to store run metadata (metrics, params, tags) and an **artifact store** for files. These can be the same location or different:

| Setup | Tracking backend | Artifact store | When to use |
|---|---|---|---|
| Local (default) | `./mlruns/` SQLite | `./mlruns/` filesystem | Local dev, one machine |
| Remote server | PostgreSQL/MySQL | S3 / GCS / Azure | Team, production |
| Databricks | Databricks | DBFS | Enterprise |

### Setting the tracking URI

```python
# In code
mlflow.set_tracking_uri("http://mlflow.internal:5000")

# In MLFlowLogger
MLFlowLogger(tracking_uri="http://mlflow.internal:5000")

# Via environment variable (works without code changes)
# export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
```

Our project reads it from Hydra config: `cfg.mlflow.tracking_uri`. This means different environments (dev/staging/prod) can point to different tracking servers without code changes.

### Starting the local MLflow UI

```bash
pip install mlflow
mlflow ui                          # opens http://localhost:5000
mlflow ui --port 5001              # custom port
mlflow server --host 0.0.0.0 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://my-bucket/mlflow
```

---

## Module 6 — The Model Registry

### Registered models and versions

The Model Registry is a central catalog of model versions with lifecycle stages. It is separate from runs — you explicitly register a model version from a run's artifact.

```
Registered Model: "matcha-tts-vi"
├── Version 1  →  run abc123  →  stage: Production   (deployed)
├── Version 2  →  run def456  →  stage: Archived     (old experiment)
└── Version 3  →  run ghi789  →  stage: Staging      (new candidate)
```

### Lifecycle stages

Our project uses four stages defined in `registry.py:12`:

```python
ModelStage = str  # "Staging" | "Shadow" | "Production" | "Archived"
```

| Stage | Meaning in our pipeline |
|---|---|
| `Staging` | Training finished, model not yet evaluated |
| `Shadow` | Evaluation passed absolute+relative gates, running shadow inference |
| `Production` | Shadow inference approved, actively serving traffic |
| `Archived` | Replaced by a newer Production version |

### `register_model()`

```python
def register_model(run_id, artifact_path, model_name, tags=None) -> str:   # registry.py:21
    client = get_client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags=tags or {},
    )
    return mv.version
```

Called in `run_training()` after the best checkpoint is logged:

```python
register_model(
    run_id=run_id,
    artifact_path="matcha_tts",
    model_name=f"matcha-tts-{cfg.data.language}",   # e.g. "matcha-tts-vi"
    tags={"dvc_commit": dvc_commit, "language": cfg.data.language},
)
```

`client.create_model_version()` links the run's artifact to the Model Registry. If the registered model `"matcha-tts-vi"` doesn't exist yet, MLflow creates it automatically.

### `transition_stage()`

```python
def transition_stage(model_name, version, stage) -> None:   # registry.py:40
    client = get_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),  # ← key detail
    )
```

`archive_existing_versions=True` when transitioning to Production means the previous Production version is automatically moved to Archived. There is always at most one Production version. For Staging and Shadow, old versions are not archived — multiple candidates can coexist.

### `get_latest_version()` and `load_model()`

```python
def get_latest_version(model_name, stage) -> str | None:    # registry.py:51
    versions = client.get_latest_versions(model_name, stages=[stage])
    return versions[0].version if versions else None

def load_model(model_name, stage) -> Any:                   # registry.py:59
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pytorch.load_model(model_uri)
```

The `models:/` URI scheme resolves the latest version in a given stage. When the Production model is promoted to a new version, any code using `models:/matcha-tts-vi/Production` automatically gets the new model on next load — no code change needed.

---

## Module 7 — The Evaluation Gate

### Two-tier quality check

The evaluation gate in `reporter.py` enforces two independent checks before a model can advance from Staging to Shadow:

**Tier 1 — Absolute floor:** hard minimums that any production-worthy model must meet, regardless of history.

**Tier 2 — Relative delta:** regression guard — the new model must not be significantly worse than the current Production baseline.

```python
def check_gates(metrics, absolute_floor, relative_delta, baseline_metrics):  # reporter.py:16
    failures = []

    # Tier 1: absolute floors
    floor_checks = [
        ("utmos_mean", metrics["utmos_mean"], absolute_floor["utmos_mean"], ">="),
        ("cer",        metrics["cer"],        absolute_floor["cer"],        "<="),
        ("stoi_mean",  metrics["stoi_mean"],  absolute_floor["stoi_mean"],  ">="),
        ("rtf_p95",    metrics["rtf_p95"],    absolute_floor["rtf_p95"],    "<="),
    ]
    for name, value, threshold, op in floor_checks:
        if op == ">=" and value < threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} < {threshold}")
        elif op == "<=" and value > threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} > {threshold}")

    # Tier 2: relative delta (only when baseline exists)
    if baseline_metrics:
        for key, max_delta in relative_delta.items():
            candidate = metrics.get(key)
            base = baseline_metrics.get(key)
            if candidate is None or base is None or base == 0:
                continue
            delta = (candidate - base) / abs(base)
            if delta >= max_delta:
                failures.append(f"REGRESSION: {key} delta={delta:.4f} > allowed {max_delta:.4f}")

    return len(failures) == 0, failures
```

### The gate thresholds from `params.yaml`

```yaml
evaluation:
  absolute_floor:
    utmos_mean: 3.0      # MOS score ≥ 3.0 (1-5 scale)
    cer: 0.20            # character error rate ≤ 20%
    stoi_mean: 0.60      # speech intelligibility ≥ 0.60
    rtf_p95: 0.20        # 95th percentile RTF ≤ 0.20 (5x real-time)
  relative_delta:
    utmos_mean: -0.05    # MOS must not drop more than 5% relative
    cer: 0.02            # CER must not increase more than 2pp relative
    stoi_mean: -0.03     # STOI must not drop more than 3% relative
    rtf_p95: 0.05        # RTF must not increase more than 5% relative
```

### What the metrics mean

| Metric | Full name | Range | Better |
|---|---|---|---|
| `utmos_mean` | UTMOS (MOS predictor) | 1–5 | Higher |
| `cer` | Character Error Rate (ASR transcription) | 0–1 | Lower |
| `stoi_mean` | Short-Time Objective Intelligibility | 0–1 | Higher |
| `rtf_p95` | Real-Time Factor at 95th percentile | >0 | Lower (< 1 = faster than real-time) |

### The full evaluation flow in `run_evaluation.py`

```python
# 1. Load candidate from Staging
model = load_model(model_name, stage="Staging")                # run_evaluation.py:34

# 2. Synthesize test set
synthesis_results = synthesize_test_set(...)                   # run_evaluation.py:48

# 3. Compute all metrics
all_metrics = {**utmos_metrics, **cer_metrics, **sig_metrics, **rtf_metrics}

# 4. Load Production baseline (for relative delta check)
baseline_model, baseline_version = load_baseline(model_name)  # run_evaluation.py:78
# baseline.py: get_latest_version(stage="Production") → load_model(stage="Production")

# 5. Run gate check + write report + log to MLflow
report = generate_report(                                      # run_evaluation.py:86
    metrics=all_metrics,
    baseline_metrics=baseline_metrics,
    ...
)
# Inside generate_report():
#   mlflow.log_metrics(metrics)           reporter.py:85
#   mlflow.log_artifact(json_path)        reporter.py:86
#   mlflow.log_artifact(html_path)        reporter.py:87

# 6. If gate passes → promote to Shadow
if not report["pass"]:
    sys.exit(1)                                                # run_evaluation.py:97

version = get_latest_version(model_name, stage="Staging")
transition_stage(model_name, version, stage="Shadow")          # run_evaluation.py:104
```

If the gate fails, the process exits with code 1 — DVC sees a non-zero exit code and marks the `evaluate` stage as failed, which blocks the `export` stage from running.

### First-run behavior (no Production baseline)

```python
def load_baseline(model_name):              # baseline.py:11
    version = get_latest_version(model_name, stage="Production")
    if version is None:
        logger.info("No Production baseline found — absolute floor only")
        return None, None
```

On the very first training run, there is no Production model to compare against. `baseline_metrics` is `None`, so the Tier 2 delta check is skipped. Only the absolute floor is enforced. This handles the cold-start case gracefully without special-casing in the gate logic.

---

## Module 8 — The MlflowClient API

### When to use the client vs the fluent API

```python
# Fluent API (mlflow.log_metric, mlflow.start_run, etc.)
# — requires an active run
# — simpler syntax
# — used inside training/evaluation code

# MlflowClient (mlflow.tracking.MlflowClient)
# — works without an active run
# — can query any run, any experiment
# — used for management operations (registry, searching runs)
```

Our `registry.py` uses `MlflowClient` for all Model Registry operations because registry management doesn't happen inside a training run context.

### Common `MlflowClient` operations

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# List all registered models
models = client.search_registered_models()

# Get all versions of a model
versions = client.search_model_versions(f"name='matcha-tts-vi'")
for v in versions:
    print(v.version, v.current_stage, v.run_id)

# Get a specific run's metrics
run = client.get_run("abc123")
print(run.data.metrics)   # {"val/loss": 0.42, "train/loss": 0.31, ...}
print(run.data.params)    # {"learning_rate": "0.0001", ...}
print(run.data.tags)      # {"dvc_commit": "fd2d31f", ...}

# Search for runs with a filter
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_loss < 0.5 AND tags.language = 'vi'",
    order_by=["metrics.val_loss ASC"],
    max_results=10,
)

# Delete a run
client.delete_run("abc123")
```

### Downloading artifacts

```python
# Download a specific artifact to a local temp dir
local_path = client.download_artifacts("abc123", "evaluation_report.json", "/tmp/")

# Download everything from a run
local_dir = mlflow.artifacts.download_artifacts(f"runs:/abc123/")
```

---

## Module 9 — The Stage Transition Lifecycle in Practice

### Full model lifecycle in our pipeline

```
Training run completes
    ↓
mlflow.pytorch.log_model() → artifact stored
    ↓
register_model() → version created, stage = "None"
    ↓
transition_stage("Staging") → version is now a candidate
    ↓
dvc repro evaluate
    ↓ [gate check]
    ├── FAIL → sys.exit(1), DVC blocks export, Slack alert
    └── PASS → transition_stage("Shadow")
                    ↓
               Shadow inference (1000 requests compared vs Production)
                    ↓
               Manual approval (or auto if metrics within delta)
                    ↓
               transition_stage("Production")
               (archive_existing_versions=True → old Production → "Archived")
```

### Rollback

If a Production model misbehaves in production, rollback is one call:

```python
# Restore previous Production version
previous_version = "1"   # from audit log
transition_stage("matcha-tts-vi", previous_version, "Production")
# archive_existing_versions=True archives the bad version automatically
```

The serving code uses `models:/matcha-tts-vi/Production` — after this call, it loads the previous model on the next inference engine restart with no code change.

---

## Module 10 — Common Pitfalls

### Pitfall 1: Starting a new run instead of resuming

```python
# BAD — creates a second run, metrics and model in different runs
mlflow_logger = MLFlowLogger(...)      # creates run "abc123"
with mlflow.start_run():               # creates NEW run "xyz999"
    mlflow.pytorch.log_model(...)      # goes to "xyz999", not "abc123"

# GOOD — resume the existing run
run_id = mlflow_logger.run_id
with mlflow.start_run(run_id=run_id):  # resumes "abc123"
    mlflow.pytorch.log_model(...)      # goes to "abc123"
```

### Pitfall 2: Logging tensors as metrics

```python
# BAD — TypeError: unhashable type 'Tensor'
mlflow.log_metric("loss", loss_tensor)

# GOOD — convert to Python float first
mlflow.log_metric("loss", loss_tensor.item())
# or
metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}  # callbacks.py:51
mlflow.log_metrics(metrics)
```

### Pitfall 3: Calling `mlflow.*` outside an active run

```python
# BAD — raises MlflowException
def my_callback(...):
    mlflow.log_artifact(...)   # no guard

# GOOD — guard with active_run()
def on_validation_epoch_end(self, trainer, pl_module):
    if not mlflow.active_run():       # callbacks.py:29
        return
    mlflow.log_artifact(...)
```

### Pitfall 4: Using the `"None"` stage to mean "unregistered"

When `create_model_version()` is called, the new version has stage `"None"` (the string, not Python `None`). If you call `get_latest_versions(model_name, stages=["None"])`, you get unreviewed versions. Don't confuse this with Python `None` returned when no version exists.

```python
# stages=["None"] → versions with no stage assigned
# stages=["Staging"] → versions explicitly moved to Staging
versions = client.get_latest_versions("matcha-tts-vi", stages=["None"])
```

### Pitfall 5: The artifact store is not the tracking database

Metrics, params, and tags are stored in the tracking database (SQLite/PostgreSQL). Artifacts are stored in the artifact store (filesystem/S3). These are different backends configured separately. If the artifact store is S3 but the tracking DB is local SQLite, moving to a new machine:
- `mlflow.get_metric_history()` still works (SQLite is accessible)
- `mlflow.artifacts.download_artifacts()` fails (S3 credentials needed)

Always configure both backends in sync.

---

## Module 11 — Mini Practice Project: `tracklab`

An independent mini-project that applies the full MLflow loop — experiment tracking, artifact logging, model registry, stage transitions — to a simple sklearn classification problem. No TTS code.

**Goal:** Train three logistic regression models with different regularization strengths. Track all runs in MLflow. Register the best, move it through Staging → Production. Write a loader that always fetches the Production version.

### Project structure

```
tracklab/
├── train.py       # train + log + register
├── evaluate.py    # load Staging, check accuracy gate, promote
└── serve.py       # load Production by URI, predict
```

### `train.py`

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-logreg")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for C in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"logreg-C={C}"):
        # Params
        mlflow.log_param("C", C)
        mlflow.log_param("solver", "lbfgs")

        # Train
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)

        # Metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        # Tags
        mlflow.set_tag("candidate", "true")
        mlflow.set_tag("n_train", len(X_train))

        # Log model to registry
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "iris-classifier")

        print(f"C={C:5.2f}  acc={acc:.4f}  f1={f1:.4f}  run_id={run_id[:8]}")
```

### `evaluate.py`

```python
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ACCURACY_FLOOR = 0.85   # absolute gate

# Get all unreviewed versions ("None" stage = newly registered)
versions = client.search_model_versions("name='iris-classifier'")
candidates = [v for v in versions if v.current_stage == "None"]

best_version = None
best_acc = 0.0

for v in candidates:
    # Move to Staging
    client.transition_model_version_stage("iris-classifier", v.version, "Staging")

    # Load and evaluate
    model = mlflow.sklearn.load_model(f"models:/iris-classifier/Staging")
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Version {v.version}: accuracy={acc:.4f}", end="")

    if acc < ACCURACY_FLOOR:
        print(f"  → FAIL (floor={ACCURACY_FLOOR})")
        client.transition_model_version_stage("iris-classifier", v.version, "Archived")
    elif acc > best_acc:
        best_acc = acc
        if best_version:
            # Archive the previous best
            client.transition_model_version_stage("iris-classifier", best_version, "Archived")
        best_version = v.version
        print(f"  → new best")
    else:
        print(f"  → PASS but not best")
        client.transition_model_version_stage("iris-classifier", v.version, "Archived")

if best_version:
    client.transition_model_version_stage(
        "iris-classifier", best_version, "Production",
        archive_existing_versions=True   # archive previous Production
    )
    print(f"\nPromoted version {best_version} (acc={best_acc:.4f}) to Production")
else:
    print("\nNo version passed the gate — Production unchanged")
```

### `serve.py`

```python
import mlflow
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Always loads Production — no version number hardcoded
model = mlflow.sklearn.load_model("models:/iris-classifier/Production")
print(f"Loaded Production model: {model}")

# Predict on a new sample
X_new = np.random.randn(1, 20)
pred = model.predict(X_new)
prob = model.predict_proba(X_new)
print(f"Prediction: {pred[0]}, confidence: {max(prob[0]):.3f}")
```

### Running the project

```bash
pip install mlflow scikit-learn
python train.py     # trains 4 models, registers all
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open UI at localhost:5000
python evaluate.py  # gates and promotes best model
python serve.py     # loads Production
```

### Exercises

1. **Inspect the UI** — open `localhost:5000`. Find the experiment, compare all 4 runs on the "accuracy" metric. Sort by accuracy. Notice how MLflow automatically tracks `C` as a param axis in the parallel coordinates view.

2. **Change the gate threshold** — set `ACCURACY_FLOOR = 0.99` in `evaluate.py`. Re-run. Observe that no version passes. The Production model should remain whatever was promoted before (or "no Production" if this is first run).

3. **Add a relative delta check** — modify `evaluate.py` to load the current Production model's accuracy from its run's metrics (`client.get_run(v.run_id).data.metrics["accuracy"]`) and reject any candidate that is more than 2% worse.

4. **Log an artifact** — after training, generate a confusion matrix plot (`matplotlib`) and log it as an artifact. View it in the MLflow UI under Artifacts.

5. **Search runs programmatically** — use `client.search_runs()` with `filter_string="metrics.accuracy > 0.9 AND params.C > 0.5"`. Print the run IDs and accuracy values.

6. **Rollback** — after promotion, manually transition the previous version back to Production and the current Production to Archived. Verify `serve.py` now loads the old model.

---

## Key Takeaways

- MLflow Tracking stores metrics (time series), params (once per run), tags (mutable), and artifacts (files). Each belongs to a Run inside an Experiment.
- `mlflow.start_run(run_id=existing_id)` resumes an existing run — use this to keep Lightning's `MLFlowLogger` metrics and manually-logged artifacts in the same run.
- Guard all `mlflow.*` calls with `if mlflow.active_run()` so callbacks work without MLflow enabled.
- The Model Registry is separate from Tracking. A version is created by `create_model_version()` and transitions through stages: `None → Staging → Shadow → Production → Archived`.
- `archive_existing_versions=True` when promoting to Production ensures exactly one Production version at all times.
- `models:/model-name/Production` always resolves to the current Production version — no version numbers in serving code.
- The evaluation gate is two-tier: absolute floors (hard minimums) + relative delta (no regression vs current Production). The first training run skips the delta check because there is no baseline.
- A failed gate causes `sys.exit(1)` which makes DVC mark the `evaluate` stage as failed, blocking the downstream `export` stage automatically.