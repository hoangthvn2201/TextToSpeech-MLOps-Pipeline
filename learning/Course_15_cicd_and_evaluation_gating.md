# Course 15: CI/CD & Evaluation Gating

**Series:** MLOps for TTS  
**Prerequisites:** Courses 10 (DVC), 11 (MLflow), 12 (Prefect), 13 (Hydra+Pydantic), 14 (ONNX+FastAPI)  
**What you will build:** A full understanding of how this pipeline enforces quality before any model reaches production — from the moment a PR is opened to the moment a promoted model triggers a re-deploy.

---

## Module 1 — Why CI/CD Is Different for ML

Traditional software CI answers one question: *does the code work?* ML CI must answer a second question: *does the model perform well enough?* These are fundamentally different failure modes.

| Failure type | Caught by |
|---|---|
| Syntax error, import crash | Unit tests |
| Type mismatch, wrong shape | Type checker + unit tests |
| DVC stage broken | `dvc dag` lint |
| Model regresses on UTMOS | Evaluation gate |
| Latency regression (RTF) | Evaluation gate |
| Unsafe Shadow→Production promotion | Manual approval gate |

The pipeline uses **three separate workflows** for three separate concerns:

```
ci.yml       — code quality (every PR, every push to develop)
pipeline.yml — DVC pipeline + evaluation gate (every push to main)
promote.yml  — model promotion (triggered by pipeline success, requires human approval)
```

**Key insight:** these three workflows have completely different runners, timeouts, and environments. Confusing them is a common MLOps mistake.

---

## Module 2 — GitHub Actions Fundamentals

### Anatomy of a workflow

```yaml
name: My Workflow        # shown in GitHub UI

on:                      # trigger events
  push:
    branches: [main]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:     # manual trigger with optional inputs
    inputs:
      flag:
        type: boolean
        default: false

jobs:
  job-name:
    runs-on: ubuntu-latest          # runner environment
    environment: production         # GitHub Environment (can require manual approval)
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4   # reusable Action from marketplace
      - name: Do something
        run: echo "hello"           # shell command
        env:
          MY_SECRET: ${{ secrets.MY_SECRET }}   # secret injection
```

### Key concepts

**`secrets`** — stored encrypted in GitHub repo/org settings. Never appear in logs. Injected via `${{ secrets.NAME }}`. In this project: `DVC_S3_KEY`, `DVC_S3_SECRET`, `MLFLOW_TRACKING_URI`, `DISPATCH_TOKEN`.

**`environment`** — a named deployment environment in GitHub settings. Can require **manual reviewer approval** before the job runs. The project uses `environment: production` in both `pipeline.yml` and `promote.yml`.

**`workflow_run`** — a trigger that fires when another named workflow completes. Used in `promote.yml` to chain after `pipeline.yml` succeeds (see Module 8).

**`self-hosted` runner** — a machine you register with GitHub that runs workflow jobs. The pipeline workflow needs a GPU, so it uses `runs-on: [self-hosted, gpu]` with a label filter. GitHub-hosted runners (`ubuntu-latest`) have no GPU.

---

## Module 3 — The CI Workflow (`ci.yml`)

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every PR targeting `main` or `develop`, and every push to `develop`. It has four parallel jobs.

```yaml
on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [develop]
```

### Job 1: `lint-and-typecheck`

```yaml
- name: Install code quality tools
  run: pip install ruff black mypy pydantic>=2.7

- name: Ruff lint
  run: ruff check src/ tests/ scripts/ pipelines/

- name: Black format check
  run: black --check src/ tests/ scripts/ pipelines/

- name: Mypy type check
  run: mypy src/tts_pipeline/ --ignore-missing-imports
```

**Why these three?**
- **ruff** — fast linter (replaces flake8, isort, pyupgrade). Catches unused imports, undefined names, style issues.
- **black --check** — format verifier. `--check` exits non-zero if any file *would* be reformatted (doesn't actually modify files in CI).
- **mypy** — static type checker. Catches `Optional` not handled, wrong argument types, missing attributes. Uses `--ignore-missing-imports` because not all deps have stubs.

**Note:** The Makefile local equivalent is `make lint` + `make typecheck`. Running `make format` (which uses `black` without `--check`) fixes in-place locally, then CI just verifies.

### Job 2: `unit-tests`

```yaml
- name: Run unit tests
  run: pytest tests/unit/ -v --cov=src/tts_pipeline --cov-report=xml -m "not slow"

- name: Upload coverage
  uses: codecov/codecov-action@v4
  with:
    files: coverage.xml
    fail_ci_if_error: false
```

`-m "not slow"` filters out tests marked with `@pytest.mark.slow`. These would be model inference tests that are too slow for CI on a CPU runner. Coverage is uploaded to Codecov but `fail_ci_if_error: false` means a Codecov outage doesn't break CI.

### Job 3: `integration-smoke`

```yaml
- name: Run integration smoke tests (no GPU)
  run: pytest tests/integration/ -v -m "not slow and not gpu" --timeout=300
```

Integration tests exercise real component interactions (e.g., tokenizer + phonemizer together, config loading end-to-end) but skip anything needing GPU or long compute time.

### Job 4: `dvc-lint`

```yaml
- name: Validate dvc.yaml
  run: dvc dag --dot > /dev/null

- name: Check params diff
  run: dvc params diff --all 2>/dev/null || true
```

`dvc dag --dot` validates that `dvc.yaml` is syntactically correct and the DAG is acyclic — it will fail if a stage references a non-existent output or has a circular dependency. The `|| true` on `params diff` makes it non-blocking: it reports drift but doesn't fail CI (params may legitimately differ from what's tracked).

---

## Module 4 — Code Quality in Depth

### ruff configuration

Ruff is configured via `pyproject.toml`. Typical rules enabled for an ML project:

```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "UP", "B"]
# E = pycodestyle errors, F = pyflakes, I = isort, UP = pyupgrade, B = bugbear

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]   # allow assert in tests
```

### mypy strictness levels

The Makefile uses `--strict` locally (`make typecheck`), but CI uses `--ignore-missing-imports`. This is a deliberate asymmetry: `--strict` catches the most bugs but requires type stubs for all deps; CI uses the lighter flag to avoid false positives from untyped third-party libraries.

### The black/ruff workflow

```
Developer machine:   make format    → black rewrites, ruff --fix rewrites
CI:                  black --check  → exits 1 if any file differs
                     ruff check     → exits 1 if any lint violation
```

The developer workflow is **fix-then-commit**; CI just **verifies**. This is why `make format` is in the Makefile but CI never calls it.

---

## Module 5 — Test Markers and Test Strategy

Pytest markers are used to tag tests so CI can run the right subset:

```python
import pytest

@pytest.mark.slow          # takes > 30s
@pytest.mark.gpu           # requires CUDA
def test_full_synthesis():
    ...
```

Registration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "gpu: marks tests as requiring GPU",
]
```

### Test taxonomy

| Test type | Location | Marker | What it tests |
|---|---|---|---|
| Unit | `tests/unit/` | none | Single function/class in isolation |
| Integration smoke | `tests/integration/` | `not slow and not gpu` | Component interactions, CPU only |
| Slow integration | `tests/integration/` | `slow` | Long compute, run manually or on GPU runner |
| GPU tests | `tests/integration/` | `gpu` | CUDA-dependent behavior |

CI on `ubuntu-latest` runs unit + integration smoke. GPU tests run only on the self-hosted GPU runner (pipeline workflow).

---

## Module 6 — The Pipeline Workflow (`pipeline.yml`)

[.github/workflows/pipeline.yml](.github/workflows/pipeline.yml) runs on every push to `main` and on manual `workflow_dispatch`.

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      force_retrain:
        description: "Force re-training even if DVC cache is valid"
        type: boolean
        default: false
```

### Runner setup

```yaml
runs-on: [self-hosted, gpu]
environment: production
timeout-minutes: 480   # 8 hours max
```

`[self-hosted, gpu]` is a **label filter** — GitHub matches runners that have *both* the `self-hosted` label and the `gpu` label. You register a GPU machine with these labels in `Settings → Actions → Runners`.

### DVC remote authentication

```yaml
- name: Configure DVC remote
  run: |
    dvc remote modify storage access_key_id ${{ secrets.DVC_S3_KEY }}
    dvc remote modify storage secret_access_key ${{ secrets.DVC_S3_SECRET }}
```

Credentials are injected at runtime from GitHub Secrets into `dvc config`. They are never stored in `.dvc/config` (which is committed).

### Conditional force-retrain

```yaml
- name: Run DVC pipeline
  run: |
    if [ "${{ inputs.force_retrain }}" = "true" ]; then
      dvc repro --force
    else
      dvc repro
    fi
```

`dvc repro` is cache-aware (only reruns changed stages). `dvc repro --force` bypasses cache and reruns everything — useful after a bug fix that needs a clean re-run.

### Committing `dvc.lock` back

```yaml
- name: Commit updated dvc.lock
  run: |
    git config user.email "ci@pipeline"
    git config user.name "Pipeline Bot"
    git add dvc.lock reports/latest/evaluation_report.json
    git diff --cached --quiet || git commit -m "ci: update dvc.lock after pipeline run [skip ci]"
    git push
```

After a successful run, `dvc.lock` is updated with new hashes. Committing it back ensures reproducibility: anyone can `git checkout` this commit and `dvc repro` to get byte-identical outputs. The `[skip ci]` tag in the commit message prevents triggering another pipeline run.

---

## Module 7 — The Evaluation Gate in CI

The most important step in `pipeline.yml` is the evaluation gate:

```yaml
- name: Check evaluation gate
  run: |
    python - <<'EOF'
    import json, sys
    with open("reports/latest/evaluation_report.json") as f:
        report = json.load(f)
    if not report.get("pass"):
        print("EVALUATION FAILED. Gate failures:")
        for f in report.get("failures", []):
            print(f"  - {f}")
        sys.exit(1)
    print("All evaluation gates passed.")
    EOF
```

This is an **inline Python script** executed in a bash heredoc (`<<'EOF'`). The single quotes around `'EOF'` prevent shell variable expansion inside the script. The pattern `sys.exit(1)` causes the CI step to fail with a non-zero exit code, which fails the entire job and blocks the `promote.yml` workflow from triggering.

### What is `evaluation_report.json`?

Generated by `generate_report()` in [src/tts_pipeline/evaluation/reporter.py](src/tts_pipeline/evaluation/reporter.py):

```python
report: dict[str, Any] = {
    "pass": passed,                    # ← CI reads this
    "failures": failures,              # ← CI prints these on failure
    "metrics": metrics,
    "baseline_metrics": baseline_metrics,
    "model_info": model_info or {},
    "n_synthesized": len(synthesis_results),
}
```

The `"pass"` boolean is the single source of truth for the gate. CI doesn't re-evaluate thresholds — it trusts the report generated by the DVC evaluation stage.

---

## Module 8 — Two-Tier Quality Gate Deep Dive

The gate logic lives in [src/tts_pipeline/evaluation/reporter.py:16](src/tts_pipeline/evaluation/reporter.py#L16):

```python
def check_gates(
    metrics: dict[str, float],
    absolute_floor: dict[str, float],
    relative_delta: dict[str, float],
    baseline_metrics: dict[str, float] | None,
) -> tuple[bool, list[str]]:
```

### Tier 1: Absolute floor

```python
floor_checks = [
    ("utmos_mean", metrics.get("utmos_mean", 0.0), absolute_floor.get("utmos_mean", 3.0), ">="),
    ("cer",        metrics.get("cer", 1.0),        absolute_floor.get("cer", 0.20),        "<="),
    ("stoi_mean",  metrics.get("stoi_mean", 0.0),  absolute_floor.get("stoi_mean", 0.60),  ">="),
    ("rtf_p95",    metrics.get("rtf_p95", 1.0),    absolute_floor.get("rtf_p95", 0.20),    "<="),
]
```

Each metric has a direction (`>=` means "must be at least X", `<=` means "must be no worse than X"). From [configs/evaluation/metrics.yaml](configs/evaluation/metrics.yaml):

```yaml
absolute_floor:
  utmos_mean: 3.0      # audio quality score: must be ≥ 3.0 / 5.0
  cer: 0.20            # character error rate: must be ≤ 20%
  stoi_mean: 0.60      # speech intelligibility: must be ≥ 0.60
  rtf_p95: 0.20        # real-time factor p95: must be ≤ 0.20
```

**Defaults as safety net:** `metrics.get("utmos_mean", 0.0)` — if the metric is missing (e.g., `speechmos` failed to install), it defaults to the worst possible value, which fails the floor check. This prevents silent skips from causing a bad model to pass.

### Tier 2: Relative delta (regression check)

```python
if baseline_metrics:
    for key, max_delta in relative_delta.items():
        candidate = metrics.get(key)
        base = baseline_metrics.get(key)
        if candidate is None or base is None or base == 0:
            continue
        delta = (candidate - base) / abs(base)   # relative change
        if delta < max_delta:                     # improvement or within tolerance
            continue
        failures.append(f"REGRESSION: {key} delta={delta:.4f} > allowed {max_delta:.4f} vs baseline {base:.4f}")
```

From [configs/evaluation/metrics.yaml](configs/evaluation/metrics.yaml):

```yaml
relative_delta:
  utmos_mean: -0.05    # allow up to -5% UTMOS regression
  cer: 0.02            # allow up to +2pp CER regression
  stoi_mean: -0.03
  rtf_p95: 0.05        # allow up to +5% latency regression
```

**Cold-start:** when `baseline_metrics` is `None` (no Production model exists yet), the relative delta tier is skipped entirely — only absolute floor applies. This is the right behavior for the first model ever promoted.

**Why `delta < max_delta` means pass:**
- For `utmos_mean`, `max_delta = -0.05`. If candidate improved (delta > 0), it passes. If it regressed by 3% (delta = -0.03 > -0.05), it passes. If it regressed by 8% (delta = -0.08 < -0.05), it fails.
- For `cer`, `max_delta = 0.02`. CER going from 0.10 to 0.11 is delta = +0.10 (10%), which fails (+0.10 > 0.02).

---

## Module 9 — Evaluation Metrics

### UTMOS — Mean Opinion Score proxy

[src/tts_pipeline/evaluation/metrics/utmos.py](src/tts_pipeline/evaluation/metrics/utmos.py) uses `speechmos.dnsmos.run()`, which is Microsoft's DNSMOS P.835 model — a neural network trained to predict human MOS ratings. No human listeners required.

```python
result = speechmos.dnsmos.run(str(p), sr=sample_rate)
scores.append(float(result.get("OVRL_raw", 0.0)))
```

Scale: 1–5. Absolute floor of 3.0 means "clearly intelligible speech with acceptable quality."

### CER — Character Error Rate via Whisper

[src/tts_pipeline/evaluation/metrics/intelligibility.py](src/tts_pipeline/evaluation/metrics/intelligibility.py) uses OpenAI Whisper (ASR) to transcribe the synthesized audio, then computes CER against the original text using `jiwer`:

```python
model = whisper.load_model(whisper_model_size)   # "small" from config
result = model.transcribe(str(p), language=language)
cer_score = float(jiwer_cer(list(refs), list(hyps)))
```

The eval config uses `whisper_model: small` (faster than `medium`/`large`). Lower CER = better intelligibility. Floor of 0.20 means fewer than 1 in 5 characters must be wrong.

### STOI — Short-Time Objective Intelligibility

Computed in `src/tts_pipeline/evaluation/metrics/pesq_stoi.py`. Requires the reference audio (ground-truth recording) and hypothesis audio. Measures correlation in short-time spectral envelopes. Range 0–1.

### RTF — Real-Time Factor at p95

[src/tts_pipeline/evaluation/metrics/rtf.py](src/tts_pipeline/evaluation/metrics/rtf.py). `rtf = synthesis_time / audio_duration`. RTF < 1.0 means faster than real-time. The gate uses **p95** (95th percentile across all test utterances), not mean — tail latency matters for serving. Floor of 0.20 means the 95th-percentile utterance must synthesize in ≤ 20% of its audio duration.

### Where metrics are computed

[src/tts_pipeline/evaluation/synthesizer.py](src/tts_pipeline/evaluation/synthesizer.py) generates WAV files for all test-set utterances, then `scripts/run_evaluation.py` calls the metric functions, passes results to `generate_report()`, and the report is written to `reports/latest/evaluation_report.json`.

---

## Module 10 — The Promote Workflow (`promote.yml`)

[.github/workflows/promote.yml](.github/workflows/promote.yml) handles the final stage: Shadow→Production.

### Trigger

```yaml
on:
  workflow_run:
    workflows: ["Pipeline — Full DVC Run"]
    types: [completed]
```

`workflow_run` fires *after* the named workflow completes. The `if` guard on the job ensures it only runs when the pipeline *succeeded*:

```yaml
if: ${{ github.event.workflow_run.conclusion == 'success' }}
```

### Manual approval gate

```yaml
environment: production
```

When a GitHub Environment named `production` has required reviewers configured, this job **pauses** and waits for a human to approve in the GitHub UI before the steps execute. This is the human-in-the-loop gate between automation and production.

### The promotion step

```python
shadow_versions = client.get_latest_versions(model_name, stages=["Shadow"])
if not shadow_versions:
    print(f"No Shadow model found for {model_name}")
    sys.exit(1)

version = shadow_versions[0].version
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True,   # archives old Production
)
```

This mirrors what `transition_stage()` in `src/tts_pipeline/utils/registry.py` does, but directly from CI without importing the project package (avoids installing all training deps in the promote job).

### Repository dispatch (re-deploy trigger)

```yaml
- name: Trigger serving re-deploy
  uses: peter-evans/repository-dispatch@v3
  with:
    token: ${{ secrets.DISPATCH_TOKEN }}
    event-type: model-promoted
    client-payload: '{"model": "matcha-tts", "stage": "Production"}'
```

`repository-dispatch` sends a custom webhook event to the repository. A separate workflow (not in this repo — would be the serving infra repo) can listen for `event-type: model-promoted` and trigger a container re-deploy. This is how the promote workflow decouples from the serving infrastructure.

---

## Module 11 — Docker for ML Serving

### `Dockerfile.serve` analysis

[docker/Dockerfile.serve](docker/Dockerfile.serve):

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*   # ← clean apt cache in same RUN layer

COPY requirements/serve.txt requirements/serve.txt
RUN pip install --no-cache-dir -r requirements/serve.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serve]" --no-deps
# ↑ --no-deps because serve.txt already installed all deps

COPY configs/ configs/

ENV BUNDLE_PATH=/app/artifacts/export
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "tts_pipeline.serving.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

**Key decisions:**
- `python:3.11-slim` not `python:3.11` — no build tools, no extras (~150MB saved)
- Only `libsndfile1` — no `espeak-ng` (serving doesn't need G2P; phonemes come pre-encoded in requests, or the ONNX engine handles them via the tokenizer loaded from the bundle)
- `--no-cache-dir` in pip — reduces image size
- Model bundle is **volume-mounted at runtime**, not baked into the image: `docker run -v ./artifacts/export:/app/artifacts/export`. This means the same image works with any model version.
- `--factory` flag for uvicorn — tells uvicorn to call `create_app()` to get the ASGI app instance (because it's a factory function, not a module-level `app`)

### `docker-compose.yml` service mesh

[docker/docker-compose.yml](docker/docker-compose.yml) wires together three services:

```
mlflow  (port 5000) — tracking server with SQLite backend
prefect (port 4200) — workflow orchestration UI
api     (port 8080) — TTS serving API
```

```yaml
api:
  depends_on:
    - mlflow          # starts after mlflow
  volumes:
    - ../artifacts/export:/app/artifacts/export:ro   # read-only mount
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 15s
    timeout: 5s
    retries: 3
```

The `healthcheck` uses the `/health` endpoint (always returns 200) for container readiness. Docker restarts the container if health checks fail 3 times in a row.

---

## Module 12 — Makefile as Developer Interface

The [Makefile](Makefile) is the single entry point for all common operations. This matters: it means CI and developers use the same commands, reducing "works on my machine" issues.

### Target categories

```makefile
# Code quality
lint:        ruff check src/ tests/ scripts/ pipelines/
format:      black + ruff --fix (modifies files, use locally)
typecheck:   mypy src/tts_pipeline/ --strict

# Tests
test:        all tests with coverage
test-unit:   pytest tests/unit/ only
test-integration: pytest tests/integration/ -m "not slow"

# Pipeline
data:        dvc repro validate        # up to data validation stage
train:       dvc repro train train_vocoder
evaluate:    dvc repro evaluate
export:      dvc repro export
pipeline:    dvc repro                  # full pipeline

# Docker
docker-train:  build training image
docker-serve:  build serving image
docker-up:     docker compose up -d
docker-down:   docker compose down

# DVC
dvc-push:    push all outputs to remote
dvc-pull:    pull all outputs from remote

# MLflow
mlflow-ui:   mlflow ui --host 0.0.0.0 --port 5000
```

**Why this structure matters for CI:** CI calls `pip install -e ".[dev]"` then `pytest tests/unit/ -v ...` directly (not via Make), because GitHub Actions steps are already shell commands. But Make provides the same commands locally, so developers can `make test-unit` and know they're running the same thing CI runs.

---

## Module 13 — Workflow Chain Summary

The complete automation chain when a developer pushes to `main`:

```
developer: git push origin main
      │
      ▼
[GitHub: push event to main]
      │
      ├──► ci.yml runs in parallel (lint, typecheck, tests, dvc-lint)
      │         └── if any job fails: PR/branch blocked
      │
      └──► pipeline.yml starts
                │
                ├── checkout + install deps
                ├── configure DVC remote (secrets)
                ├── dvc pull data/raw --run-cache
                ├── dvc repro  (may use cache for unchanged stages)
                ├── dvc push   (upload new outputs)
                ├── CHECK evaluation_report.json
                │       └── sys.exit(1) if "pass" is false → job fails
                │
                ├── git commit dvc.lock + report [skip ci]
                └── git push
                      │
                      ▼
              [workflow_run event: pipeline succeeded]
                      │
                      ▼
              promote.yml starts
                      │
                      ├── CHECK evaluation_report.json again
                      ├── WAIT for human approval (GitHub Environment gate)
                      ├── get Shadow model version from MLflow Registry
                      ├── transition Shadow → Production (archive old)
                      └── repository-dispatch: model-promoted event
                                │
                                ▼
                        [serving infra re-deploys with new Production model]
```

---

## Mini Project: `gatekitchen`

Build a minimal CI/CD simulation with a two-tier evaluation gate, using a toy regression problem. This is **standalone** — no TTS code.

### Project structure

```
gatekitchen/
├── src/
│   └── gatekitchen/
│       ├── train.py        # train a sklearn LinearRegression
│       ├── evaluate.py     # compute metrics + run gate
│       ├── reporter.py     # check_gates() + generate_report()
│       └── metrics.py      # rmse, mae, r2, latency_p95
├── configs/
│   ├── thresholds.yaml     # absolute floor + relative delta
│   └── experiment.yaml     # model hyperparams
├── reports/                # generated evaluation_report.json
├── models/                 # saved .pkl files
├── tests/
│   ├── test_reporter.py    # unit tests for check_gates()
│   └── test_metrics.py     # unit tests for metric functions
├── Makefile
└── .github/
    └── workflows/
        └── ci.yml
```

### Step 1 — implement `check_gates()`

```python
# src/gatekitchen/reporter.py
from __future__ import annotations
import json
from pathlib import Path

def check_gates(
    metrics: dict[str, float],
    absolute_floor: dict[str, float],
    relative_delta: dict[str, float],
    baseline_metrics: dict[str, float] | None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    # Tier 1: absolute floor
    floor_checks = [
        ("r2",          metrics.get("r2", -1.0),         absolute_floor.get("r2", 0.80),       ">="),
        ("rmse",        metrics.get("rmse", 1e9),         absolute_floor.get("rmse", 50.0),     "<="),
        ("latency_p95", metrics.get("latency_p95", 1e9),  absolute_floor.get("latency_p95", 0.05), "<="),
    ]
    for name, value, threshold, op in floor_checks:
        if op == ">=" and value < threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} < floor {threshold}")
        elif op == "<=" and value > threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} > floor {threshold}")

    # Tier 2: relative delta (skip if no baseline)
    if baseline_metrics:
        for key, max_delta in relative_delta.items():
            candidate = metrics.get(key)
            base = baseline_metrics.get(key)
            if candidate is None or base is None or base == 0:
                continue
            delta = (candidate - base) / abs(base)
            if delta >= max_delta:
                failures.append(
                    f"REGRESSION: {key} delta={delta:.4f} > allowed {max_delta:.4f} "
                    f"(candidate={candidate:.4f}, baseline={base:.4f})"
                )

    return len(failures) == 0, failures


def generate_report(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    absolute_floor: dict[str, float],
    relative_delta: dict[str, float],
    report_dir: Path,
) -> dict:
    passed, failures = check_gates(metrics, absolute_floor, relative_delta, baseline_metrics)
    report = {
        "pass": passed,
        "failures": failures,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2)
    )
    status = "PASSED" if passed else "FAILED"
    print(f"Gate: {status}")
    for f in failures:
        print(f"  - {f}")
    return report
```

### Step 2 — metrics

```python
# src/gatekitchen/metrics.py
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def compute_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_timing_runs: int = 100,
) -> dict[str, float]:
    y_pred = model.predict(X_test)

    # Predictive quality
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    # Latency: time predict() on single samples, take p95
    latencies = []
    for i in range(n_timing_runs):
        idx = i % len(X_test)
        t0 = time.perf_counter()
        model.predict(X_test[idx : idx + 1])
        latencies.append(time.perf_counter() - t0)

    latency_p95 = float(np.percentile(latencies, 95))

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "latency_p95": latency_p95,
    }
```

### Step 3 — training script

```python
# src/gatekitchen/train.py
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X, y = make_regression(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        noise=cfg["noise"],
        random_state=cfg["random_state"],
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=cfg["alpha"])
    model.fit(X_train, y_train)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save test split for evaluation
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)
    print(f"Saved model to {out / 'model.pkl'}")

if __name__ == "__main__":
    main()
```

### Step 4 — evaluation + gate script

```python
# src/gatekitchen/evaluate.py
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

from gatekitchen.metrics import compute_metrics
from gatekitchen.reporter import generate_report


def load_baseline_metrics(baseline_path: Path | None) -> dict | None:
    if baseline_path is None or not baseline_path.exists():
        return None
    with open(baseline_path) as f:
        return json.load(f).get("metrics")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--report-dir", default="reports/latest")
    parser.add_argument("--baseline-report", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    X_test = np.load(model_dir / "X_test.npy")
    y_test = np.load(model_dir / "y_test.npy")

    with open(args.thresholds) as f:
        thresholds = yaml.safe_load(f)

    metrics = compute_metrics(model, X_test, y_test)
    print("Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

    baseline_path = Path(args.baseline_report) if args.baseline_report else None
    baseline_metrics = load_baseline_metrics(baseline_path)

    report = generate_report(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        absolute_floor=thresholds["absolute_floor"],
        relative_delta=thresholds["relative_delta"],
        report_dir=Path(args.report_dir),
    )

    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
```

### Step 5 — configs

```yaml
# configs/thresholds.yaml
absolute_floor:
  r2: 0.80
  rmse: 50.0
  latency_p95: 0.05

relative_delta:
  r2: -0.05         # allow up to -5% R² regression
  rmse: 0.10        # allow up to +10% RMSE regression
  latency_p95: 0.20 # allow up to +20% latency regression
```

```yaml
# configs/experiment.yaml
n_samples: 1000
n_features: 20
noise: 10.0
random_state: 42
alpha: 1.0
```

### Step 6 — GitHub Actions workflow

```yaml
# .github/workflows/ci.yml
name: gatekitchen CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff black mypy
      - run: ruff check src/ tests/
      - run: black --check src/ tests/

  test-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install scikit-learn numpy pyyaml pytest
      - run: pip install -e .
      - name: Unit tests
        run: pytest tests/ -v
      - name: Train model
        run: python -m gatekitchen.train
      - name: Evaluate + gate
        run: python -m gatekitchen.evaluate
      - name: Check report
        run: |
          python - <<'EOF'
          import json, sys
          with open("reports/latest/evaluation_report.json") as f:
              r = json.load(f)
          sys.exit(0 if r["pass"] else 1)
          EOF
```

### Step 7 — unit tests for `check_gates()`

```python
# tests/test_reporter.py
import pytest
from gatekitchen.reporter import check_gates


@pytest.fixture
def good_metrics():
    return {"r2": 0.90, "rmse": 30.0, "latency_p95": 0.02}

@pytest.fixture
def floors():
    return {"r2": 0.80, "rmse": 50.0, "latency_p95": 0.05}

@pytest.fixture
def deltas():
    return {"r2": -0.05, "rmse": 0.10, "latency_p95": 0.20}


def test_pass_no_baseline(good_metrics, floors, deltas):
    passed, failures = check_gates(good_metrics, floors, deltas, baseline_metrics=None)
    assert passed
    assert failures == []


def test_fail_absolute_floor_r2(floors, deltas):
    bad = {"r2": 0.70, "rmse": 30.0, "latency_p95": 0.02}
    passed, failures = check_gates(bad, floors, deltas, baseline_metrics=None)
    assert not passed
    assert any("r2" in f and "ABSOLUTE FLOOR" in f for f in failures)


def test_pass_with_good_baseline(good_metrics, floors, deltas):
    baseline = {"r2": 0.88, "rmse": 32.0, "latency_p95": 0.025}
    passed, failures = check_gates(good_metrics, floors, deltas, baseline)
    assert passed


def test_fail_regression_r2(floors, deltas):
    # r2 dropped 10% relative: (0.80 - 0.90) / 0.90 = -0.111 < -0.05
    regressed = {"r2": 0.80, "rmse": 30.0, "latency_p95": 0.02}
    baseline = {"r2": 0.90, "rmse": 30.0, "latency_p95": 0.02}
    passed, failures = check_gates(regressed, floors, deltas, baseline)
    assert not passed
    assert any("r2" in f and "REGRESSION" in f for f in failures)


def test_cold_start_skips_relative_delta(good_metrics, floors, deltas):
    # Even if metrics are barely above floor, no baseline = no relative check
    borderline = {"r2": 0.81, "rmse": 49.0, "latency_p95": 0.049}
    passed, failures = check_gates(borderline, floors, deltas, baseline_metrics=None)
    assert passed


def test_missing_metric_defaults_to_worst():
    # Missing metric defaults to 0.0 (via .get(key, 0.0)), fails floor check
    metrics = {}  # missing r2
    floors = {"r2": 0.80}
    passed, failures = check_gates(metrics, floors, {}, baseline_metrics=None)
    assert not passed
```

---

## Exercises

1. **Trigger analysis:** In `ci.yml`, add a `workflow_dispatch` trigger with a boolean input `run_slow_tests`. When it's `true`, change the pytest command to include slow tests (`-m ""`). When false, keep `"not slow"`.

2. **Gate extension:** Add a fourth metric `mae` to `check_gates()` with an absolute floor of 40.0. Write a test case where `rmse` passes but `mae` fails.

3. **Cold-start simulation:** Modify `evaluate.py` to accept a `--save-as-baseline` flag. When set, copy `evaluation_report.json` to `reports/baseline/evaluation_report.json`. Run evaluation twice: once without baseline, once using the first run as baseline. Observe the difference in output.

4. **Dockerfile:** Write a `Dockerfile` for the `gatekitchen` project that:
   - Uses `python:3.11-slim`
   - Installs only runtime dependencies (not dev)
   - COPYs `src/`, `configs/`, `models/`
   - Sets `CMD` to `python -m gatekitchen.evaluate`

5. **Relative delta direction:** The current `check_gates()` uses `delta >= max_delta` to detect failure. Explain in a comment why this correctly handles *both* RMSE (where bigger is worse) and R² (where smaller is worse) with the same formula. Hint: think about what `max_delta` means for each direction.

6. **`[skip ci]` tag:** Research and explain: why does the `pipeline.yml` commit with `[skip ci]` in the message? What would happen if it didn't? Write a 3-sentence explanation in your own words.

---

## Summary

| Concept | Where in project |
|---|---|
| CI code quality gate | `.github/workflows/ci.yml` — ruff, black, mypy, pytest |
| GPU pipeline run | `.github/workflows/pipeline.yml` — self-hosted runner, DVC repro |
| Evaluation gate in CI | `pipeline.yml` "Check evaluation gate" step — inline Python, sys.exit(1) |
| `evaluation_report.json` | Generated by `reporter.py:generate_report()` |
| Two-tier check logic | `reporter.py:check_gates()` — absolute floor + relative delta |
| Cold-start handling | `check_gates()` skips relative delta when `baseline_metrics is None` |
| Shadow→Production promotion | `.github/workflows/promote.yml` — workflow_run + manual approval |
| Re-deploy trigger | `promote.yml` `repository-dispatch` — decoupled from serving infra |
| Metrics (UTMOS/CER/STOI/RTF) | `src/tts_pipeline/evaluation/metrics/` — one file per metric |
| Docker serving image | `docker/Dockerfile.serve` — slim, volume-mounted bundle |
| Service mesh | `docker/docker-compose.yml` — mlflow + prefect + api |
| Developer interface | `Makefile` — all common commands in one place |