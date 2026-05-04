# Course 12: Prefect — Workflow Orchestration

**Prerequisites:** Course 10 (DVC) — you need to understand DVC stages and `dvc repro`. Course 11 (MLflow) — helpful for understanding what the training and evaluation stages produce.

**What this course covers:** Prefect is the scheduling and orchestration layer that wraps the DVC pipeline. By the end you will understand the `@task` / `@flow` contract, how retries work, how Prefect and DVC divide responsibilities, and how to deploy a flow that runs on a schedule. Every concept is grounded in the actual `pipelines/` directory.

**Project anchors:**
- [pipelines/data_pipeline.py](../pipelines/data_pipeline.py) — data flow with 6 tasks
- [pipelines/training_pipeline.py](../pipelines/training_pipeline.py) — acoustic + vocoder training
- [pipelines/evaluation_pipeline.py](../pipelines/evaluation_pipeline.py) — evaluate + export

---

## Module 1 — Why Prefect Alongside DVC

You already have DVC. Why do you also need Prefect?

| Concern | DVC | Prefect |
|---|---|---|
| Caching / skip stale stages | ✅ | ❌ |
| Artifact hashing + reproducibility | ✅ | ❌ |
| Scheduling (cron, interval) | ❌ | ✅ |
| Retry on transient failure | ❌ | ✅ |
| Real-time run status UI | ❌ | ✅ |
| Alerting on failure | ❌ | ✅ |
| Parametrized deployments | ❌ | ✅ |

DVC answers "what needs to run and in what order, given which inputs changed." Prefect answers "when should this run, what happens if a step fails, and who gets notified?"

The project's division of labour is exactly this: every Prefect task in `pipelines/` is just a thin wrapper that calls `dvc repro <stage>`. DVC handles the logic and caching; Prefect handles the ops.

```python
# All three pipeline files follow the same pattern:
def _dvc_repro(stage: str) -> None:                 # data_pipeline.py:13
    logger = get_run_logger()
    result = subprocess.run(["dvc", "repro", stage], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dvc repro {stage} failed:\n{result.stderr}")
    logger.info(result.stdout)
```

The `RuntimeError` on non-zero exit is what tells Prefect "this task failed" — Prefect sees the uncaught exception and marks the task as `FAILED`, triggering retries if configured.

---

## Module 2 — The `@task` Decorator

### Basic anatomy

```python
@task(retries=2, retry_delay_seconds=30)    # data_pipeline.py:22
def ingest() -> None:
    _dvc_repro("ingest")
```

`@task` turns a plain Python function into a Prefect task. A task is the smallest unit of work in a flow. It gets its own:
- State (`PENDING` → `RUNNING` → `COMPLETED` / `FAILED`)
- Retry policy
- Logs (separate from the flow's logs)
- Cache (optional — skip if output is fresh)

The function body is unchanged. Prefect wraps it, not modifies it.

### `retries` and `retry_delay_seconds`

```python
@task(retries=2, retry_delay_seconds=30)   # ingest, normalize_audio
@task(retries=1)                            # clean_transcripts, phonemize, extract_features, validate
@task(retries=1, retry_delay_seconds=60)   # train_acoustic, train_vocoder
```

When a task raises an exception:
1. Prefect marks it `RETRYING`
2. Waits `retry_delay_seconds` (default 0)
3. Calls the function again
4. Repeats until success or `retries` exhausted
5. If still failing → marks `FAILED` → the whole flow is marked `FAILED`

Why different retry settings per task?
- `ingest` and `normalize_audio` touch the network/disk heavily → 2 retries, 30s gap (transient I/O errors)
- `train_acoustic` → 1 retry, 60s gap (GPU OOM can sometimes recover after a pause)
- `clean_transcripts` → 1 retry, no delay (fast and deterministic — if it fails twice it won't recover)

### `task_input_hash` — content-based caching

```python
from prefect.tasks import task_input_hash

@task(retries=1, cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def expensive_step(data_hash: str) -> dict:
    ...
```

`task_input_hash` generates a cache key from the task's input arguments. If the same task is called with the same arguments within `cache_expiration`, Prefect returns the cached result without re-running the function body.

Our pipeline doesn't use this — DVC already handles caching at the artifact level. `task_input_hash` is useful when Prefect tasks compute results directly (not via DVC).

### `get_run_logger()`

```python
def _dvc_repro(stage: str) -> None:
    logger = get_run_logger()           # data_pipeline.py:14
    logger.info("Running: dvc repro %s", stage)
```

`get_run_logger()` returns a Python logger that sends messages to Prefect's logging infrastructure. These logs appear in the Prefect UI under the specific task run — not just in the terminal. You cannot use a module-level logger inside tasks if you want logs to appear in the UI.

**Important:** `get_run_logger()` only works inside a task or flow function. Calling it outside raises `MissingContextError`. For module-level logging, use a standard Python logger.

---

## Module 3 — The `@flow` Decorator

### Basic anatomy

```python
@flow(name="data-pipeline", log_prints=True)    # data_pipeline.py:52
def data_pipeline() -> None:
    ingest()
    normalize_audio()
    clean_transcripts()
    phonemize()
    extract_features()
    validate()
```

`@flow` marks the top-level orchestration function. When called, Prefect:
1. Creates a **flow run** with a unique ID
2. Calls each task function in order
3. Tracks the state of each task run
4. Marks the flow run `COMPLETED` or `FAILED` at the end

`log_prints=True` captures any `print()` calls inside the flow and tasks and routes them to Prefect's log store.

### Sequential execution (the default)

When you call tasks sequentially inside a flow:

```python
def data_pipeline() -> None:
    ingest()           # waits until COMPLETED before proceeding
    normalize_audio()  # only runs if ingest() COMPLETED
    clean_transcripts()
    ...
```

Each task call **blocks** until that task finishes. This matches the DVC DAG dependency order exactly — later stages depend on earlier stages' outputs.

### Concurrent execution with `.submit()`

When tasks don't depend on each other, you can run them in parallel:

```python
@flow
def parallel_pipeline() -> None:
    # Submit both without waiting — they run concurrently
    acoustic_future = train_acoustic.submit()
    vocoder_future  = train_vocoder.submit()

    # Block until both finish before proceeding
    acoustic_future.result()
    vocoder_future.result()

    # Only runs after both complete
    evaluate()
```

`.submit()` returns a `PrefectFuture`. Calling `.result()` blocks until the task completes and returns its return value (or re-raises its exception).

Our `training_pipeline` currently runs `train_acoustic()` and `train_vocoder()` sequentially. They could safely run concurrently on separate GPUs with `.submit()` — this is a potential optimization left as an exercise.

### Flow return value and state

A flow function can return values:

```python
@flow
def data_pipeline() -> str:
    ingest()
    ...
    return "data-pipeline-complete"
```

The return value is stored in the flow run state. If any task fails and is not handled, the flow raises `prefect.exceptions.CrashDetected` and the return value is `None`.

---

## Module 4 — Task States

Every task run transitions through these states:

```
PENDING
  ↓ (Prefect schedules it)
RUNNING
  ↓ (function executes)
  ├── returns normally → COMPLETED
  ├── raises exception, retries remain → RETRYING → RUNNING → ...
  └── raises exception, retries exhausted → FAILED
                                              ↓
                                    flow run → FAILED
```

### Handling task failure in a flow

By default, if a task fails (raises an unhandled exception after all retries), Prefect marks the flow as `FAILED` and stops executing remaining tasks. To handle failure explicitly:

```python
from prefect import allow_failure

@flow
def resilient_pipeline() -> None:
    result = validate.submit()
    # allow_failure wraps the future — downstream tasks run even if validate fails
    export.submit(wait_for=[allow_failure(result)])
```

Or catch the exception directly:

```python
@flow
def resilient_pipeline() -> None:
    try:
        validate()
    except Exception as e:
        notify_team(str(e))
        raise   # re-raise so the flow is still marked FAILED
```

### State inspection

```python
from prefect.states import Completed, Failed

@flow
def checked_pipeline() -> None:
    state = ingest(return_state=True)   # return State object, not result
    if state.is_failed():
        print("Ingest failed, skipping downstream")
        return
    normalize_audio()
```

`return_state=True` makes the task return its `State` object instead of raising on failure.

---

## Module 5 — Running Flows

### Local execution

The simplest way to run a flow is to call it like a regular function:

```python
# data_pipeline.py:62
if __name__ == "__main__":
    data_pipeline()
```

```bash
python pipelines/data_pipeline.py
```

This runs the flow locally with no server required. Prefect still tracks state in a local SQLite database at `~/.prefect/prefect.db`.

### Prefect server (local UI)

```bash
prefect server start          # starts the Prefect UI at http://localhost:4200
# In another terminal:
python pipelines/data_pipeline.py
```

Now the flow run appears in the UI with full task-level state, logs, and timing.

### `prefect.yaml` — deployment configuration

A **deployment** packages a flow for remote execution with a schedule. You define it in `prefect.yaml` (not present in the project — this is what you'd add):

```yaml
# prefect.yaml (would live at project root)
name: tts-pipeline
prefect-version: 2.x.x

deployments:
  - name: data-pipeline-daily
    entrypoint: pipelines/data_pipeline:data_pipeline
    schedule:
      cron: "0 2 * * *"          # run at 2am every night
    work_pool:
      name: default-agent-pool

  - name: training-pipeline
    entrypoint: pipelines/training_pipeline:training_pipeline
    schedule: null               # triggered manually or by another flow
    work_pool:
      name: gpu-pool
```

Deploy and run:

```bash
prefect deploy --all             # register all deployments
prefect worker start --pool default-agent-pool   # start the worker
```

---

## Module 6 — Schedules and Deployments

### CronSchedule

```python
from prefect.client.schemas.schedules import CronSchedule

# Programmatic deployment (alternative to prefect.yaml)
from prefect.deployments import Deployment

deployment = Deployment.build_from_flow(
    flow=data_pipeline,
    name="data-pipeline-nightly",
    schedule=CronSchedule(cron="0 2 * * *", timezone="Asia/Ho_Chi_Minh"),
    tags=["data", "nightly"],
)
deployment.apply()
```

Common cron expressions:

| Expression | Meaning |
|---|---|
| `"0 2 * * *"` | Every day at 02:00 |
| `"0 */6 * * *"` | Every 6 hours |
| `"0 2 * * 1"` | Every Monday at 02:00 |
| `"0 2 1 * *"` | First of every month at 02:00 |

### IntervalSchedule

```python
from datetime import timedelta
from prefect.client.schemas.schedules import IntervalSchedule

schedule = IntervalSchedule(interval=timedelta(hours=12))
```

### Triggering a deployment manually

```bash
prefect deployment run data-pipeline/data-pipeline-nightly
```

Or with parameter overrides:

```bash
prefect deployment run training-pipeline/training-pipeline \
  --param language=en
```

---

## Module 7 — How DVC and Prefect Interact in the Project

### The layered architecture

```
Prefect layer (scheduling + ops)
    ├─ data_pipeline flow
    │   └─ calls: dvc repro ingest
    │   └─ calls: dvc repro normalize_audio
    │   └─ ...
    ├─ training_pipeline flow
    │   └─ calls: dvc repro train
    │   └─ calls: dvc repro train_vocoder
    └─ evaluation_pipeline flow
        └─ calls: dvc repro evaluate
        └─ calls: dvc repro export

DVC layer (reproducibility + caching)
    ├─ ingest stage → data/metadata.csv
    ├─ normalize_audio → data/processed/audio_normalized/
    ├─ ... (6 more stages)
    └─ export → artifacts/export/*.onnx
```

Prefect calls `subprocess.run(["dvc", "repro", stage])`. DVC runs the stage, applies caching, and exits. Prefect sees the exit code: 0 → task `COMPLETED`, non-zero → task `FAILED` → retry policy kicks in.

### What this means for retries

When Prefect retries a task, it calls `dvc repro <stage>` again. DVC will:
- **Skip** the stage if its inputs haven't changed (already cached) — a free no-op retry
- **Re-run** the stage if it failed mid-write and left partial outputs

This is an important property: Prefect retries are cheap because DVC's caching prevents redundant work.

### The full pipeline chain

In practice, the three flows run sequentially as a pipeline:

```
data_pipeline() → training_pipeline() → evaluation_pipeline()
```

This chain can be expressed as a parent flow:

```python
@flow(name="full-tts-pipeline")
def full_pipeline() -> None:
    data_pipeline()       # subflow call
    training_pipeline()
    evaluation_pipeline()
```

When you call a flow inside another flow, Prefect treats the inner flow as a **subflow** — it gets its own flow run ID and shows up nested in the UI.

---

## Module 8 — Notifications and Observability

### Flow run notifications

Prefect can send notifications when a flow run changes state. Configure in the UI under **Notifications**, or programmatically:

```python
from prefect.blocks.notifications import SlackWebhook

slack = SlackWebhook(url="https://hooks.slack.com/services/...")
slack.save("slack-tts-alerts")
```

Then in the UI, create an Automation: "When flow run fails → Send notification to slack-tts-alerts".

### Logging best practices

```python
@task(retries=2, retry_delay_seconds=30)
def ingest() -> None:
    logger = get_run_logger()            # ← Prefect UI logger
    logger.info("Starting ingestion")
    try:
        _dvc_repro("ingest")
    except RuntimeError as e:
        logger.error("DVC stage failed: %s", e)
        raise                            # must re-raise — Prefect sees exception → FAILED
```

If you catch the exception without re-raising, Prefect marks the task `COMPLETED` even though it failed. Always re-raise after logging.

### Task run tags

```python
@task(retries=1, tags=["gpu", "training"])
def train_acoustic() -> None:
    _dvc_repro("train")
```

Tags appear in the Prefect UI and can filter flow runs. Useful for routing tasks to specific workers (e.g., only a GPU worker picks up tasks tagged `"gpu"`).

---

## Module 9 — Common Patterns in the Project

### Pattern: subprocess wrapper with structured error

Every task in the project follows the same pattern:

```python
def _dvc_repro(stage: str) -> None:
    logger = get_run_logger()
    logger.info("Running: dvc repro %s", stage)
    result = subprocess.run(
        ["dvc", "repro", stage],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc repro {stage} failed:\n{result.stderr}")
    logger.info(result.stdout)
```

`capture_output=True` captures both stdout and stderr so they can be logged before raising. Without it, subprocess output goes to the terminal but not to Prefect's log store.

`text=True` decodes bytes to str automatically (avoids `result.stdout.decode("utf-8")`).

### Pattern: flow as sequential stage orchestrator

```python
@flow(name="data-pipeline", log_prints=True)
def data_pipeline() -> None:
    ingest()            # each line waits for completion
    normalize_audio()
    clean_transcripts()
    phonemize()
    extract_features()
    validate()
```

This mirrors the DVC DAG order exactly. The tasks are already in dependency order because DVC enforces it. Prefect just provides the outer retry/monitor shell.

### Pattern: explicit flow name

```python
@flow(name="training-pipeline", log_prints=True)
def training_pipeline() -> None:
```

The `name` argument sets the flow name in the Prefect UI independently of the Python function name. Convention: use kebab-case names for UI clarity.

---

## Module 10 — Common Pitfalls

### Pitfall 1: `get_run_logger()` called at module level

```python
# BAD — crashes at import time with MissingContextError
logger = get_run_logger()

@task
def my_task():
    logger.info("...")

# GOOD — call inside the task body
@task
def my_task():
    logger = get_run_logger()
    logger.info("...")
```

`get_run_logger()` requires an active Prefect task or flow context. Module-level code runs outside that context.

### Pitfall 2: Catching exceptions without re-raising

```python
# BAD — Prefect thinks the task succeeded
@task(retries=2)
def ingest():
    try:
        _dvc_repro("ingest")
    except RuntimeError:
        pass   # silently swallowed — task marked COMPLETED, retries never fire

# GOOD
@task(retries=2)
def ingest():
    try:
        _dvc_repro("ingest")
    except RuntimeError as e:
        get_run_logger().error("Ingest failed: %s", e)
        raise   # Prefect sees the exception → FAILED → retry
```

### Pitfall 3: `retries` on non-idempotent operations

```python
# DANGEROUS — if this writes a partial file then crashes, retrying duplicates work
@task(retries=3)
def upload_to_database(records):
    db.insert_many(records)   # if crash midway → retry inserts duplicates
```

Only retry tasks that are **idempotent** — running them twice produces the same result as running once. DVC stages are idempotent (DVC detects the partial output and re-runs from scratch). Database inserts are not unless you use upsert semantics.

### Pitfall 4: Long-running tasks without timeout

```python
# Risk: task hangs forever (GPU deadlock, network stall)
@task(retries=1)
def train_acoustic():
    _dvc_repro("train")

# Better: add a timeout
@task(retries=1, timeout_seconds=86400)   # 24h hard limit
def train_acoustic():
    _dvc_repro("train")
```

`timeout_seconds` causes Prefect to mark the task `TimedOut` (a subtype of `FAILED`) if it exceeds the limit. Training can legitimately take hours — set the timeout generously but not infinitely.

### Pitfall 5: Running flows from inside tasks

```python
# BAD — nested flow inside a task
@task
def my_task():
    data_pipeline()    # calling a @flow from inside a @task
```

Flows should call flows (as subflows) or tasks. Tasks should not call flows — it creates ambiguous state tracking. If `data_pipeline()` fails inside a task, Prefect doesn't know whether to retry the task or the inner flow.

---

## Module 11 — Mini Practice Project: `prefectlab`

An independent mini-project that builds a 3-task Prefect flow with retries, concurrent tasks, a local schedule, and a failure notification (print-based). No TTS or DVC dependency.

**Goal:** ETL pipeline that downloads mock data, transforms it, and loads it to a CSV. One task has a random failure to exercise the retry policy.

### Project structure

```
prefectlab/
├── flows/
│   ├── etl_flow.py      # main flow + tasks
│   └── daily_flow.py    # scheduled deployment
└── README.md
```

### `flows/etl_flow.py`

```python
import random
import time
import csv
import math
from datetime import datetime
from pathlib import Path
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta


@task(retries=3, retry_delay_seconds=2)
def extract(n_records: int = 500) -> list[dict]:
    """Simulate fetching data from an API that fails ~30% of the time."""
    logger = get_run_logger()
    logger.info("Extracting %d records...", n_records)

    # Simulate transient API failure
    if random.random() < 0.3:
        raise ConnectionError("Simulated API timeout — will retry")

    # Generate synthetic records
    records = []
    for i in range(n_records):
        t = i / n_records * 2 * math.pi
        records.append({
            "id": i,
            "timestamp": datetime.utcnow().isoformat(),
            "value": round(math.sin(t) + random.gauss(0, 0.1), 4),
            "category": "A" if math.sin(t) > 0 else "B",
        })

    logger.info("Extracted %d records successfully", len(records))
    return records


@task
def transform(records: list[dict]) -> list[dict]:
    """Normalize values to [0, 1] range and add a derived field."""
    logger = get_run_logger()
    values = [r["value"] for r in records]
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0

    transformed = []
    for r in records:
        transformed.append({
            **r,
            "value_normalized": round((r["value"] - lo) / span, 4),
            "is_positive": r["value"] > 0,
        })

    logger.info("Transformed %d records (range: %.3f → %.3f)", len(records), lo, hi)
    return transformed


@task
def load(records: list[dict], output_path: str = "output/data.csv") -> int:
    """Write records to a CSV file. Returns count of written rows."""
    logger = get_run_logger()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    logger.info("Loaded %d records → %s", len(records), output_path)
    return len(records)


@task
def summarize(output_path: str = "output/data.csv") -> dict:
    """Read the output CSV and compute summary statistics."""
    logger = get_run_logger()
    rows = list(csv.DictReader(open(output_path)))
    values = [float(r["value_normalized"]) for r in rows]
    summary = {
        "n": len(rows),
        "mean": round(sum(values) / len(values), 4),
        "n_positive": sum(1 for r in rows if r["is_positive"] == "True"),
        "n_category_A": sum(1 for r in rows if r["category"] == "A"),
    }
    logger.info("Summary: %s", summary)
    return summary


@flow(name="etl-pipeline", log_prints=True)
def etl_pipeline(n_records: int = 500, output_path: str = "output/data.csv") -> dict:
    """Full ETL: extract → transform → load → summarize."""
    raw = extract(n_records)
    clean = transform(raw)
    load(clean, output_path)
    stats = summarize(output_path)
    return stats


if __name__ == "__main__":
    result = etl_pipeline(n_records=200)
    print(f"\nFinal stats: {result}")
```

### `flows/daily_flow.py` — scheduling a deployment

```python
from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments import Deployment
from etl_flow import etl_pipeline


def deploy():
    deployment = Deployment.build_from_flow(
        flow=etl_pipeline,
        name="etl-pipeline-daily",
        schedule=CronSchedule(cron="0 6 * * *"),   # run at 6am daily
        parameters={"n_records": 1000, "output_path": "output/daily.csv"},
        tags=["etl", "daily"],
    )
    deployment.apply()
    print("Deployment registered.")


if __name__ == "__main__":
    deploy()
```

### Running the project

```bash
pip install prefect

# 1. Start the Prefect UI (optional but recommended)
prefect server start
# Open http://localhost:4200

# 2. Run the flow (from prefectlab/)
cd flows
python etl_flow.py
```

You will see the retry in action: the `extract` task will fail ~30% of the time and retry automatically. Watch the Prefect UI to see task states change in real time.

### Exercises

1. **Observe retries** — run the flow 5 times. At least once, `extract` should fail on the first attempt and succeed on retry. In the Prefect UI, find the run where the retry occurred and examine the task run logs.

2. **Make tasks concurrent** — modify `etl_pipeline` to run `load` and `summarize` concurrently using `.submit()`. Note: `summarize` reads the file that `load` writes, so you need to call `.result()` on the `load` future before calling `summarize`. Think carefully about the correct ordering.

3. **Add a timeout** — add `timeout_seconds=5` to the `extract` task. Add `time.sleep(6)` inside `extract` (after the failure check) and run the flow. Observe the `TimedOut` state in the UI.

4. **Parametrize the flow** — modify the `if __name__ == "__main__"` block to accept `n_records` from `sys.argv`. Run with `python etl_flow.py 100` and `python etl_flow.py 2000`. Both runs appear in the UI under the same flow name, with different parameter values.

5. **Subflow composition** — create a `master_flow.py` that calls `etl_pipeline` as a subflow twice with different `output_path` values ("output/batch1.csv" and "output/batch2.csv"). Run it and observe how the two subflows appear nested in the Prefect UI.

6. **Failure alert** — add a `try/except` around the `etl_pipeline()` call in `__main__` that prints `"ALERT: pipeline failed — would send Slack message here"` when the flow raises. Temporarily increase the failure probability to 1.0 in `extract` to force the alert to fire.

---

## Key Takeaways

- Prefect and DVC are complementary. DVC owns caching and reproducibility; Prefect owns scheduling, retries, and observability. The project's pattern is: every Prefect task calls `dvc repro <stage>` and raises `RuntimeError` on non-zero exit.
- `@task(retries=N, retry_delay_seconds=S)` is the entire retry configuration. When the task raises, Prefect waits S seconds and calls the function again up to N times.
- `get_run_logger()` must be called inside a task or flow body — never at module level. It routes logs to the Prefect UI per-task-run.
- Sequential task calls (`task()`) block until completion; concurrent calls (`task.submit()`) return a `PrefectFuture` and the caller must `future.result()` to block and get the value.
- Always re-raise exceptions after catching them in a task. A silently swallowed exception marks the task `COMPLETED` and disables retries.
- A flow calling another flow creates a **subflow** — nested in the UI with its own run ID. This is how the three pipeline flows (`data_pipeline`, `training_pipeline`, `evaluation_pipeline`) chain into a full `full_pipeline`.
- `prefect.yaml` or `Deployment.build_from_flow()` registers a flow for scheduled execution. Workers poll the Prefect server for queued runs and execute them on a schedule.