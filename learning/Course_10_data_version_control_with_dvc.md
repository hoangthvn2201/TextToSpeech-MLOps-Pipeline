# Course 10: Data Version Control with DVC

**Prerequisites:** Course 1 (Python), basic Git (commits, branches, remotes).

**What this course covers:** DVC is the reproducibility and data versioning layer of the pipeline. By the end you will understand what every line in `dvc.yaml` does, how DVC decides which stages are stale and need re-running, how large data files are kept out of Git, and how `params.yaml` acts as the single source of truth that both DVC and Hydra read.

**Project anchors:**
- [dvc.yaml](../dvc.yaml) — the full pipeline DAG (9 stages)
- [params.yaml](../params.yaml) — all pipeline parameters tracked by DVC
- [scripts/run_data_pipeline.py](../scripts/run_data_pipeline.py) — the CLI DVC calls for data stages
- [scripts/run_training.py](../scripts/run_training.py) — the CLI DVC calls for the train stage

---

## Module 1 — Why DVC Exists

### The problem with raw data in ML projects

Git tracks code well. It does not track data well:
- A 50 GB audio dataset cannot be committed to Git (file size limits, repo bloat)
- If you retrain with new data, there is no record of which dataset version produced which model
- If a teammate runs the same script on a slightly different dataset, results diverge silently

DVC solves three distinct problems:

| Problem | DVC solution |
|---|---|
| Large files can't go in Git | Store files in a remote (S3, GCS, local path), put a tiny `.dvc` pointer in Git |
| "What produced this artifact?" | `dvc.lock` records every input hash that produced every output |
| "Which stages need to re-run?" | Fingerprint deps/params; skip stages whose inputs haven't changed |

### DVC's relationship to Git

DVC is not a replacement for Git — it is an extension. The rule is:

```
Git tracks:        DVC tracks:
- source code      - data files
- configs          - model checkpoints
- dvc.yaml         - processed artifacts
- dvc.lock         (the actual bytes, stored in a remote)
- params.yaml
```

When you `git checkout` a branch, you get `dvc.lock` from that branch. Then `dvc checkout` fetches the exact data files that `dvc.lock` points to. Git + DVC together give you reproducible snapshots of both code and data.

---

## Module 2 — The Pipeline DAG

### Reading `dvc.yaml`

DVC's pipeline is a Directed Acyclic Graph (DAG) defined in `dvc.yaml`. Each **stage** has:
- `cmd` — the shell command DVC will run
- `deps` — files/directories this stage reads
- `outs` — files/directories this stage writes
- `params` — specific keys from `params.yaml` this stage reads

Our pipeline has 9 stages. Here is the full dependency graph:

```
data/raw/audio  data/raw/transcripts
       │                │
       └───── ingest ───┘
                │
          data/metadata.csv
          ┌────┘    └────────┐
normalize_audio        clean_transcripts
          │                  │
data/processed/        data/processed/
audio_normalized       transcripts_cleaned
          │                  │
          │              phonemize
          │                  │
          │         data/processed/phonemes.csv
          │                  │
          └──── extract_features ────┘
                      │
        data/processed/mels + f0 + stats.json
                      │
                  validate
                      │
              data/splits/{train,val,test}.txt
                      │
          ┌───────────┼───────────┐
         train    train_vocoder   │
          │            │          │
     best_model.ckpt  hifigan.ckpt│
          └────────────┴──── evaluate
                              │
                   reports/latest/evaluation_report.json
                              │
                           export
                              │
              artifacts/export/{acoustic,vocoder}.onnx
```

DVC computes this graph by reading `deps` and `outs` across all stages. It never needs to run code to understand the shape — it reads the YAML.

### A single stage in detail: `extract_features`

```yaml
extract_features:                                    # dvc.yaml:60
  cmd: python scripts/run_data_pipeline.py --stage features
  deps:
    - data/processed/audio_normalized                # output of normalize_audio
    - data/processed/phonemes.csv                    # output of phonemize
    - src/tts_pipeline/data/feature_extractor.py    # source code is also a dep
  outs:
    - data/processed/mels                            # directory of .npy files
    - data/processed/f0
    - data/processed/stats.json
  params:
    - params.yaml:
        - data.sample_rate                           # dotted path into the YAML
        - data.n_fft
        - data.hop_length
        - data.win_length
        - data.n_mels
        - data.f_min
        - data.f_max
```

**Why source code is a dep:** If `feature_extractor.py` changes (e.g., you fix a mel computation bug), the stage is stale and must re-run. Without it as a dep, DVC would skip re-running even though the code changed and the output would be wrong.

**Why params are separate from deps:** DVC reads `params.yaml` directly and hashes only the listed keys — not the entire file. Changing `training.learning_rate` does not invalidate `extract_features`, even though both are in `params.yaml`.

---

## Module 3 — `params.yaml`: The Single Source of Truth

### Why one file governs everything

```yaml
# params.yaml — both DVC and Hydra read this file.
# Changing any value here will invalidate downstream DVC stages automatically.

data:
  sample_rate: 22050
  n_fft: 1024
  hop_length: 256
  ...

training:
  learning_rate: 1.0e-4
  warmup_steps: 1000
  ...
```

DVC reads `params.yaml` to detect when a tracked key changes. Hydra reads the same file to configure the training run. There is only one place to change `n_mels` — you never have to keep two config files in sync.

### How DVC hashes params

DVC does not hash the entire `params.yaml` file. It extracts the specific keys listed in each stage's `params:` block and hashes their values. This is why:

```
Changing data.n_mels=80 → 100:
  - ingest:           NOT invalidated (does not list n_mels)
  - normalize_audio:  NOT invalidated
  - extract_features: INVALIDATED (lists data.n_mels)
  - validate:         INVALIDATED (depends on mels output)
  - train:            INVALIDATED (depends on mels output and lists model params)
```

DVC propagates invalidation downstream through the DAG. If `extract_features` is stale, all stages that depend on its outputs are also stale.

### Dotted key paths

The syntax `data.sample_rate` in the `params:` block is a dotted path into the YAML hierarchy:

```yaml
# params.yaml
data:
  sample_rate: 22050   # ← accessed as data.sample_rate
```

You can also track an entire section:

```yaml
params:
  - params.yaml:
      - model          # ← tracks ALL keys under model.*
      - training       # ← tracks ALL keys under training.*
```

This is what the `train` stage does (dvc.yaml:116-118). Any change to any model or training parameter invalidates the train stage.

---

## Module 4 — `dvc.lock`: The Reproducibility Ledger

### What `dvc.lock` contains

After `dvc repro` runs successfully, DVC writes `dvc.lock`. This file records the exact state of every input that produced every output:

```yaml
# dvc.lock (generated — never edit by hand)
schema: '2.0'
stages:
  extract_features:
    cmd: python scripts/run_data_pipeline.py --stage features
    deps:
    - path: data/processed/audio_normalized
      hash: md5
      md5: a8f3c2...  # hash of the directory's contents
    - path: src/tts_pipeline/data/feature_extractor.py
      hash: md5
      md5: 9b1d4e...
    params:
      params.yaml:
        data.n_fft: 1024
        data.hop_length: 256
        data.n_mels: 80
        ...
    outs:
    - path: data/processed/mels
      hash: md5
      md5: c5a2f7...
      size: 2147483648
```

`dvc.lock` is committed to Git. When you checkout a commit, `dvc.lock` tells DVC exactly which data files (by hash) correspond to that commit. Running `dvc checkout` then fetches those exact files from the remote cache.

### The MD5 hash of a directory

For directories (like `data/processed/mels`), DVC computes a manifest: it hashes every file inside the directory and then hashes the list of (filename, hash) pairs. This means:
- Adding or removing one file invalidates the directory hash
- Modifying one file inside the directory invalidates the directory hash
- Reordering files does NOT invalidate (the list is sorted before hashing)

### `dvc.lock` vs `.dvc` files

You will see two ways DVC tracks files:

| Mechanism | When used | What it looks like |
|---|---|---|
| `dvc.lock` | Outputs declared in `dvc.yaml` stages | Auto-generated file in project root |
| `<file>.dvc` | Standalone files tracked with `dvc add` | `data/raw/audio.dvc` next to the file |

Our project uses `dvc.yaml` stages for all processed data. The `.dvc` approach is for raw data files that aren't produced by any stage (e.g., the original audio download).

---

## Module 5 — Remote Storage

### What a DVC remote is

A remote is a storage location outside your Git repo where DVC pushes the actual file bytes. When a teammate clones your repo and runs `dvc pull`, DVC fetches the data matching `dvc.lock` from the remote.

### Configuring a remote (`.dvc/config`)

```ini
# .dvc/config — committed to Git
[core]
    remote = myremote

['remote "myremote"']
    url = s3://my-bucket/tts-pipeline/dvc-cache
```

Common remote types:

| Remote type | URL format | Use case |
|---|---|---|
| Local path | `/mnt/nas/dvc` | Team NAS, local testing |
| S3 | `s3://bucket/path` | AWS, most common in production |
| GCS | `gs://bucket/path` | Google Cloud |
| Azure Blob | `azure://container/path` | Azure |
| SSH | `ssh://host/path` | Any Linux server |

### The DVC cache

DVC maintains a local cache at `.dvc/cache/` (or `~/.dvc/cache/` on some setups). Files are stored by content hash:

```
.dvc/cache/
  files/
    md5/
      a8/f3c2abcdef...   ← file whose MD5 starts with "a8f3c2..."
      9b/1d4e567890...
```

This is a content-addressable store — identical files are stored once regardless of how many stages produce them.

### `dvc push` / `dvc pull`

```bash
dvc push          # upload local cache → remote (run after dvc repro)
dvc pull          # download remote → local cache + workspace (run after git pull)
dvc fetch         # download remote → local cache only (don't put in workspace)
dvc checkout      # copy from local cache → workspace (using dvc.lock)
```

The typical team workflow:

```bash
# Developer A: runs pipeline and shares results
dvc repro
git add dvc.lock
git commit -m "run: extract features with n_mels=80"
dvc push                  # push data to shared remote
git push

# Developer B: gets the exact same data without re-running
git pull
dvc pull                  # fetch data matching the new dvc.lock
```

---

## Module 6 — `dvc repro`: The Core Command

### How DVC decides what to run

`dvc repro` walks the DAG in topological order. For each stage it computes a **run signature** and compares it to what's stored in `dvc.lock`:

```
run signature = hash(cmd + deps_hashes + params_values + outs_hashes)
```

If the current signature matches `dvc.lock`: **skip** (outputs are up to date).
If different or missing: **run** the `cmd`.

### What triggers a stage to re-run

| Change | Which stages re-run |
|---|---|
| Edit `feature_extractor.py` | `extract_features` + all downstream |
| Change `data.n_mels` in `params.yaml` | `extract_features` + `validate` + `train` + ... |
| Add a new audio file to `data/raw/audio` | `ingest` + all downstream |
| Change `training.learning_rate` | `train` only (data stages are unaffected) |
| Run `dvc repro` with nothing changed | Nothing (all stages skipped) |

### Running specific stages

```bash
dvc repro                        # run full pipeline from first stale stage
dvc repro extract_features       # run only this stage (+ its deps if stale)
dvc repro --downstream train     # run train + everything downstream of it
dvc repro --force                # re-run all stages regardless of cache
dvc repro --dry                  # print what would run without running it
```

### The `--dry` flag is your friend

Before touching a production run:

```bash
$ dvc repro --dry
Stage 'ingest' didn't change, skipping
Stage 'normalize_audio' didn't change, skipping
Stage 'extract_features' is outdated:
  changed params: data.n_mels (80 → 100)
Would run: python scripts/run_data_pipeline.py --stage features
Stage 'validate' is outdated (dep 'data/processed/mels' changed)
Would run: python scripts/run_data_pipeline.py --stage validate
...
```

---

## Module 7 — How Our Pipeline Uses DVC

### Stage 1: `ingest`

```yaml
ingest:
  cmd: python scripts/run_data_pipeline.py --stage ingest
  deps:
    - data/raw/audio           # raw WAV files
    - data/raw/transcripts     # raw transcript files
    - src/tts_pipeline/data/ingestion.py
  outs:
    - data/metadata.csv
  params:
    - params.yaml:
        - data.language
        - data.val_ratio       # controls train/val/test split ratio
        - data.test_ratio
```

`run_data_pipeline.py` dispatches to `run_ingest()` which calls `build_metadata()` from `ingestion.py`. The output `data/metadata.csv` is tracked by DVC — the file itself is never committed to Git.

### Stage 6: `validate`

```yaml
validate:
  deps:
    - data/processed/audio_normalized
    - data/processed/mels
    - data/processed/phonemes.csv
    - src/tts_pipeline/data/validator.py
  outs:
    - data/splits/train.txt
    - data/splits/val.txt
    - data/splits/test.txt
    - data/processed/validation/rejected_samples.csv
    - data/processed/validation/phoneme_distribution.json
    - data/processed/validation/duration_histogram.json
    - data/processed/validation/snr_distribution.json
    - data/processed/validation/dataset_scorecard.json
  params:
    - params.yaml:
        - data.min_duration
        - data.max_duration
        - data.min_snr_db
        - data.dataset_grade_threshold
```

The split files (`train.txt`, `val.txt`, `test.txt`) are DVC-tracked outputs. The training stage declares them as deps — so if the validation threshold changes (`min_snr_db: 20 → 25`), the validate stage re-runs, the splits change, and the train stage is automatically invalidated.

### Stage 7: `train`

```yaml
train:
  cmd: python scripts/run_training.py
  deps:
    - data/processed/mels
    - data/processed/f0
    - data/processed/stats.json
    - data/splits/train.txt
    - data/splits/val.txt
    - src/tts_pipeline/models/matcha    # ← entire directory tracked
    - src/tts_pipeline/training
  outs:
    - artifacts/checkpoints/best_model.ckpt
  params:
    - params.yaml:
        - model                          # ← all model.* keys
        - training                       # ← all training.* keys
```

The model checkpoint `best_model.ckpt` is a DVC output. It is never committed to Git — only its hash in `dvc.lock` is. Any model or training parameter change invalidates this stage and forces a full retraining.

### Stage 9: `export`

```yaml
export:
  deps:
    - artifacts/checkpoints/best_model.ckpt
    - artifacts/checkpoints/hifigan_best.ckpt
    - src/tts_pipeline/models
    - reports/latest/evaluation_report.json   # ← export only if eval passed
  outs:
    - artifacts/export/acoustic.onnx
    - artifacts/export/vocoder.onnx
    - artifacts/export/config.json
    - artifacts/export/phoneme_vocab.json
    - artifacts/export/model_card.json
```

`export` depends on `evaluation_report.json` — the evaluation stage must succeed and produce its report before export can run. This enforces that you never export a model that hasn't been evaluated.

---

## Module 8 — Git + DVC Workflow in Practice

### What goes where

```
Git repository (small, text-only):
  dvc.yaml          ← pipeline definition
  dvc.lock          ← fingerprints of everything DVC tracked
  params.yaml       ← all parameters
  src/              ← all Python source code
  scripts/          ← all CLI entry points
  .dvc/config       ← remote storage config (URLs only, no credentials)

DVC remote (large, binary):
  data/metadata.csv
  data/processed/mels/
  data/processed/f0/
  data/splits/train.txt
  artifacts/checkpoints/best_model.ckpt
  artifacts/export/acoustic.onnx
  ... (all outs from dvc.yaml)
```

### The daily workflow

```bash
# 1. Experiment: change a parameter
vim params.yaml   # set data.n_mels: 80 → 100

# 2. Check what will re-run (dry run first)
dvc repro --dry

# 3. Run the pipeline
dvc repro

# 4. Commit both code and data snapshot
git add dvc.lock params.yaml
git commit -m "experiment: n_mels=100"
dvc push          # push new artifacts to remote

# 5. Compare experiments
dvc params diff   # show changed params vs last commit
dvc metrics diff  # show changed metrics vs last commit
```

### Branching and experiment tracking

```bash
# Create an experiment branch
git checkout -b experiment/n-mels-100

# Run and commit
dvc repro
git add dvc.lock
git commit -m "experiment: n_mels=100"
dvc push

# Compare with main branch
git checkout main
dvc params diff experiment/n-mels-100
# Output:
#   Path         Param            HEAD   experiment/n-mels-100
#   params.yaml  data.n_mels      80     100
```

---

## Module 9 — Useful DVC Commands

### Status and debugging

```bash
dvc status                  # show which stages are stale and why
dvc status --cloud          # compare local cache vs remote
dvc dag                     # print the DAG as ASCII tree
dvc dag --md                # print as Markdown (pipe to a file)
```

### Data management

```bash
dvc add data/raw/audio      # start tracking a raw data file/dir
dvc push                    # upload local → remote
dvc pull                    # download remote → local
dvc gc --cloud -w           # delete remote files not referenced by current workspace
dvc gc -w                   # delete local cache files not in current workspace
```

### Params and metrics

```bash
dvc params diff             # compare params vs last git commit
dvc params diff HEAD~3      # compare vs 3 commits ago
dvc metrics show            # display all tracked metric files
dvc metrics diff            # compare metrics vs last commit
```

### Running individual stages

```bash
dvc repro ingest                      # run only ingest
dvc repro -s extract_features         # same (shorthand)
dvc repro --force train               # force re-run train even if inputs unchanged
dvc run -n my_stage -d ... -o ... cmd # define and run a one-off stage
```

---

## Module 10 — Common Pitfalls

### Pitfall 1: Committing DVC-tracked files to Git

If you `git add data/processed/mels/` and commit it, you now have the data in both Git and DVC. The repository bloats, and future `dvc checkout` may conflict with Git-tracked files.

**Fix:** Add the paths to `.gitignore`. DVC automatically adds `.gitignore` entries for its `outs` when you first run the stage.

```
# .gitignore (auto-generated by DVC)
/data/processed/mels
/data/processed/f0
/artifacts/checkpoints/best_model.ckpt
```

### Pitfall 2: Forgetting to commit `dvc.lock` after `dvc repro`

```bash
dvc repro
# ← forgot to commit dvc.lock
git push
dvc push

# Teammate pulls:
git pull              # gets code but old dvc.lock
dvc pull              # fetches data matching OLD dvc.lock — wrong data!
```

**Fix:** Always commit `dvc.lock` immediately after `dvc repro`. CI should fail if `dvc.lock` is not up to date with the committed `dvc.yaml` and params.

### Pitfall 3: Editing `dvc.lock` manually

`dvc.lock` is generated by DVC. Manually editing it corrupts the hash ledger and causes `dvc repro` to either skip stages that should run or re-run everything unnecessarily.

### Pitfall 4: Source code not listed as a dep

```yaml
# BAD — editing feature_extractor.py doesn't trigger re-run
extract_features:
  deps:
    - data/processed/audio_normalized
    # ← missing src/tts_pipeline/data/feature_extractor.py

# GOOD
  deps:
    - data/processed/audio_normalized
    - src/tts_pipeline/data/feature_extractor.py
```

Always list the source files a stage reads. For stages that read an entire module directory (like `train`), list the directory:

```yaml
deps:
  - src/tts_pipeline/models/matcha    # entire directory tracked
```

### Pitfall 5: Params not listed but used at runtime

```yaml
# BAD — code reads data.n_mels from params.yaml at runtime
#         but dvc.yaml doesn't list it
extract_features:
  params:
    - params.yaml:
        - data.sample_rate
        # ← missing data.n_mels
```

If `n_mels` changes, DVC won't know the output is stale. You get cached (wrong) outputs silently. Always keep `params:` in sync with what the code actually reads.

---

## Module 11 — Mini Practice Project: `dvctoy`

An independent mini-project that builds a 3-stage DVC pipeline on simple numerical data — no TTS code. The goal is to understand the full DVC loop (define → run → lock → change → re-run) on a project small enough to fit in one terminal session.

**Goal:** Process a synthetic CSV through three stages: `generate` → `transform` → `summarize`. Track params, track outputs, observe which stages re-run when params change.

### Project structure

```
dvctoy/
├── dvc.yaml
├── params.yaml
├── scripts/
│   ├── generate.py     # stage 1: generate synthetic data
│   ├── transform.py    # stage 2: apply scaling and noise
│   └── summarize.py    # stage 3: compute and print stats
└── data/               # DVC will auto-create .gitignore here
```

### `params.yaml`

```yaml
generate:
  n_samples: 1000
  seed: 42

transform:
  scale: 2.0
  noise_std: 0.1

summarize:
  n_bins: 10
```

### `dvc.yaml`

```yaml
stages:
  generate:
    cmd: python scripts/generate.py
    deps:
      - scripts/generate.py
    outs:
      - data/raw.csv
    params:
      - params.yaml:
          - generate.n_samples
          - generate.seed

  transform:
    cmd: python scripts/transform.py
    deps:
      - data/raw.csv
      - scripts/transform.py
    outs:
      - data/transformed.csv
    params:
      - params.yaml:
          - transform.scale
          - transform.noise_std

  summarize:
    cmd: python scripts/summarize.py
    deps:
      - data/transformed.csv
      - scripts/summarize.py
    outs:
      - data/summary.json
    params:
      - params.yaml:
          - summarize.n_bins
```

### `scripts/generate.py`

```python
import csv
import math
import random
import yaml
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["generate"]
random.seed(params["seed"])
n = params["n_samples"]

Path("data").mkdir(exist_ok=True)
with open("data/raw.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for i in range(n):
        x = i / n * 2 * math.pi
        y = math.sin(x) + random.gauss(0, 0.05)
        writer.writerow([x, y])

print(f"Generated {n} samples → data/raw.csv")
```

### `scripts/transform.py`

```python
import csv
import random
import yaml
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["transform"]
scale = params["scale"]
noise_std = params["noise_std"]

rows = list(csv.DictReader(open("data/raw.csv")))
with open("data/transformed.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["x", "y"])
    writer.writeheader()
    for row in rows:
        y_new = float(row["y"]) * scale + random.gauss(0, noise_std)
        writer.writerow({"x": row["x"], "y": y_new})

print(f"Transformed {len(rows)} rows with scale={scale}, noise_std={noise_std}")
```

### `scripts/summarize.py`

```python
import csv
import json
import yaml
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["summarize"]
n_bins = params["n_bins"]

rows = list(csv.DictReader(open("data/transformed.csv")))
values = [float(r["y"]) for r in rows]
lo, hi = min(values), max(values)
bin_width = (hi - lo) / n_bins
bins = [0] * n_bins
for v in values:
    idx = min(int((v - lo) / bin_width), n_bins - 1)
    bins[idx] += 1

summary = {
    "n": len(values),
    "mean": sum(values) / len(values),
    "min": lo,
    "max": hi,
    "histogram": bins,
}
json.dump(summary, open("data/summary.json", "w"), indent=2)
print(f"Summary: mean={summary['mean']:.4f}, min={summary['min']:.4f}, max={summary['max']:.4f}")
```

### Running the project

```bash
# Setup
pip install dvc pyyaml
git init dvctoy && cd dvctoy
dvc init

# Copy the files above into place, then:
dvc repro
```

Expected output:

```
Running stage 'generate':
> python scripts/generate.py
Generated 1000 samples → data/raw.csv

Running stage 'transform':
> python scripts/transform.py
Transformed 1000 rows with scale=2.0, noise_std=0.1

Running stage 'summarize':
> python scripts/summarize.py
Summary: mean=0.0032, min=-2.2341, max=2.1987
```

### Exercises

1. **Observe caching** — run `dvc repro` a second time. All stages should be skipped. Then check `dvc status` — it should report "Data and pipelines are up to date".

2. **Trigger a partial re-run** — change `transform.scale: 2.0 → 3.0` in `params.yaml`. Run `dvc repro --dry` first. Observe that only `transform` and `summarize` are listed (not `generate` — its inputs didn't change). Then run `dvc repro`.

3. **Trigger a full re-run** — change `generate.seed: 42 → 99`. Run `dvc repro --dry`. All three stages should be listed. Run `dvc repro`.

4. **Commit the lock** — after exercise 3, run:
   ```bash
   git add dvc.yaml dvc.lock params.yaml scripts/
   git commit -m "experiment: seed=99"
   ```
   Then change seed back to 42, run `dvc repro`, commit again. Now run `git log --oneline` — you have two reproducible experiment snapshots in Git.

5. **Add a source code dep** — edit `scripts/transform.py` to add a `clamp` step (clip values to [-3, 3]). Before adding `scripts/transform.py` as a dep, check if `dvc status` detects the change. Then add it as a dep in `dvc.yaml` and verify it works correctly.

6. **Set up a local remote** — run:
   ```bash
   dvc remote add -d localremote /tmp/dvctoy-remote
   dvc push
   ```
   Delete `data/` and run `dvc pull`. Your data should reappear.

---

## Key Takeaways

- DVC separates code (Git) from data (remote storage). `dvc.lock` in Git links the two — it records which data hash corresponds to which code commit.
- A stage's `deps`, `outs`, and `params` fully describe what it reads and writes. DVC hashes all three to decide if a stage is stale.
- `params.yaml` is the single source of truth — DVC tracks specific dotted-path keys, so changing an unrelated param doesn't invalidate unrelated stages.
- `dvc repro --dry` shows what would run without running it. Use it before touching production.
- Source code files must be listed in `deps` or code changes won't trigger re-runs.
- `dvc.lock` is auto-generated. Always commit it after `dvc repro`. Never edit it by hand.
- The team workflow: `dvc repro` → `git add dvc.lock` → `git commit` → `dvc push` → `git push`. Pull: `git pull` → `dvc pull`.