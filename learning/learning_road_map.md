# Learning Roadmap: Senior MLOps for Text-to-Speech

This roadmap is split into **8 phases** and **15 courses**, ordered so each course builds on the last. The project uses all of this — nothing here is filler.

## Phase 1 — Python Engineering Foundations

### Course 1: Python for ML Engineers

**What the project uses it for:** Packaging (`pyproject.toml`), optional deps `[train]/[serve]/[dev]`, entry points, type hints everywhere with `mypy --strict`, Pydantic v2 config models, dataclasses, `ProcessPoolExecutor` parallelism.

**Topics to cover:**

* Type hints: generics, `Protocol`, `TypeVar`, `Literal`, `TypedDict`
* Pydantic v2: field validation, enum types, custom validators, nested models
* `pyproject.toml`: packaging, optional deps, entry points (`tts-train`, `tts-serve`, etc.)
* `ProcessPoolExecutor` and `concurrent.futures` for parallel data processing
* `pathlib.Path` over `os.path` everywhere
* `abc.ABC` and abstract base classes (used for `BasePhonemizer`)

### Course 2: Software Engineering Practices for ML

**What the project uses it for:** `ruff`, `black`, `mypy`, pre-commit hooks, `pytest` with markers, `Makefile` targets, code coverage.

**Topics to cover:**

* `ruff` rules and configuration (`E`, `F`, `W`, `I`, `N`, `UP`, `B`, `C4`, `SIM`, `RUF`)
* `black` code formatting, pre-commit hooks setup
* `mypy` strict mode: how to annotate functions, generics, and third-party stubs
* `pytest`: fixtures, `tmp_path_factory`, `pytest.mark.slow` / `.gpu`, `pytest-asyncio`, `pytest-cov`
* Writing unit vs integration tests for ML code
* `Makefile` as a developer interface for complex projects

## Phase 2 — Digital Signal Processing & Audio ML

### Course 3: Digital Signal Processing for Audio

**What the project uses it for:** Mel-spectrogram extraction (80 bins, 1024 FFT, 256 hop), SNR estimation, resampling to 22050 Hz, silence trimming (VAD), peak normalization.

**Topics to cover:**

* Sampling rate, Nyquist theorem, resampling
* FFT, STFT, and their parameters (window, hop, `n_fft`)
* Mel scale and mel-filterbank construction
* Log-mel spectrogram as neural network input
* Signal-to-Noise Ratio (SNR) estimation techniques
* Voice Activity Detection (VAD)
* Peak normalization and dynamic range

### Course 4: Audio ML with PyTorch (`torchaudio` + `librosa`)

**What the project uses it for:** Audio loading, resampling, feature extraction, CREPE F0 estimation (`torchcrepe`), PYIN F0 fallback (`librosa`), WAV writing (`soundfile`).

**Topics to cover:**

* `torchaudio.load`, `torchaudio.transforms` (`Resample`, `MelSpectrogram`, `AmplitudeToDB`)
* `librosa.pyin` for fundamental frequency extraction
* `torchcrepe` for GPU-accelerated F0 with CREPE model
* `soundfile` for lossless read/write
* `scipy` for signal utilities
* Audio augmentation patterns for TTS data

## Phase 3 — Deep Learning

### Course 5: Deep Learning with PyTorch

**What the project uses it for:** Every neural network in the project — Matcha-TTS encoder, duration predictor, flow decoder, HiFi-GAN vocoder. All built as `nn.Module` subclasses.

**Topics to cover:**

* Tensors, autograd, and the computation graph
* `nn.Module`: layers, parameters, `forward()`
* Convolutional, linear, and normalization layers
* Sinusoidal positional encoding (used in phoneme encoder)
* Attention mechanisms (Transformer encoder in Matcha-TTS)
* Optimizers, schedulers, gradient clipping
* Mixed precision training (`torch.amp`, `bfloat16`)
* `torch.compile` for A100/H100 acceleration

### Course 6: PyTorch Lightning for Training Orchestration

**What the project uses it for:** `LightningModule` wraps training loop, validation step, callbacks (early stopping, `ModelCheckpoint` top-k=3), distributed training.

**Topics to cover:**

* `LightningModule`: `training_step`, `validation_step`, `configure_optimizers`
* Trainer flags: `max_epochs`, `gradient_clip_val`, `accumulate_grad_batches`, `precision`
* Built-in callbacks: `EarlyStopping`, `ModelCheckpoint`
* Custom callbacks (the project uses them for audio logging to MLflow)
* `LightningDataModule` for clean data separation
* Learning rate warmup schedulers

## Phase 4 — Text-to-Speech Architecture

### Course 7: Text Processing for TTS

**What the project uses it for:** Full pipeline — Unicode normalization → number expansion (`inflect`) → punctuation cleanup → G2P with `phonemizer` (eSpeak-NG) → Vietnamese-specific ViPhonemizer → phoneme tokenizer with `_PAD_`/`_BOS_`/`_EOS_`/`_BLANK_` tokens.

**Topics to cover:**

* Unicode normalization (NFC, NFC vs NFD vs NFKC)
* Text cleaning: control chars, smart quotes, em-dashes, ellipsis
* Number-to-word expansion with `inflect`
* Grapheme-to-phoneme (G2P): what it is, why it exists
* eSpeak-NG and the `phonemizer` Python library
* IPA phoneme sets and language coverage
* Building a `BasePhonemizer` abstract interface for language swap
* Phoneme tokenizer: vocabulary, special tokens, blank insertion (CTC-style)

### Course 8: Acoustic Modeling — Flow Matching & Matcha-TTS

**What the project uses it for:** The core TTS acoustic model. Phoneme encoder → duration predictor → conditional flow matching decoder producing mel spectrograms.

**Topics to cover:**

* Encoder-decoder TTS architecture overview (FastSpeech, VITS lineage)
* Transformer-based phoneme encoder with positional encoding
* Duration predictor: what it predicts and why (phoneme alignment)
* Stochastic Duration Predictor (SDP)
* Flow-based generative models: normalizing flows basics
* Conditional Flow Matching (CFM): the core idea vs diffusion
* Optimal Transport CFM (OT-CFM): why it's more efficient
* ODE solvers: `dopri5` (training) vs Euler (fast inference)
* U-Net decoder with time conditioning
* Training losses: flow matching MSE, duration masked MSE, mel L1

### Course 9: Vocoder — HiFi-GAN

**What the project uses it for:** Converts mel spectrograms → raw waveform audio. V1 architecture with multi-receptive field fusion.

**Topics to cover:**

* Vocoders: WaveNet → WaveGlow → HiFi-GAN progression
* GAN basics: generator vs discriminator, adversarial loss
* HiFi-GAN generator: transposed convolution upsampling, residual blocks, multi-dilation
* Multi-period discriminator (MPD) and multi-scale discriminator (MSD)
* Feature matching loss: why it stabilizes GAN training
* Mel reconstruction loss (L1) as auxiliary objective
* Common GAN training pitfalls and how HiFi-GAN avoids them

## Phase 5 — MLOps Toolchain

### Course 10: Data Version Control with DVC

**What the project uses it for:** 9-stage reproducible data pipeline (`dvc.yaml`), `dvc.lock` checksums, `params.yaml` as single source of truth, S3 remote storage for artifacts.

**Topics to cover:**

* DVC concepts: stages, deps, outs, params, cache
* Writing `dvc.yaml`: stages, commands, dependencies, outputs
* `params.yaml` integration (double-reading with Hydra)
* `dvc repro` and smart cache invalidation
* DVC remote storage: S3, GCS, local
* `dvc push` / `dvc pull` for artifact sharing
* `dvc.lock` for reproducibility guarantees
* DAG visualization (`dvc dag`)

### Course 11: Experiment Tracking & Model Registry with MLflow

**What the project uses it for:** Logging runs (params, metrics, artifacts), model versioning in registry, Staging → Shadow → Production stage transitions, loading Production baseline for regression gating.

**Topics to cover:**

* MLflow tracking: experiments, runs, `log_param`, `log_metric`, `log_artifact`
* `mlflow.pytorch` / `mlflow.onnx` model flavors
* Model Registry: register, version, stage transitions
* Loading production model for baseline comparison
* MLflow server setup (SQLite backend, artifact store)
* Integration with `docker-compose` for local dev stack
* Auto-logging vs manual logging tradeoffs

### Course 12: Workflow Orchestration with Prefect

**What the project uses it for:** `@flow` + `@task` wrappers around DVC stages with retry logic (2–3 retries), scheduling the pipeline, monitoring without duplicating DVC's execution logic.

**Topics to cover:**

* Prefect 2.x concepts: flows, tasks, deployments
* `@flow` and `@task` decorators, task inputs/outputs
* Retry policies and exponential backoff
* Running DVC from Prefect without dual control-plane
* Prefect server: self-hosted setup, UI dashboard
* Scheduling: interval, cron, event-based triggers
* Prefect vs Airflow vs simple cron — when to use what

### Course 13: Configuration Management with Hydra + Pydantic

**What the project uses it for:** Config composition from a `configs/` tree (`data/`, `model/`, `training/`, `evaluation/`, `serving/`), CLI overrides, OmegaConf interpolation, Pydantic validation layer on top.

**Topics to cover:**

* Hydra basics: `@hydra.main`, `DictConfig`, OmegaConf
* Config composition with defaults: lists
* Config groups and overriding from CLI (`+training=finetune`)
* Variable interpolation in YAML (`${data.sample_rate}`)
* Hydra multi-run sweeps for hyperparameter search
* Pydantic as a validation layer over Hydra config
* The `params.yaml` pattern: single file read by both DVC and Hydra

## Phase 6 — Model Serving & Deployment

### Course 14: Model Export, Optimization & Serving with ONNX + FastAPI

**What the project uses it for:** Export PyTorch models to ONNX (opset 17, dynamic axes), int8 post-training quantization, dual-model shadow inference, FastAPI REST + streaming endpoints, ONNX Runtime CPU/CUDA provider chain.

**Topics to cover:**

* `torch.onnx.export`: opset version, dynamic axes, input names
* ONNX Runtime: `InferenceSession`, providers (`CUDAExecutionProvider`, `CPUExecutionProvider`)
* int8 dynamic quantization with `onnxruntime.quantization`
* Model bundle pattern: ONNX + config JSON + vocab JSON (no PyTorch at inference)
* FastAPI: `async def` endpoints, Pydantic request/response models, `StreamingResponse`
* CORS middleware, request logging middleware
* `/health` vs `/ready` probe semantics
* Real-Time Factor (RTF): how to compute and interpret it
* Shadow inference pattern: fire-and-forget secondary model for A/B evaluation
* Docker multi-stage build: `Dockerfile.train` (5 GB GPU) vs `Dockerfile.serve` (300 MB CPU)

## Phase 7 — Production MLOps Patterns

### Course 15: CI/CD, Evaluation Gating & Production MLOps

**What the project uses it for:** GitHub Actions workflows (CI lint/test, GPU pipeline run, manual promote gate), two-tier quality gate (absolute floor + relative regression delta), Staging → Shadow → Production canary pattern.

**Topics to cover:**

* GitHub Actions: workflow syntax, job dependencies, `on:` triggers
* Self-hosted GPU runners for `dvc repro` (8-hour timeout)
* Matrix builds for lint + typecheck + unit + integration
* Manual approval step (`environment: protection rules`)
* Repository dispatch for cross-workflow triggers
* Two-tier evaluation gating:

  * Absolute floor: hard minimums (`UTMOS ≥ 3.0`, `CER ≤ 0.20`, `STOI ≥ 0.60`, `RTF p95 ≤ 0.20`)
  * Relative delta: regression vs production baseline (`UTMOS -5%`, `CER +2pp`, `STOI -3%`, `RTF +5%`)
* Speech quality metrics: UTMOS (MOS prediction), CER (via Whisper), PESQ, STOI
* Staging → Shadow → Production promotion pattern
* Data lineage artifacts: SHA256 dedup, rejected samples CSV, scorecard grading (A–F)
* Idempotency as a design principle for ML stages
* Reproducibility patterns: deterministic splits (hash-based), frozen deps, DVC checksums

## Recommended Study Order

* **Phase 1 → Courses 1, 2** — 2–3 weeks — Python engineering
* **Phase 2 → Courses 3, 4** — 2–3 weeks — Audio & DSP
* **Phase 3 → Courses 5, 6** — 3–4 weeks — Deep learning
* **Phase 4 → Courses 7, 8, 9** — 4–6 weeks — TTS architecture
* **Phase 5 → Courses 10–13** — 3–4 weeks — MLOps toolchain
* **Phase 6 → Course 14** — 2–3 weeks — Serving & deployment
* **Phase 7 → Course 15** — 2–3 weeks — CI/CD & production

**Total estimated time:** 18–26 weeks of focused study alongside hands-on work in this codebase.

When you're ready, say **"create content for Course N"** and I'll write the full detailed content — concepts, code examples, exercises, and direct pointers into this project's source files.
