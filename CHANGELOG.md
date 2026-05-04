# Changelog

## [0.1.0] — 2026-04-30

### Initial pipeline scaffold — full greenfield build

#### Project foundation
- `pyproject.toml` — single source for ruff, black, mypy, pytest config; optional extras `[train]`, `[serve]`, `[dev]`
- `requirements/` — split into `base.txt`, `train.txt`, `serve.txt`, `dev.txt`
- `Makefile` — targets: `install`, `lint`, `format`, `typecheck`, `test`, `data`, `train`, `evaluate`, `export`, `serve`, `docker-*`, `mlflow-ui`
- `.gitignore`, `.dvcignore`, `.pre-commit-config.yaml`, `.ruff.toml`
- `README.md` — architecture overview, quick start, pipeline stages, dataset requirements, API endpoints

#### Pipeline backbone
- `params.yaml` — single parameter source of truth consumed by both DVC (dependency tracking) and Hydra (config composition)
- `dvc.yaml` — 9 reproducible pipeline stages: `ingest → normalize_audio → clean_transcripts → phonemize → extract_features → validate → train → train_vocoder → evaluate → export`
- `configs/` — full Hydra config tree:
  - `config.yaml` (root, composes defaults list)
  - `data/base.yaml`, `data/vi.yaml` (Vietnamese phoneme inventory)
  - `model/matcha_tts.yaml`, `model/hifigan.yaml`
  - `training/base.yaml`, `training/finetune.yaml`
  - `evaluation/metrics.yaml` (two-tier gate: absolute floor + relative delta)
  - `serving/api.yaml`

#### Source package (`src/tts_pipeline/`)

**data/**
- `ingestion.py` — raw scan → `metadata.csv`, SHA256 dedup, hash-based deterministic splits
- `audio_processor.py` — resample (torchaudio), mono, VAD silence trim, peak normalize; parallelized via `ProcessPoolExecutor`
- `transcript_cleaner.py` — Unicode NFC, number expansion, punctuation normalization, length filter
- `feature_extractor.py` — mel spectrogram + F0 (PYIN/torchcrepe) + energy → `.npy`; computes `stats.json` for normalization
- `validator.py` — QA gate with full **data lineage artifacts**: `rejected_samples.csv`, `phoneme_distribution.json`, `duration_histogram.json`, `snr_distribution.json`, `dataset_scorecard.json` (graded A–F); halts pipeline if grade below threshold
- `dataset.py` — `TTSDataset` + `TTSDataModule` (Lightning), feature cache key logic

**text/**
- `base_phonemizer.py` — abstract `BasePhonemizer` G2P interface; the only language-specific plug-in point in the entire system
- `phonemizers/espeak_phonemizer.py` — eSpeak-NG backend (50+ languages)
- `phonemizers/vi_phonemizer.py` — Vietnamese-specific rule-based G2P with tone mark detection
- `cleaners.py`, `symbols.py`, `tokenizer.py` — text normalization pipeline; `Tokenizer` with save/load for model bundle

**models/**
- `matcha/encoder.py` — Transformer phoneme encoder with sinusoidal PE, conv-FFN layers, optional speaker embedding hook
- `matcha/duration_predictor.py` — convolutional duration predictor + `regulate()` for upsampling encoder output
- `matcha/flow.py` — `ConditionalFlowMatcher` (OT-CFM): samples `x_t`, computes target vector field, Euler ODE integration at inference
- `matcha/decoder.py` — U-Net flow-matching decoder with sinusoidal time embedding, residual blocks, conditioning on regulated encoder output
- `matcha/model.py` — `MatchaTTS` (nn.Module) + `MatchaTTSLightning` (LightningModule) with **auto-resume** from latest checkpoint
- `vocoder/hifigan.py` — HiFi-GAN V1 generator + `HiFiGANLightning`

**training/**
- `trainer.py` — Lightning Trainer setup, MLflow experiment logging, DVC commit hash tagging, model registration to `Staging`
- `callbacks.py` — `AudioLoggerCallback` (logs synthesized val samples to MLflow), `MLflowMetricsCallback`
- `losses.py` — flow matching loss, duration loss (masked MSE on log scale), mel reconstruction, feature matching

**evaluation/**
- `synthesizer.py` — batch test-set inference; idempotent (skips if outputs exist)
- `baseline.py` — loads Production model from MLflow registry for delta comparison
- `metrics/utmos.py` — UTMOS MOS prediction via speechmos
- `metrics/intelligibility.py` — CER via OpenAI Whisper + jiwer
- `metrics/pesq_stoi.py` — PESQ (wideband) + STOI with automatic resampling
- `metrics/rtf.py` — RTF statistics (mean, p50, p95, max)
- `reporter.py` — **two-tier evaluation gate**: absolute floor (language-agnostic safety net) + relative delta vs Production baseline; HTML + JSON report; MLflow artifact logging; promotes `Staging → Shadow` on pass

**serving/**
- `inference_engine.py` — `ONNXInferenceEngine`: loads `acoustic.onnx` + `vocoder.onnx`, runs full text→audio pipeline
- `routes/tts.py` — `POST /synthesize` (full WAV, base64), `POST /synthesize/wav` (raw bytes); fire-and-forget shadow inference
- `routes/tts_stream.py` — `GET /synthesize_stream`: sentence-chunked `StreamingResponse` for low first-chunk latency (TTFA)
- `routes/health.py` — `/health`, `/ready` (model loaded check)
- `schemas.py` — Pydantic request/response models with validation
- `middleware.py` — request logging middleware
- `app.py` — FastAPI factory with lifespan (loads ONNX engine on startup), CORS, shadow model support

**utils/**
- `config.py` — Pydantic models (`DataConfig`, `ModelConfig`, `TrainingConfig`, `EvaluationConfig`, `PipelineConfig`) + `from_hydra()` converter
- `audio_utils.py` — load/save audio, WAV bytes conversion, SNR estimation, WAV header for streaming
- `logging.py` — structured logger factory
- `registry.py` — MLflow Model Registry client wrapper (register, transition stage, load model)

#### Scripts (`scripts/`)
- `run_data_pipeline.py` — CLI dispatcher for all 6 data stages (called by `dvc.yaml`)
- `run_training.py` — Hydra entrypoint; builds tokenizer from corpus, runs `run_training()`
- `run_evaluation.py` — loads Staging model, runs full eval, promotes to Shadow on pass; idempotent synthesis
- `export_model.py` — ONNX export with explicit dynamic axes (opset 17), int8 post-export quantization, ONNX validation, `model_card.json`
- `serve.py` — uvicorn launcher

#### Prefect pipelines (`pipelines/`)
- `data_pipeline.py`, `training_pipeline.py`, `evaluation_pipeline.py` — thin `@flow` wrappers calling `dvc repro <stage>`
- **Design principle:** DVC owns execution logic; Prefect owns scheduling, retry, and monitoring — no dual control-plane

#### Docker (`docker/`)
- `Dockerfile.train` — `pytorch/pytorch:2.3.0-cuda12.1` base, includes espeak-ng
- `Dockerfile.serve` — `python:3.11-slim` base, ONNX Runtime only (~300 MB total)
- `docker-compose.yml` — three services: `mlflow` (SQLite backend), `prefect` (server), `api` (TTS serving)

#### CI/CD (`.github/workflows/`)
- `ci.yml` — ruff lint, black check, mypy strict, unit tests + coverage, integration smoke, DVC dag lint; runs on every PR
- `pipeline.yml` — full `dvc repro` on GPU self-hosted runner on merge to main; eval gate check; commits updated `dvc.lock`
- `promote.yml` — triggered after pipeline success; requires manual approval via GitHub Environments; transitions `Shadow → Production`; dispatches re-deploy event

#### Tests (`tests/`)
- `conftest.py` — session-scoped synthetic dataset: 50 sine-wave utterances + transcripts; `Tokenizer` fixture
- **Unit tests:** `test_audio_processor.py`, `test_phonemizer.py`, `test_tokenizer.py`, `test_feature_extractor.py`, `test_model_forward.py`, `test_schemas.py`, `test_validator.py`
- **Integration tests:** `test_data_pipeline_e2e.py` (all 6 stages on synthetic data), `test_training_smoke.py` (1 forward+backward pass CPU), `test_api_endpoints.py` (health, ready, synthesize, stream — mocked engine)

#### Key architectural decisions recorded
1. **DVC = execution graph; Prefect = scheduler/monitor** — eliminates dual control-plane
2. **`params.yaml` as single parameter source** — DVC + Hydra both read it; no duplication
3. **Two-tier evaluation gate** — absolute floor (language-agnostic) + relative delta vs Production baseline (handles language/dataset variance)
4. **Model bundle as deployment unit** — `artifacts/export/{acoustic.onnx, vocoder.onnx, config.json, phoneme_vocab.json, model_card.json}`; serving image never imports PyTorch
5. **`BasePhonemizer` as the only language-specific runtime component** — swap language = new subclass + new config + new data
6. **Deployment: Staging → Shadow → Production** — shadow runs fire-and-forget on real traffic; canary (5% routing) deferred to v2 (requires Kubernetes/Istio)
7. **Checkpoint auto-resume** — trainer detects existing checkpoints on startup; all data stages are idempotent
