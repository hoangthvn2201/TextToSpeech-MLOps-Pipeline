# TextToSpeech MLOps Pipeline

Production-grade MLOps pipeline for training and shipping a lightweight, high-quality Text-to-Speech model targeting a specific language.

## Architecture

**Model:** Matcha-TTS (flow-matching acoustic model) + HiFi-GAN vocoder
**Stack:** PyTorch + Lightning · DVC · Prefect · MLflow · Hydra · FastAPI · ONNX Runtime

```
Raw data → Data Pipeline → Feature Extraction → Training → Evaluation → ONNX Export → Serving
              (DVC)            (DVC)            (Lightning)  (UTMOS/CER)   (ONNX)      (FastAPI)
```

**Orchestration:** DVC owns the execution graph (`dvc.yaml`). Prefect wraps DVC for scheduling and monitoring. No dual control-plane.

## Quick Start

```bash
# Install
pip install -e ".[dev,train]"
pre-commit install

# Start local infrastructure (MLflow + Prefect)
make docker-up

# Run full pipeline
make pipeline          # dvc repro (all stages)

# Or run stages individually
make data              # data pipeline through validation
make train             # acoustic model + vocoder
make evaluate          # evaluation + pass/fail gate
make export            # ONNX bundle

# Serve
make serve
# or via Docker
make docker-serve
docker run -p 8080:8080 tts-serve:latest
```

## Pipeline Stages

```
ingest → normalize_audio → clean_transcripts → phonemize
                                              ↓
                                      extract_features → validate
                                                             ↓
                                            train ← ─ ─ ─ ─ ┘
                                            train_vocoder
                                                 ↓
                                             evaluate → export
```

All stages are defined in `dvc.yaml` and parameterized via `params.yaml`.

## Adding a New Language

1. Implement `BasePhonemizer` in `src/tts_pipeline/text/phonemizers/<lang>_phonemizer.py`
2. Create `configs/data/<lang>.yaml` with the phoneme inventory
3. Update `params.yaml`: `text.phonemizer: <lang>` and `data.language: <lang>`
4. Add language data to `data/raw/`
5. Run `dvc repro --force`

## Deployment Model

```
Training → Staging → Shadow → Production
```

The serving layer runs both the Production model (live responses) and the Shadow model (fire-and-forget, no user exposure) simultaneously. Promotion to Production requires eval gate pass + shadow comparison pass.

## Dataset Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Total duration | 5 hours | 20+ hours |
| Utterances | 5,000 | 20,000+ |
| Median SNR | > 25 dB | > 35 dB |
| Phoneme coverage | ≥ 10× per phoneme | ≥ 50× |

## Project Structure

```
src/tts_pipeline/   # Importable package
configs/            # Hydra configuration tree
data/               # Raw + processed data (DVC tracked)
pipelines/          # Prefect @flow definitions
scripts/            # CLI entry points
docker/             # Dockerfiles + compose
tests/              # Unit + integration tests
reports/            # Evaluation outputs (DVC tracked)
artifacts/          # Checkpoints + ONNX bundle (DVC tracked)
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/synthesize` | POST | Full audio response (WAV bytes) |
| `/synthesize_stream` | GET | Chunked streaming audio (sentence-by-sentence) |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (model loaded check) |
