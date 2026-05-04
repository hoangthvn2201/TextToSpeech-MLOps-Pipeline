.PHONY: install install-dev install-train install-serve \
        lint format typecheck test test-unit test-integration \
        data train evaluate export serve \
        docker-train docker-serve docker-up docker-down \
        clean dvc-init dvc-push dvc-pull mlflow-ui

# ── Installation ──────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-train:
	pip install -e ".[train]"

install-serve:
	pip install -e ".[serve]"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/ pipelines/

format:
	black src/ tests/ scripts/ pipelines/
	ruff check --fix src/ tests/ scripts/ pipelines/

typecheck:
	mypy src/tts_pipeline/ --strict

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src/tts_pipeline --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m "not slow"

# ── Pipeline stages ───────────────────────────────────────────────────────────
data:
	dvc repro validate

train:
	dvc repro train train_vocoder

evaluate:
	dvc repro evaluate

export:
	dvc repro export

pipeline:
	dvc repro

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	python scripts/serve.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-train:
	docker build -f docker/Dockerfile.train -t tts-train:latest .

docker-serve:
	docker build -f docker/Dockerfile.serve -t tts-serve:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── DVC ───────────────────────────────────────────────────────────────────────
dvc-init:
	dvc init
	dvc remote add -d storage s3://your-bucket/tts-pipeline

dvc-push:
	dvc push

dvc-pull:
	dvc pull

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
