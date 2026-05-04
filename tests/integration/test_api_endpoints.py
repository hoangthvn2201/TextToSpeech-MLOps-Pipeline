"""Integration tests for the FastAPI serving endpoints."""
from __future__ import annotations

import base64
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tts_pipeline.serving.app import create_app
from tts_pipeline.utils.audio_utils import audio_to_wav_bytes


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.is_loaded = True
    engine.sample_rate = 22050
    dummy_audio = np.zeros(22050, dtype=np.float32)  # 1 second of silence
    wav_bytes = audio_to_wav_bytes(dummy_audio, 22050)
    engine.synthesize.return_value = (wav_bytes, 0.05)
    engine.synthesize_mel.return_value = np.zeros((1, 80, 100), dtype=np.float32)
    engine.vocode.return_value = np.zeros(25600, dtype=np.float32)
    return engine


@pytest.fixture
def client(mock_engine: MagicMock) -> Generator[TestClient, None, None]:
    app = create_app.__wrapped__ if hasattr(create_app, "__wrapped__") else None
    app = create_app(bundle_path="/nonexistent/path")  # engine mocked below

    # Override state after app creation
    with TestClient(app) as c:
        c.app.state.engine = mock_engine  # type: ignore[union-attr]
        c.app.state.shadow_engine = None  # type: ignore[union-attr]
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "model_loaded" in data


def test_synthesize_returns_audio(client: TestClient) -> None:
    resp = client.post("/synthesize", json={"text": "Hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert "audio_bytes_b64" in data
    assert "rtf" in data
    audio_bytes = base64.b64decode(data["audio_bytes_b64"])
    assert len(audio_bytes) > 0


def test_synthesize_empty_text_rejected(client: TestClient) -> None:
    resp = client.post("/synthesize", json={"text": "   "})
    assert resp.status_code == 422


def test_synthesize_wav_endpoint(client: TestClient) -> None:
    resp = client.post("/synthesize/wav", json={"text": "test"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 0
