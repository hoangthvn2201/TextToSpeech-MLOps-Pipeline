from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tts_pipeline.serving.middleware import RequestLoggingMiddleware
from tts_pipeline.serving.routes import health, tts, tts_stream
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(
    bundle_path: str | Path = "artifacts/export",
    shadow_bundle_path: str | Path | None = None,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        from tts_pipeline.serving.inference_engine import ONNXInferenceEngine

        app.state.engine = ONNXInferenceEngine(Path(bundle_path))
        if shadow_bundle_path:
            app.state.shadow_engine = ONNXInferenceEngine(Path(shadow_bundle_path))
            logger.info("Shadow model loaded from %s", shadow_bundle_path)
        else:
            app.state.shadow_engine = None
        yield
        logger.info("Shutting down inference engine")

    app = FastAPI(
        title="TTS Pipeline API",
        description="Text-to-Speech inference API (Matcha-TTS + HiFi-GAN)",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(tts.router, tags=["synthesis"])
    app.include_router(tts_stream.router, tags=["synthesis"])

    return app


def cli() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run TTS serving API")
    parser.add_argument("--bundle-path", default="artifacts/export")
    parser.add_argument("--shadow-bundle-path", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    app = create_app(args.bundle_path, args.shadow_bundle_path)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
