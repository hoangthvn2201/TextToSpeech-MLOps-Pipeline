from __future__ import annotations

from fastapi import APIRouter, Request

from tts_pipeline.serving.schemas import HealthResponse, ReadyResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    engine = getattr(request.app.state, "engine", None)
    shadow = getattr(request.app.state, "shadow_engine", None)
    loaded = engine is not None and engine.is_loaded
    return ReadyResponse(
        ready=loaded,
        model_loaded=loaded,
        shadow_loaded=shadow is not None and shadow.is_loaded,
    )
