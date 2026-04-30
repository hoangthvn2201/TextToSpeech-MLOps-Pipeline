from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from tts_pipeline.serving.schemas import SynthesizeRequest, SynthesizeResponse

router = APIRouter()


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: Request, body: SynthesizeRequest) -> SynthesizeResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        wav_bytes, rtf = engine.synthesize(
            body.text,
            n_steps=body.n_steps,
            length_scale=body.length_scale,
            speaker_id=body.speaker_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Fire-and-forget shadow inference
    shadow_engine = getattr(request.app.state, "shadow_engine", None)
    if shadow_engine is not None and shadow_engine.is_loaded:
        import asyncio

        asyncio.create_task(_run_shadow(shadow_engine, body.text, body.n_steps))

    duration_sec = len(wav_bytes) / (engine.sample_rate * 2)  # 16-bit PCM
    return SynthesizeResponse(
        audio_bytes_b64=base64.b64encode(wav_bytes).decode(),
        duration_sec=duration_sec,
        rtf=rtf,
        sample_rate=engine.sample_rate,
    )


@router.post("/synthesize/wav")
async def synthesize_wav(request: Request, body: SynthesizeRequest) -> Response:
    """Return raw WAV bytes (Content-Type: audio/wav) for direct playback."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    wav_bytes, _ = engine.synthesize(body.text, n_steps=body.n_steps, length_scale=body.length_scale)
    return Response(content=wav_bytes, media_type="audio/wav")


async def _run_shadow(shadow_engine: object, text: str, n_steps: int) -> None:
    try:
        import mlflow

        wav_bytes, rtf = shadow_engine.synthesize(text, n_steps=n_steps)  # type: ignore[union-attr]
        if mlflow.active_run():
            mlflow.log_metric("shadow_rtf", rtf)
    except Exception:
        pass
