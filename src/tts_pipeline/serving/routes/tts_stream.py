from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


@router.get("/synthesize_stream")
async def synthesize_stream(
    request: Request,
    text: str,
    n_steps: int = 50,
    length_scale: float = 1.0,
    chunk_sentences: bool = True,
) -> StreamingResponse:
    """Stream audio chunks sentence-by-sentence for reduced first-chunk latency."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sentences = _split_sentences(text) if chunk_sentences else [text]

    async def audio_generator() -> object:
        from tts_pipeline.utils.audio_utils import audio_to_wav_bytes, wav_header_bytes

        # Emit a WAV header first (data size = 0; client handles streaming)
        yield wav_header_bytes(engine.sample_rate)

        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                mel = engine.synthesize_mel(sentence, n_steps=n_steps)
                audio_chunk = engine.vocode(mel)
                import numpy as np

                pcm = (audio_chunk.flatten() * 32767).clip(-32768, 32767).astype(np.int16)
                yield pcm.tobytes()
            except Exception:
                pass  # skip failed sentence, continue streaming

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={"X-Accel-Buffering": "no"},
    )
