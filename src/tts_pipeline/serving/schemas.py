from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Input text to synthesize")
    speaker_id: int | None = Field(default=None, description="Speaker ID for multi-speaker models")
    length_scale: float = Field(default=1.0, ge=0.5, le=2.0, description="Duration scaling factor")
    n_steps: int = Field(default=50, ge=1, le=200, description="ODE solver steps (more = higher quality)")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


class SynthesizeResponse(BaseModel):
    audio_bytes_b64: str = Field(description="Base64-encoded WAV audio")
    duration_sec: float
    rtf: float
    sample_rate: int = 22050


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    shadow_loaded: bool
