from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_pipeline.serving.schemas import SynthesizeRequest


def test_valid_request() -> None:
    req = SynthesizeRequest(text="Hello world")
    assert req.text == "Hello world"
    assert req.n_steps == 50
    assert req.length_scale == 1.0


def test_text_stripped() -> None:
    req = SynthesizeRequest(text="  hello  ")
    assert req.text == "hello"


def test_empty_text_raises() -> None:
    with pytest.raises(ValidationError):
        SynthesizeRequest(text="   ")


def test_text_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        SynthesizeRequest(text="x" * 501)


def test_invalid_length_scale_raises() -> None:
    with pytest.raises(ValidationError):
        SynthesizeRequest(text="test", length_scale=5.0)


def test_invalid_n_steps_raises() -> None:
    with pytest.raises(ValidationError):
        SynthesizeRequest(text="test", n_steps=0)
