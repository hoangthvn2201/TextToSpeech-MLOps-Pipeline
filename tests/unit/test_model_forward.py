from __future__ import annotations

import pytest
import torch

from tts_pipeline.models.matcha.model import MatchaTTS


@pytest.fixture
def small_model() -> MatchaTTS:
    return MatchaTTS(
        vocab_size=20,
        n_mels=16,
        encoder_dim=32,
        encoder_heads=2,
        encoder_layers=2,
        encoder_kernel_size=3,
        decoder_channels=32,
        decoder_layers=2,
    )


def test_forward_returns_loss(small_model: MatchaTTS) -> None:
    B, T_text, T_mel = 2, 8, 30
    out = small_model(
        phoneme_ids=torch.randint(0, 20, (B, T_text)),
        phoneme_lengths=torch.tensor([T_text, T_text - 2]),
        mel_targets=torch.randn(B, 16, T_mel),
        mel_lengths=torch.tensor([T_mel, T_mel - 5]),
    )
    assert "loss" in out
    assert not torch.isnan(out["loss"])
    assert out["loss"].requires_grad


def test_synthesize_no_grad(small_model: MatchaTTS) -> None:
    small_model.eval()
    B, T_text = 1, 6
    mel = small_model.synthesize(
        phoneme_ids=torch.randint(0, 20, (B, T_text)),
        phoneme_lengths=torch.tensor([T_text]),
        n_steps=2,
    )
    assert mel.ndim == 3
    assert mel.shape[0] == B
    assert mel.shape[1] == 16  # n_mels
    assert not mel.requires_grad
