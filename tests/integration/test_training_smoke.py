"""Smoke test: 1 batch, 1 step — verifies training runs end-to-end without GPU."""
from __future__ import annotations

import pytest
import torch

from tts_pipeline.models.matcha.model import MatchaTTS


@pytest.mark.slow
def test_training_smoke_cpu() -> None:
    """1 forward + backward pass on CPU with a tiny model."""
    model = MatchaTTS(
        vocab_size=20,
        n_mels=16,
        encoder_dim=32,
        encoder_heads=2,
        encoder_layers=2,
        encoder_kernel_size=3,
        decoder_channels=32,
        decoder_layers=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, T_text, T_mel = 2, 8, 24
    batch = {
        "phoneme_ids": torch.randint(0, 20, (B, T_text)),
        "phoneme_len": torch.tensor([T_text, T_text]),
        "mel": torch.randn(B, 16, T_mel),
        "mel_len": torch.tensor([T_mel, T_mel]),
    }

    model.train()
    optimizer.zero_grad()
    out = model(batch["phoneme_ids"], batch["phoneme_len"], batch["mel"], batch["mel_len"])
    loss = out["loss"]
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()
    optimizer.step()
    assert True, "Training smoke test passed"


def test_synthesize_smoke() -> None:
    model = MatchaTTS(vocab_size=20, n_mels=16, encoder_dim=32, encoder_heads=2,
                      encoder_layers=2, decoder_channels=32, decoder_layers=2)
    model.eval()
    mel = model.synthesize(
        torch.randint(0, 20, (1, 5)),
        torch.tensor([5]),
        n_steps=2,
    )
    assert mel.shape[0] == 1
    assert mel.shape[1] == 16
    assert mel.shape[2] > 0
