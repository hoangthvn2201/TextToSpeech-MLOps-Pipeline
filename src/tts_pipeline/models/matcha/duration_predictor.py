from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """Convolutional duration predictor — predicts log-durations per phoneme."""

    def __init__(self, dim: int = 192, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # applied after transpose
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.conv_layers = nn.ModuleList(layers[::4])
        self.act_layers = nn.ModuleList(layers[1::4])
        self.norm_layers = nn.ModuleList(layers[2::4])
        self.drop_layers = nn.ModuleList(layers[3::4])
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T_text, dim)
        h = x.transpose(1, 2)  # (B, dim, T)
        for conv, act, norm, drop in zip(
            self.conv_layers, self.act_layers, self.norm_layers, self.drop_layers
        ):
            h = act(conv(h))
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = drop(h)
        log_dur = self.proj(h.transpose(1, 2)).squeeze(-1)  # (B, T_text)
        if mask is not None:
            log_dur = log_dur.masked_fill(mask, 0.0)
        return log_dur

    @staticmethod
    def regulate(
        encoder_out: torch.Tensor,    # (B, T_text, dim)
        durations: torch.Tensor,      # (B, T_text) — integer frame counts
    ) -> torch.Tensor:
        """Repeat encoder output according to predicted durations → mel-length tensor."""
        B, T, D = encoder_out.shape
        max_len = int(durations.sum(dim=1).max().item())
        output = torch.zeros(B, max_len, D, device=encoder_out.device)
        for b in range(B):
            idx = 0
            for t in range(T):
                d = int(durations[b, t].item())
                if d > 0:
                    output[b, idx : idx + d] = encoder_out[b, t].unsqueeze(0).expand(d, -1)
                    idx += d
        return output  # (B, T_mel, dim)
