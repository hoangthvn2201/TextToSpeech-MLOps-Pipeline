from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.mish(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.mish(x + h)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(embedding)


class FlowMatchingDecoder(nn.Module):
    """U-Net style decoder that predicts the vector field for flow matching.

    Conditions on: (1) encoder output after duration regulation,
                   (2) continuous flow-matching time embedding.
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 256,
        n_layers: int = 4,
        encoder_dim: int = 192,
    ) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(channels)
        self.cond_proj = nn.Conv1d(encoder_dim, channels, 1)
        self.input_proj = nn.Conv1d(n_mels, channels, 1)

        self.encoder_layers = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_layers // 2)]
        )
        self.middle = ResidualBlock(channels)
        self.decoder_layers = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_layers // 2)]
        )
        self.out_proj = nn.Conv1d(channels, n_mels, 1)

    def forward(
        self,
        x_t: torch.Tensor,          # (B, n_mels, T_mel)
        t: torch.Tensor,            # (B,)
        condition: torch.Tensor,    # (B, T_mel, encoder_dim) — regulated encoder output
    ) -> torch.Tensor:
        time_emb = self.time_emb(t)  # (B, channels)

        cond = self.cond_proj(condition.transpose(1, 2))  # (B, channels, T_mel)
        h = self.input_proj(x_t) + cond  # (B, channels, T_mel)
        h = h + time_emb[:, :, None]

        skips: list[torch.Tensor] = []
        for layer in self.encoder_layers:
            h = layer(h)
            skips.append(h)

        h = self.middle(h)

        for layer, skip in zip(self.decoder_layers, reversed(skips)):
            h = layer(h + skip)

        return self.out_proj(h)  # (B, n_mels, T_mel) — predicted vector field
