from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 4, dim, kernel_size, padding=pad),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x.transpose(1, 2)).transpose(1, 2))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)
        return self.ffn(x)


class PhonemeEncoder(nn.Module):
    """Transformer-based phoneme encoder with optional speaker embedding hook."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 192,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        speaker_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(dim)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, n_heads, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)

        # Speaker embedding hook (stubbed for multi-speaker v2)
        self.speaker_proj: nn.Linear | None = None
        if speaker_embedding_dim > 0:
            self.speaker_proj = nn.Linear(speaker_embedding_dim, dim, bias=False)

    def forward(
        self,
        phoneme_ids: torch.Tensor,          # (B, T_text)
        phoneme_lengths: torch.Tensor,      # (B,)
        speaker_embedding: torch.Tensor | None = None,  # (B, speaker_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = phoneme_ids.shape
        # Build padding mask (True = padding position)
        mask = torch.arange(T, device=phoneme_ids.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(1)

        x = self.embedding(phoneme_ids)
        x = self.pos_enc(x)

        if speaker_embedding is not None and self.speaker_proj is not None:
            x = x + self.speaker_proj(speaker_embedding).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        x = self.norm(x)
        return x, mask  # (B, T_text, dim), (B, T_text)
