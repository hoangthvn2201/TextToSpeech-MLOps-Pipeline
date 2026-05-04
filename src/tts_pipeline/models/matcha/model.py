from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tts_pipeline.models.matcha.decoder import FlowMatchingDecoder
from tts_pipeline.models.matcha.duration_predictor import DurationPredictor
from tts_pipeline.models.matcha.encoder import PhonemeEncoder
from tts_pipeline.models.matcha.flow import ConditionalFlowMatcher
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class MatchaTTS(nn.Module):
    """Matcha-TTS acoustic model — Encoder → Duration → Flow Decoder."""

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        encoder_dim: int = 192,
        encoder_heads: int = 2,
        encoder_layers: int = 6,
        encoder_kernel_size: int = 3,
        decoder_channels: int = 256,
        decoder_layers: int = 4,
        sigma_min: float = 1e-4,
        speaker_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = PhonemeEncoder(
            vocab_size=vocab_size,
            dim=encoder_dim,
            n_heads=encoder_heads,
            n_layers=encoder_layers,
            kernel_size=encoder_kernel_size,
            speaker_embedding_dim=speaker_embedding_dim,
        )
        self.duration_predictor = DurationPredictor(dim=encoder_dim)
        self.flow_decoder = FlowMatchingDecoder(
            n_mels=n_mels,
            channels=decoder_channels,
            n_layers=decoder_layers,
            encoder_dim=encoder_dim,
        )
        self.cfm = ConditionalFlowMatcher(sigma_min=sigma_min)

    def forward(
        self,
        phoneme_ids: torch.Tensor,      # (B, T_text)
        phoneme_lengths: torch.Tensor,  # (B,)
        mel_targets: torch.Tensor,      # (B, n_mels, T_mel)
        mel_lengths: torch.Tensor,      # (B,)
    ) -> dict[str, torch.Tensor]:
        enc_out, enc_mask = self.encoder(phoneme_ids, phoneme_lengths)
        log_durations = self.duration_predictor(enc_out, enc_mask)

        # Ground-truth durations from mel/text length ratio (simple upsampling heuristic)
        text_len = phoneme_lengths.float()
        mel_len = mel_lengths.float()
        dur_ratio = (mel_len / text_len.clamp(min=1)).round().long()
        gt_durations = dur_ratio.unsqueeze(1).expand_as(log_durations).clamp(min=1)

        regulated = DurationPredictor.regulate(enc_out, gt_durations)  # (B, T_mel, dim)
        # Trim/pad to mel_targets length
        T_mel = mel_targets.shape[2]
        if regulated.shape[1] > T_mel:
            regulated = regulated[:, :T_mel, :]
        elif regulated.shape[1] < T_mel:
            pad = torch.zeros(regulated.shape[0], T_mel - regulated.shape[1], regulated.shape[2], device=regulated.device)
            regulated = torch.cat([regulated, pad], dim=1)

        # Sample flow matching time and compute loss
        B = phoneme_ids.shape[0]
        t = torch.rand(B, device=phoneme_ids.device)
        x0 = torch.randn_like(mel_targets)
        x_t, u_t = self.cfm.sample_xt(x0, mel_targets, t)

        pred_vf = self.flow_decoder(x_t, t, regulated)
        flow_loss = ((pred_vf - u_t) ** 2).mean()

        # Duration loss (MSE on log scale)
        log_gt = gt_durations.float().clamp(min=1).log()
        dur_loss = ((log_durations - log_gt) ** 2).masked_fill(enc_mask, 0.0).mean()

        return {
            "loss": flow_loss + dur_loss,
            "flow_loss": flow_loss,
            "dur_loss": dur_loss,
        }

    @torch.no_grad()
    def synthesize(
        self,
        phoneme_ids: torch.Tensor,      # (B, T_text)
        phoneme_lengths: torch.Tensor,  # (B,)
        n_steps: int = 10,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        enc_out, enc_mask = self.encoder(phoneme_ids, phoneme_lengths)
        log_durations = self.duration_predictor(enc_out, enc_mask)
        durations = (log_durations.exp() * length_scale).round().long().clamp(min=1)
        regulated = DurationPredictor.regulate(enc_out, durations)

        T_mel = regulated.shape[1]
        x0 = torch.randn(phoneme_ids.shape[0], self.flow_decoder.out_proj.out_channels, T_mel, device=phoneme_ids.device)
        mel = self.cfm.integrate(self.flow_decoder, x0, regulated, n_steps=n_steps)
        return mel  # (B, n_mels, T_mel)


try:
    import lightning as L

    class MatchaTTSLightning(L.LightningModule):
        """LightningModule wrapper with auto-resume support."""

        def __init__(self, model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.model = MatchaTTS(**model_cfg)
            self._training_cfg = training_cfg

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.model(*args, **kwargs)

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            out = self.model(
                batch["phoneme_ids"],
                batch["phoneme_len"],
                batch["mel"],
                batch["mel_len"],
            )
            self.log_dict({f"train/{k}": v for k, v in out.items()}, prog_bar=True)
            return out["loss"]

        def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
            out = self.model(
                batch["phoneme_ids"],
                batch["phoneme_len"],
                batch["mel"],
                batch["mel_len"],
            )
            self.log_dict({f"val/{k}": v for k, v in out.items()}, prog_bar=True)

        def configure_optimizers(self) -> Any:
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            opt = AdamW(self.parameters(), lr=self._training_cfg["learning_rate"], weight_decay=1e-2)
            warmup = LinearLR(opt, start_factor=0.1, total_iters=self._training_cfg["warmup_steps"])
            cosine = CosineAnnealingLR(opt, T_max=self._training_cfg["max_epochs"])
            scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self._training_cfg["warmup_steps"]])
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        @classmethod
        def auto_resume(cls, checkpoint_dir: Path, model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> "MatchaTTSLightning":
            """Load from latest checkpoint if available, else initialize fresh."""
            checkpoints = sorted(Path(checkpoint_dir).glob("*.ckpt"))
            if checkpoints:
                latest = checkpoints[-1]
                logger.info("Auto-resuming from %s", latest)
                return cls.load_from_checkpoint(latest, model_cfg=model_cfg, training_cfg=training_cfg)
            logger.info("No checkpoint found, initializing fresh model")
            return cls(model_cfg=model_cfg, training_cfg=training_cfg)

except ImportError:
    pass
