from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from tts_pipeline.utils.audio_utils import save_audio
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


try:
    import lightning as L

    class AudioLoggerCallback(L.Callback):
        """Synthesizes N validation sentences and logs audio + mel images to MLflow."""

        def __init__(self, val_texts: list[str], tokenizer: Any, n_samples: int = 5, sample_rate: int = 22050) -> None:
            self._val_texts = val_texts[:n_samples]
            self._tokenizer = tokenizer
            self._sample_rate = sample_rate

        def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if not mlflow.active_run():
                return
            pl_module.eval()
            with torch.no_grad():
                for i, text in enumerate(self._val_texts):
                    try:
                        phonemes = self._tokenizer.encode(text.split())
                        phoneme_ids = torch.tensor([phonemes], dtype=torch.long, device=pl_module.device)
                        phoneme_lengths = torch.tensor([len(phonemes)], device=pl_module.device)
                        mel = pl_module.model.synthesize(phoneme_ids, phoneme_lengths, n_steps=10)
                        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                            np.save(f.name, mel.cpu().numpy())
                            mlflow.log_artifact(f.name, artifact_path=f"val_mels/epoch_{trainer.current_epoch}")
                    except Exception as exc:
                        logger.warning("Audio logging failed for sample %d: %s", i, exc)
            pl_module.train()

    class MLflowMetricsCallback(L.Callback):
        """Ensures all logged metrics reach MLflow (supplements Lightning's MLflow logger)."""

        def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if mlflow.active_run():
                metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
                mlflow.log_metrics(metrics, step=trainer.global_step)

except ImportError:
    pass
