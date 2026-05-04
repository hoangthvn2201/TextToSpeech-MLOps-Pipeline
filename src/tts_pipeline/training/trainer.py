from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch

from tts_pipeline.utils.logging import get_logger
from tts_pipeline.utils.registry import register_model, transition_stage

logger = get_logger(__name__)


def run_training(cfg: Any, data_module: Any, model_name: str = "matcha-tts") -> str:
    """Run the full training loop and register the best model to MLflow staging.

    Returns the MLflow run_id.
    """
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from tts_pipeline.training.callbacks import AudioLoggerCallback, MLflowMetricsCallback

    checkpoint_dir = Path(cfg.paths.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment.name,
        run_name=cfg.experiment.run_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        tags=dict(cfg.experiment.tags),
    )

    # Build model with auto-resume
    from tts_pipeline.models.matcha.model import MatchaTTSLightning
    from tts_pipeline.utils.config import from_hydra

    pipeline_cfg = from_hydra(cfg)
    model_kw = pipeline_cfg.model.model_dump()
    model_kw["vocab_size"] = data_module.tokenizer.vocab_size if hasattr(data_module, "tokenizer") else 256

    pl_model = MatchaTTSLightning.auto_resume(
        checkpoint_dir=checkpoint_dir,
        model_cfg=model_kw,
        training_cfg=pipeline_cfg.training.model_dump(),
    )

    if cfg.training.use_compile:
        import torch
        pl_model = torch.compile(pl_model)  # type: ignore[assignment]

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best_model",
        monitor="val/loss",
        mode="min",
        save_top_k=cfg.training.save_top_k,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.grad_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=mlflow_logger,
        callbacks=[checkpoint_cb, early_stop_cb, MLflowMetricsCallback()],
        enable_progress_bar=True,
    )

    run_id = mlflow_logger.run_id
    logger.info("MLflow run_id: %s", run_id)

    with mlflow.start_run(run_id=run_id):
        import subprocess, sys
        dvc_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        mlflow.set_tag("dvc_commit", dvc_commit)
        mlflow.set_tag("language", cfg.data.language)

        trainer.fit(pl_model, datamodule=data_module)

        # Register best checkpoint
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt:
            mlflow.pytorch.log_model(pl_model, artifact_path="matcha_tts")
            register_model(
                run_id=run_id,
                artifact_path="matcha_tts",
                model_name=f"matcha-tts-{cfg.data.language}",
                tags={"dvc_commit": dvc_commit, "language": cfg.data.language},
            )
            transition_stage(
                model_name=f"matcha-tts-{cfg.data.language}",
                version="1",
                stage="Staging",
            )

    return run_id
