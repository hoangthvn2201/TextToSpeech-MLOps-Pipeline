"""Prefect flow wrapper for the training pipeline."""
from __future__ import annotations

import subprocess

from prefect import flow, get_run_logger, task


def _dvc_repro(stage: str) -> None:
    logger = get_run_logger()
    logger.info("Running: dvc repro %s", stage)
    result = subprocess.run(["dvc", "repro", stage], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dvc repro {stage} failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(retries=1, retry_delay_seconds=60)
def train_acoustic() -> None:
    _dvc_repro("train")


@task(retries=1, retry_delay_seconds=60)
def train_vocoder() -> None:
    _dvc_repro("train_vocoder")


@flow(name="training-pipeline", log_prints=True)
def training_pipeline() -> None:
    train_acoustic()
    train_vocoder()


if __name__ == "__main__":
    training_pipeline()
