"""Prefect flow wrapper for the data pipeline.

DVC owns the execution logic. Prefect provides scheduling, retry, and monitoring.
"""
from __future__ import annotations

import subprocess

from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash


def _dvc_repro(stage: str) -> None:
    logger = get_run_logger()
    logger.info("Running: dvc repro %s", stage)
    result = subprocess.run(["dvc", "repro", stage], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dvc repro {stage} failed:\n{result.stderr}")
    logger.info(result.stdout)


@task(retries=2, retry_delay_seconds=30)
def ingest() -> None:
    _dvc_repro("ingest")


@task(retries=2, retry_delay_seconds=30)
def normalize_audio() -> None:
    _dvc_repro("normalize_audio")


@task(retries=1)
def clean_transcripts() -> None:
    _dvc_repro("clean_transcripts")


@task(retries=1)
def phonemize() -> None:
    _dvc_repro("phonemize")


@task(retries=1)
def extract_features() -> None:
    _dvc_repro("extract_features")


@task(retries=1)
def validate() -> None:
    _dvc_repro("validate")


@flow(name="data-pipeline", log_prints=True)
def data_pipeline() -> None:
    ingest()
    normalize_audio()
    clean_transcripts()
    phonemize()
    extract_features()
    validate()


if __name__ == "__main__":
    data_pipeline()
