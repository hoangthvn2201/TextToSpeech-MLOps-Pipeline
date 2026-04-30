"""Prefect flow wrapper for the evaluation pipeline."""
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


@task(retries=1)
def evaluate() -> None:
    _dvc_repro("evaluate")


@task(retries=1)
def export() -> None:
    _dvc_repro("export")


@flow(name="evaluation-pipeline", log_prints=True)
def evaluation_pipeline() -> None:
    evaluate()
    export()


if __name__ == "__main__":
    evaluation_pipeline()
