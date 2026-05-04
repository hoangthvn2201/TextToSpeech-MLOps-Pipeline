from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

ModelStage = str  # "Staging" | "Shadow" | "Production" | "Archived"


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def register_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    tags: dict[str, str] | None = None,
) -> str:
    """Register a model version from a run and return the version string."""
    client = get_client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags=tags or {},
    )
    logger.info("Registered model %s version %s from run %s", model_name, mv.version, run_id)
    return mv.version


def transition_stage(model_name: str, version: str, stage: ModelStage) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info("Transitioned %s v%s → %s", model_name, version, stage)


def get_latest_version(model_name: str, stage: ModelStage) -> str | None:
    client = get_client()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    return versions[0].version


def load_model(model_name: str, stage: ModelStage) -> Any:
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pytorch.load_model(model_uri)
