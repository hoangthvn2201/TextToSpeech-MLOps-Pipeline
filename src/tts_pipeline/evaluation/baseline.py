from __future__ import annotations

from typing import Any

from tts_pipeline.utils.logging import get_logger
from tts_pipeline.utils.registry import get_latest_version, load_model

logger = get_logger(__name__)


def load_baseline(model_name: str) -> tuple[Any, str] | tuple[None, None]:
    """Load Production model for delta comparison. Returns (model, version) or (None, None)."""
    version = get_latest_version(model_name, stage="Production")
    if version is None:
        logger.info("No Production baseline found for %s — absolute floor only", model_name)
        return None, None
    model = load_model(model_name, stage="Production")
    logger.info("Loaded baseline model %s v%s", model_name, version)
    return model, version
