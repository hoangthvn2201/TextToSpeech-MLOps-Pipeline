from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_metrics(logger: logging.Logger, metrics: dict[str, Any], prefix: str = "") -> None:
    for k, v in metrics.items():
        key = f"{prefix}/{k}" if prefix else k
        logger.info("%s = %s", key, v)
