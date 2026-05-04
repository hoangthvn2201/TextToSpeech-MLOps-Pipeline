from __future__ import annotations

import pandas as pd

from tts_pipeline.text.cleaners import clean_text
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def clean_metadata(
    df: pd.DataFrame,
    language: str = "vi",
    min_chars: int = 2,
    max_chars: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean transcripts in-place. Returns (accepted_df, rejected_df)."""
    df = df.copy()
    df["text_clean"] = df["text"].apply(lambda t: clean_text(str(t), language=language))

    mask_short = df["text_clean"].str.len() < min_chars
    mask_long = df["text_clean"].str.len() > max_chars
    mask_empty = df["text_clean"].str.strip() == ""

    rejected_mask = mask_short | mask_long | mask_empty
    rejected = df[rejected_mask].copy()
    accepted = df[~rejected_mask].copy()

    if not rejected.empty:
        logger.warning("Rejected %d transcripts (too short/long/empty)", len(rejected))

    return accepted, rejected
