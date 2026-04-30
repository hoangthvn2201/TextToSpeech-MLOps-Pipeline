from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _assign_split(file_id: str, val_ratio: float = 0.05, test_ratio: float = 0.05) -> str:
    """Deterministic hash-based split assignment."""
    h = int(hashlib.md5(file_id.encode()).hexdigest(), 16) % 1000
    if h < test_ratio * 1000:
        return "test"
    if h < (test_ratio + val_ratio) * 1000:
        return "val"
    return "train"


def build_metadata(
    audio_dir: Path,
    transcript_dir: Path,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> pd.DataFrame:
    audio_files = {
        p.stem: p for p in audio_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS
    }
    transcript_files = {
        p.stem: p for p in transcript_dir.rglob("*.txt")
    }

    common = set(audio_files) & set(transcript_files)
    missing_audio = set(transcript_files) - set(audio_files)
    missing_transcript = set(audio_files) - set(transcript_files)

    if missing_audio:
        logger.warning("No audio for %d transcript(s): %s …", len(missing_audio), list(missing_audio)[:3])
    if missing_transcript:
        logger.warning("No transcript for %d audio file(s): %s …", len(missing_transcript), list(missing_transcript)[:3])

    rows = []
    for stem in sorted(common):
        audio_path = audio_files[stem]
        text = transcript_files[stem].read_text(encoding="utf-8").strip()
        try:
            info = sf.info(str(audio_path))
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0
        rows.append(
            {
                "id": stem,
                "audio_path": str(audio_path),
                "text": text,
                "duration": duration,
                "sha256": _sha256(audio_path),
                "split": _assign_split(stem, val_ratio=val_ratio, test_ratio=test_ratio),
            }
        )

    df = pd.DataFrame(rows)
    # Deduplicate by SHA256
    before = len(df)
    df = df.drop_duplicates(subset="sha256", keep="first")
    if len(df) < before:
        logger.info("Removed %d duplicate audio files", before - len(df))

    logger.info(
        "Built metadata: %d samples (train=%d val=%d test=%d)",
        len(df),
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    return df
