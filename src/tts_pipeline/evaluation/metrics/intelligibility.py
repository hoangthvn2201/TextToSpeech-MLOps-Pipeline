from __future__ import annotations

from pathlib import Path

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def compute_cer(
    wav_paths: list[Path],
    references: list[str],
    whisper_model_size: str = "small",
    language: str | None = None,
) -> dict[str, float]:
    """Compute Character Error Rate using OpenAI Whisper transcription."""
    try:
        import whisper
        from jiwer import cer as jiwer_cer

        model = whisper.load_model(whisper_model_size)
        hypotheses = []
        for p in wav_paths:
            try:
                result = model.transcribe(str(p), language=language)
                hypotheses.append(result["text"].strip())
            except Exception as exc:
                logger.warning("Whisper failed for %s: %s", p, exc)
                hypotheses.append("")

        valid = [(h, r) for h, r in zip(hypotheses, references) if r.strip()]
        if not valid:
            return {"cer": 1.0, "n_evaluated": 0}

        hyps, refs = zip(*valid)
        cer_score = float(jiwer_cer(list(refs), list(hyps)))
        return {"cer": cer_score, "n_evaluated": len(valid)}
    except ImportError as e:
        logger.error("Missing dependency for CER: %s", e)
        return {"cer": 1.0, "n_evaluated": 0}
