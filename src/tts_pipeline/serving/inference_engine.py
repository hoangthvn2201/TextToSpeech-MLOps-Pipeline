from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tts_pipeline.text.tokenizer import Tokenizer
from tts_pipeline.utils.audio_utils import audio_to_wav_bytes
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class ONNXInferenceEngine:
    """Loads acoustic + vocoder ONNX models and runs the full synthesis pipeline."""

    def __init__(self, bundle_path: Path) -> None:
        import onnxruntime as ort

        bundle_path = Path(bundle_path)
        self._sample_rate: int = 22050
        self._loaded = False

        config_path = bundle_path / "config.json"
        vocab_path = bundle_path / "phoneme_vocab.json"
        acoustic_path = bundle_path / "acoustic.onnx"
        vocoder_path = bundle_path / "vocoder.onnx"

        if not all(p.exists() for p in [config_path, vocab_path, acoustic_path, vocoder_path]):
            logger.error("Incomplete model bundle at %s", bundle_path)
            return

        with open(config_path) as f:
            config = json.load(f)
        self._sample_rate = config.get("sample_rate", 22050)

        self._tokenizer = Tokenizer.load(vocab_path)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._acoustic = ort.InferenceSession(str(acoustic_path), opts, providers=providers)
        self._vocoder = ort.InferenceSession(str(vocoder_path), opts, providers=providers)
        self._loaded = True
        logger.info("Inference engine loaded from %s (sr=%d)", bundle_path, self._sample_rate)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(
        self,
        text: str,
        n_steps: int = 50,
        length_scale: float = 1.0,
        speaker_id: int | None = None,
    ) -> tuple[bytes, float]:
        """Synthesize text → WAV bytes. Returns (wav_bytes, rtf)."""
        from tts_pipeline.text.cleaners import clean_text

        clean = clean_text(text)
        phoneme_ids = self._tokenizer.encode(clean.split())
        phoneme_array = np.array([phoneme_ids], dtype=np.int64)
        phoneme_lengths = np.array([len(phoneme_ids)], dtype=np.int64)

        t0 = time.perf_counter()
        mel_out = self._acoustic.run(
            None,
            {"phoneme_ids": phoneme_array, "phoneme_lengths": phoneme_lengths},
        )[0]  # (1, n_mels, T_mel)

        audio_out = self._vocoder.run(None, {"mel": mel_out})[0]  # (1, 1, T_audio)
        elapsed = time.perf_counter() - t0

        audio = audio_out.squeeze()
        duration_sec = audio.shape[0] / self._sample_rate
        rtf = elapsed / max(duration_sec, 1e-8)

        wav_bytes = audio_to_wav_bytes(audio, self._sample_rate)
        return wav_bytes, rtf

    def synthesize_mel(self, text: str, n_steps: int = 50) -> np.ndarray:
        from tts_pipeline.text.cleaners import clean_text

        clean = clean_text(text)
        phoneme_ids = self._tokenizer.encode(clean.split())
        phoneme_array = np.array([phoneme_ids], dtype=np.int64)
        phoneme_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
        mel = self._acoustic.run(None, {"phoneme_ids": phoneme_array, "phoneme_lengths": phoneme_lengths})[0]
        return mel

    def vocode(self, mel: np.ndarray) -> np.ndarray:
        audio = self._vocoder.run(None, {"mel": mel})[0]
        return audio.squeeze()
