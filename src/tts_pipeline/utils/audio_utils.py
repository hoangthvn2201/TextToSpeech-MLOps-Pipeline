from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: Path, target_sr: int = 22050) -> tuple[torch.Tensor, int]:
    import torchaudio

    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def save_audio(path: Path, waveform: torch.Tensor | np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.squeeze().cpu().numpy()
    else:
        waveform_np = waveform.squeeze()
    sf.write(str(path), waveform_np, sr)


def audio_to_wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def compute_snr_db(waveform: np.ndarray) -> float:
    signal_power = np.mean(waveform**2)
    if signal_power < 1e-10:
        return -np.inf
    # Estimate noise as the bottom 10th percentile of frame energies
    frame_size = 512
    n_frames = len(waveform) // frame_size
    if n_frames < 2:
        return float("nan")
    frames = waveform[: n_frames * frame_size].reshape(n_frames, frame_size)
    frame_energies = np.mean(frames**2, axis=1)
    noise_power = float(np.percentile(frame_energies, 10))
    if noise_power < 1e-10:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def wav_header_bytes(sr: int, n_channels: int = 1, bit_depth: int = 16) -> bytes:
    # Minimal WAV header for streaming (data size = 0, to be filled by client)
    byte_rate = sr * n_channels * bit_depth // 8
    block_align = n_channels * bit_depth // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0,  # file size placeholder
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        n_channels,
        sr,
        byte_rate,
        block_align,
        bit_depth,
        b"data",
        0,  # data size placeholder
    )
    return header
