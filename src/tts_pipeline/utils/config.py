from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, field_validator


class DataConfig(BaseModel):
    language: str = "vi"
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    trim_silence: bool = True
    peak_normalize_db: float = -3.0
    min_duration: float = 0.5
    max_duration: float = 15.0
    min_snr_db: float = 20.0
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    dataset_grade_threshold: str = "C"

    @field_validator("dataset_grade_threshold")
    @classmethod
    def valid_grade(cls, v: str) -> str:
        if v not in ("A", "B", "C", "D", "F"):
            raise ValueError(f"Invalid grade threshold: {v}")
        return v


class TextConfig(BaseModel):
    phonemizer: str = "espeak"
    language: str = "vi"
    add_blank: bool = True
    use_ipa: bool = True


class ModelConfig(BaseModel):
    encoder_dim: int = 192
    encoder_heads: int = 2
    encoder_layers: int = 6
    encoder_kernel_size: int = 3
    decoder_channels: int = 256
    decoder_layers: int = 4
    n_ode_steps: int = 10
    ode_solver: str = "dopri5"
    sigma_min: float = 1e-4
    speaker_embedding_dim: int = 0
    use_sdp: bool = True


class TrainingConfig(BaseModel):
    max_epochs: int = 1000
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    grad_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    save_top_k: int = 3
    early_stopping_patience: int = 50
    resume_from_checkpoint: str | None = None
    precision: str = "bf16-mixed"
    use_compile: bool = False
    num_workers: int = 4
    audio_log_n_samples: int = 5


class EvaluationConfig(BaseModel):
    n_ode_steps_eval: int = 50
    batch_size: int = 8
    whisper_model: str = "small"
    absolute_floor: dict[str, float] = {
        "utmos_mean": 3.0,
        "cer": 0.20,
        "stoi_mean": 0.60,
        "rtf_p95": 0.20,
    }
    relative_delta: dict[str, float] = {
        "utmos_mean": -0.05,
        "cer": 0.02,
        "stoi_mean": -0.03,
        "rtf_p95": 0.05,
    }
    shadow_n_requests: int = 1000


class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    text: TextConfig = TextConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def from_hydra(cfg: DictConfig) -> PipelineConfig:
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        text=TextConfig(**raw.get("text", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )


def load_params(params_path: Path = Path("params.yaml")) -> PipelineConfig:
    cfg = OmegaConf.load(params_path)
    return from_hydra(cfg)  # type: ignore[arg-type]
