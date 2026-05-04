"""Training pipeline CLI entry point — called by dvc.yaml train stage."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from tts_pipeline.data.dataset import TTSDataModule
    from tts_pipeline.text.tokenizer import Tokenizer
    from tts_pipeline.training.trainer import run_training
    from tts_pipeline.utils.config import from_hydra

    pipeline_cfg = from_hydra(cfg)

    from pathlib import Path
    vocab_path = Path("artifacts/export/phoneme_vocab.json")
    if vocab_path.exists():
        tokenizer = Tokenizer.load(vocab_path)
    else:
        from tts_pipeline.text.phonemizers.espeak_phonemizer import EspeakPhonemizer
        from tts_pipeline.text.phonemizers.vi_phonemizer import ViPhonemizer
        import pandas as pd

        phonemizer_map = {"espeak": EspeakPhonemizer, "vi": ViPhonemizer}
        PhonemClass = phonemizer_map.get(pipeline_cfg.text.phonemizer, EspeakPhonemizer)
        ph = PhonemClass(language=pipeline_cfg.text.language)

        df = pd.read_csv("data/processed/phonemes.csv")
        all_phonemes = [p for row in df["phonemes"].fillna("") for p in row.split()]
        inventory = sorted(set(all_phonemes))
        tokenizer = Tokenizer.from_inventory(inventory, add_blank=pipeline_cfg.text.add_blank)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(vocab_path)

    data_module = TTSDataModule(
        data_root=Path("data"),
        tokenizer=tokenizer,
        batch_size=pipeline_cfg.training.batch_size,
        num_workers=pipeline_cfg.training.num_workers,
    )
    data_module.tokenizer = tokenizer  # type: ignore[attr-defined]

    run_id = run_training(cfg, data_module, model_name=f"matcha-tts-{pipeline_cfg.data.language}")
    print(f"Training complete. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
