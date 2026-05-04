from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tts_pipeline.text.tokenizer import Tokenizer
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class TTSDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        file_list: Path,
        mel_dir: Path,
        f0_dir: Path,
        metadata_path: Path,
        tokenizer: Tokenizer,
        stats: dict[str, float] | None = None,
    ) -> None:
        ids = file_list.read_text().strip().splitlines()
        meta = pd.read_csv(metadata_path).set_index("id")
        self._items = [uid for uid in ids if uid in meta.index]
        self._meta = meta
        self._mel_dir = mel_dir
        self._f0_dir = f0_dir
        self._tokenizer = tokenizer
        self._stats = stats or {}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        uid = self._items[idx]
        row = self._meta.loc[uid]

        # Mel spectrogram
        mel = np.load(self._mel_dir / f"{uid}.npy")  # (n_mels, T)
        if self._stats:
            mel = (mel - self._stats["mel_mean"]) / (self._stats["mel_std"] + 1e-8)
        mel_tensor = torch.from_numpy(mel).float()

        # F0
        f0_path = self._f0_dir / f"{uid}.npy"
        f0 = np.load(f0_path).astype(np.float32) if f0_path.exists() else np.zeros(mel.shape[1])
        f0_tensor = torch.from_numpy(f0).float()

        # Phoneme IDs
        phoneme_str = str(row.get("phonemes", ""))
        phonemes = phoneme_str.split() if phoneme_str else []
        phoneme_ids = self._tokenizer.encode(phonemes)
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)

        return {
            "id": uid,
            "mel": mel_tensor,
            "f0": f0_tensor,
            "phoneme_ids": phoneme_tensor,
            "mel_len": torch.tensor(mel_tensor.shape[1], dtype=torch.long),
            "phoneme_len": torch.tensor(len(phoneme_ids), dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    def pad_sequence(seqs: list[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
        max_len = max(s.shape[-1] for s in seqs)
        padded = torch.full((len(seqs), *seqs[0].shape[:-1], max_len), pad_val)
        for i, s in enumerate(seqs):
            padded[i, ..., : s.shape[-1]] = s
        return padded

    return {
        "mel": pad_sequence([b["mel"] for b in batch]),
        "f0": pad_sequence([b["f0"] for b in batch]),
        "phoneme_ids": pad_sequence([b["phoneme_ids"].float() for b in batch]).long(),
        "mel_len": torch.stack([b["mel_len"] for b in batch]),
        "phoneme_len": torch.stack([b["phoneme_len"] for b in batch]),
    }


try:
    import lightning as L

    class TTSDataModule(L.LightningDataModule):
        def __init__(
            self,
            data_root: Path,
            tokenizer: Tokenizer,
            batch_size: int = 16,
            num_workers: int = 4,
        ) -> None:
            super().__init__()
            self._root = Path(data_root)
            self._tokenizer = tokenizer
            self._batch_size = batch_size
            self._num_workers = num_workers
            self._stats: dict[str, float] = {}

        def setup(self, stage: str | None = None) -> None:
            stats_path = self._root / "processed/stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    self._stats = json.load(f)

        def _make_dataset(self, split: str) -> TTSDataset:
            return TTSDataset(
                file_list=self._root / f"splits/{split}.txt",
                mel_dir=self._root / "processed/mels",
                f0_dir=self._root / "processed/f0",
                metadata_path=self._root / "metadata.csv",
                tokenizer=self._tokenizer,
                stats=self._stats,
            )

        def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
            return DataLoader(
                self._make_dataset("train"),
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
            )

        def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
            return DataLoader(
                self._make_dataset("val"),
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
            return DataLoader(
                self._make_dataset("test"),
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=collate_fn,
            )

except ImportError:
    pass  # lightning optional at data-pipeline time
