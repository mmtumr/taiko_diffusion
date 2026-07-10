from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def local_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or "\\" not in value:
        return path
    return Path(value.replace("\\", "/"))


class TaikoDiffusionDataset(Dataset):
    def __init__(self, split_csv: str | Path, stats_path: str | Path):
        self.split_csv = Path(split_csv)
        self.stats_path = Path(stats_path)
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.condition_mean = np.asarray(self.stats["condition_mean"], dtype=np.float32)
        self.condition_std = np.asarray(self.stats["condition_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        data = np.load(local_path(row["npz_path"]), allow_pickle=False)
        chart = data["chart"].astype(np.float32)
        condition_raw = data["condition"].astype(np.float32)
        condition = (condition_raw - self.condition_mean) / self.condition_std
        return {
            "chart": torch.from_numpy(chart.transpose(1, 0)),
            "condition": torch.from_numpy(condition),
            "condition_raw": torch.from_numpy(condition_raw),
            "chunk_id": row["chunk_id"],
            "sample_id": row["sample_id"],
            "title": row.get("title", ""),
        }


class TaikoAudioDiffusionDataset(TaikoDiffusionDataset):
    def __init__(
        self,
        split_csv: str | Path,
        stats_path: str | Path,
        audio_csv: str | Path,
        audio_stats_path: str | Path,
    ):
        super().__init__(split_csv, stats_path)
        with Path(audio_csv).open("r", encoding="utf-8-sig", newline="") as file:
            audio_rows = list(csv.DictReader(file))
        self.audio_by_chunk = {row["chunk_id"]: row for row in audio_rows}
        missing = [row["chunk_id"] for row in self.rows if row["chunk_id"] not in self.audio_by_chunk]
        if missing:
            raise ValueError(f"Missing audio features for {len(missing)} chart chunks. First: {missing[0]}")
        audio_stats = json.loads(Path(audio_stats_path).read_text(encoding="utf-8"))
        self.audio_mean = np.asarray(audio_stats["feature_mean"], dtype=np.float32)
        self.audio_std = np.asarray(audio_stats["feature_std"], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = super().__getitem__(index)
        chunk_id = str(item["chunk_id"])
        audio_row = self.audio_by_chunk[chunk_id]
        audio_path = local_path(audio_row["audio_npz_path"])
        data = np.load(audio_path, allow_pickle=False)
        audio = data["audio"].astype(np.float32)
        item["raw_onset"] = torch.from_numpy(audio[:, -2].copy())
        audio = (audio - self.audio_mean) / self.audio_std
        item["audio"] = torch.from_numpy(audio.transpose(1, 0))
        item["audio_npz_path"] = str(audio_path)
        return item
