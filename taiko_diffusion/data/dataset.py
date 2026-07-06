from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    raise ModuleNotFoundError(
        "PyTorch is required for taiko_diffusion.data.dataset. "
        "Install torch in the Miniforge environment before training."
    ) from exc


class TaikoTensorDataset(Dataset):
    def __init__(self, split_csv: str | Path, stats_path: str | Path):
        self.split_csv = Path(split_csv)
        self.stats_path = Path(stats_path)
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.label_names = [str(x) for x in self.stats["label_names"]]
        self.transforms = dict(self.stats["transforms"])
        self.mean = np.asarray(self.stats["mean"], dtype=np.float32)
        self.std = np.asarray(self.stats["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _transform_y(self, y: np.ndarray, stored_labels: list[str]) -> np.ndarray:
        values = {name: float(y[index]) for index, name in enumerate(stored_labels)}
        ordered = np.asarray([values[name] for name in self.label_names], dtype=np.float32)
        return self._normalize_ordered_y(ordered)

    def _raw_y(self, y: np.ndarray, stored_labels: list[str]) -> np.ndarray:
        values = {name: float(y[index]) for index, name in enumerate(stored_labels)}
        return np.asarray([values[name] for name in self.label_names], dtype=np.float32)

    def _normalize_ordered_y(self, ordered: np.ndarray) -> np.ndarray:
        ordered = ordered.copy()
        for index, name in enumerate(self.label_names):
            if self.transforms.get(name) == "log1p":
                ordered[index] = np.log1p(max(ordered[index], 0.0))
            elif self.transforms.get(name) == "logit100":
                clipped = min(max(float(ordered[index]), 0.0), 100.0)
                probability = (clipped + 0.5) / 101.0
                ordered[index] = np.log(probability / (1.0 - probability))
        return (ordered - self.mean) / self.std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        data = np.load(row["npz_path"], allow_pickle=False)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        stored_labels = [str(label) for label in data["label_names"]]
        y_raw = self._raw_y(y, stored_labels)
        y = self._normalize_ordered_y(y_raw)
        duration_frames = int(data["duration_frames"][0])
        return {
            "x": torch.from_numpy(x.transpose(1, 0)),
            "y": torch.from_numpy(y),
            "y_raw": torch.from_numpy(y_raw),
            "duration_frames": torch.tensor(duration_frames, dtype=torch.long),
            "sample_id": row["sample_id"],
            "title": row["title"],
        }
