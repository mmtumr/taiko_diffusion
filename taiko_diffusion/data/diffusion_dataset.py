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
    def __init__(self, split_csv: str | Path, stats_path: str | Path, hold_substitution_hint_dir: str | Path | None = None):
        self.split_csv = Path(split_csv)
        self.stats_path = Path(stats_path)
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.condition_mean = np.asarray(self.stats["condition_mean"], dtype=np.float32)
        self.condition_std = np.asarray(self.stats["condition_std"], dtype=np.float32)
        self.hold_substitution_hint_dir = Path(hold_substitution_hint_dir) if hold_substitution_hint_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        data = np.load(local_path(row["npz_path"]), allow_pickle=False)
        chart = data["chart"].astype(np.float32)
        hold_substitution_hint = np.zeros(chart.shape[0], dtype=np.float32)
        if self.hold_substitution_hint_dir is not None:
            hint_path = self.hold_substitution_hint_dir / f"{row['chunk_id']}.npz"
            if hint_path.exists():
                hint = np.load(hint_path, allow_pickle=False)["hold_hint"].astype(np.float32) > 0.5
                hold_substitution_hint = hint.astype(np.float32)
                names = [str(name) for name in self.stats["target_channels"]]
                channel = {name: index for index, name in enumerate(names)}
                for begin in np.flatnonzero(hint & ~np.r_[False, hint[:-1]]):
                    later = np.flatnonzero(~hint[begin:])
                    end = int(begin + later[0] - 1) if later.size else len(hint) - 1
                    if end <= begin:
                        continue
                    chart[begin : end + 1, [channel[name] for name in ["don", "ka", "big_don", "big_ka"]]] = 0.0
                    chart[begin, channel["hold_start"]] = 1.0
                    chart[begin + 1 : end, channel["hold_body"]] = 1.0
                    chart[end, channel["hold_end"]] = 1.0
        condition_raw = data["condition"].astype(np.float32)
        condition = (condition_raw - self.condition_mean) / self.condition_std
        item = {
            "chart": torch.from_numpy(chart.transpose(1, 0)),
            "condition": torch.from_numpy(condition),
            "condition_raw": torch.from_numpy(condition_raw),
            "chunk_id": row["chunk_id"],
            "sample_id": row["sample_id"],
            "title": row.get("title", ""),
            "hold_substitution_hint": torch.from_numpy(hold_substitution_hint),
        }
        if "legal_mask" in data.files:
            item["legal_mask"] = torch.from_numpy(data["legal_mask"].astype(np.float32))
        if "legal_masks" in data.files:
            item["legal_masks"] = torch.from_numpy(data["legal_masks"].astype(np.float32))
        return item


class TaikoAudioDiffusionDataset(TaikoDiffusionDataset):
    def __init__(
        self,
        split_csv: str | Path,
        stats_path: str | Path,
        audio_csv: str | Path,
        audio_stats_path: str | Path,
        hold_substitution_hint_dir: str | Path | None = None,
    ):
        super().__init__(split_csv, stats_path, hold_substitution_hint_dir)
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
