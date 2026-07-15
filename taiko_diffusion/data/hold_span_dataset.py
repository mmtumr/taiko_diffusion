from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset


def extract_hold_spans(chart: np.ndarray, channels: list[str]) -> list[tuple[int, int]]:
    channel = {name: index for index, name in enumerate(channels)}
    starts = chart[:, channel["hold_start"]] > 0.5
    body = chart[:, channel["hold_body"]] > 0.5
    ends = chart[:, channel["hold_end"]] > 0.5
    spans: list[tuple[int, int]] = []
    for start in np.flatnonzero(starts):
        cursor = int(start) + 1
        while cursor < len(body) and body[cursor] and not starts[cursor]:
            cursor += 1
        if cursor < len(ends) and ends[cursor]:
            end = cursor
        else:
            end = cursor - 1
        if end > start:
            spans.append((int(start), int(end)))
    return spans


class HoldSpanDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        stats_path: str,
        audio_csv: str,
        audio_stats_path: str,
        query_count: int,
        context_channels: list[str],
    ):
        self.base = TaikoAudioDiffusionDataset(split_csv, stats_path, audio_csv, audio_stats_path)
        self.query_count = int(query_count)
        self.target_channels = [str(name) for name in self.base.stats["target_channels"]]
        channel = {name: index for index, name in enumerate(self.target_channels)}
        self.context_indices = [channel[name] for name in context_channels]
        self.context_channels = list(context_channels)
        self.window_frames = int(self.base.stats["window_frames"])
        self.duration_log_scale = float(np.log1p(self.window_frames))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.base[index]
        chart = item["chart"]
        chart_time_first = chart.transpose(0, 1).numpy()
        spans = extract_hold_spans(chart_time_first, self.target_channels)[: self.query_count]
        active_mask = item["active_mask"].float()
        active_frames = max(int(active_mask.sum().item()), 2)
        exists = np.zeros(self.query_count, dtype=np.float32)
        starts = np.zeros(self.query_count, dtype=np.float32)
        durations = np.zeros(self.query_count, dtype=np.float32)
        for span_index, (start, end) in enumerate(spans):
            exists[span_index] = 1.0
            starts[span_index] = float(start) / float(self.window_frames - 1)
            durations[span_index] = float(np.log1p(end - start) / self.duration_log_scale)
        return {
            "audio": item["audio"],
            "chart_context": chart[self.context_indices],
            "legal_mask": item["legal_mask"].float(),
            "active_mask": active_mask,
            "condition": item["condition"],
            "span_exists": torch.from_numpy(exists),
            "span_start": torch.from_numpy(starts),
            "span_duration_log": torch.from_numpy(durations),
            "span_count": torch.tensor(len(spans), dtype=torch.long),
            "active_frames": torch.tensor(active_frames, dtype=torch.long),
            "window_frames": torch.tensor(self.window_frames, dtype=torch.long),
            "chunk_id": item["chunk_id"],
            "title": item["title"],
        }


def hold_span_features(batch: dict[str, torch.Tensor | str]) -> torch.Tensor:
    return torch.cat(
        [
            batch["audio"],
            batch["chart_context"],
            batch["legal_mask"].unsqueeze(1),
            batch["active_mask"].unsqueeze(1),
        ],
        dim=1,
    )
