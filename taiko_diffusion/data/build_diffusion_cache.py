from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config
from taiko_diffusion.data.build_v5_cache import load_label_map
from taiko_diffusion.data.build_v8_cache import direct_features


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_by_sample(index_path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_rows(index_path)}


def target_grid(source_x: np.ndarray, channels: list[str], target_channels: list[str]) -> np.ndarray:
    channel_index = {name: index for index, name in enumerate(channels)}
    values: dict[str, np.ndarray] = {}
    don = np.clip(source_x[:, channel_index["don"]] + source_x[:, channel_index["big_don"]], 0.0, 1.0)
    ka = np.clip(source_x[:, channel_index["ka"]] + source_x[:, channel_index["big_ka"]], 0.0, 1.0)
    note = np.clip(don + ka, 0.0, 1.0)
    values["don"] = don
    values["ka"] = ka
    values["note_event"] = note
    values["ka_probability"] = ka
    for name in [
        "roll_start",
        "roll_body",
        "roll_end",
        "balloon_start",
        "balloon_body",
        "balloon_end",
    ]:
        values[name] = source_x[:, channel_index[name]]
    return np.stack([values[name] for name in target_channels], axis=1).astype(np.float32)


def condition_values(
    source_x: np.ndarray,
    channels: list[str],
    labels: dict[str, float],
    condition_names: list[str],
    frame_ms: float,
) -> np.ndarray:
    channel_index = {name: index for index, name in enumerate(channels)}
    don = np.clip(source_x[:, channel_index["don"]] + source_x[:, channel_index["big_don"]], 0.0, 1.0)
    ka = np.clip(source_x[:, channel_index["ka"]] + source_x[:, channel_index["big_ka"]], 0.0, 1.0)
    note_count = max(float(np.clip(don + ka, 0.0, 1.0).sum()), 1.0)
    values = dict(labels)
    values.update(direct_features(source_x, channels, labels, frame_ms))
    values["bpm_rhythm_bin"] = float(np.searchsorted([1e-6, 25.0], labels["bpm_change"], side="right"))
    values["note_type_high"] = float(labels["note_type"] >= 25.0)
    values["ka_ratio"] = float(ka.sum() / note_count)
    return np.asarray([float(values[name]) for name in condition_names], dtype=np.float32)


def chunk_starts(duration_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    if duration_frames <= window_frames:
        return [0]
    starts = list(range(0, max(duration_frames - window_frames + 1, 1), stride_frames))
    last = max(duration_frames - window_frames, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_split(
    split_name: str,
    split_rows: list[dict[str, str]],
    source_rows: dict[str, dict[str, str]],
    output_dir: Path,
    target_channels: list[str],
    condition_names: list[str],
    frame_ms: float,
    window_frames: int,
    stride_frames: int,
    min_events: int,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rows: list[dict[str, object]] = []
    conditions: list[np.ndarray] = []
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_row in split_rows:
        sample_id = split_row["sample_id"]
        source_row = source_rows[sample_id]
        source_path = Path(source_row["npz_path"])
        source = np.load(source_path, allow_pickle=False)
        source_x = source["x"].astype(np.float32)
        channels = [str(name) for name in source["channels"]]
        labels = load_label_map(source_path)
        duration_frames = int(source["duration_frames"][0])
        target = target_grid(source_x, channels, target_channels)
        cond = condition_values(source_x, channels, labels, condition_names, frame_ms)

        for start in chunk_starts(duration_frames, window_frames, stride_frames):
            end = start + window_frames
            chunk = np.zeros((window_frames, len(target_channels)), dtype=np.float32)
            available = max(0, min(end, target.shape[0]) - start)
            if available > 0:
                chunk[:available] = target[start : start + available]
            event_indices = [
                index
                for index, name in enumerate(target_channels)
                if name in {"note_event", "don", "ka", "roll_start", "roll_end", "balloon_start", "balloon_end"}
            ]
            event_count = int(chunk[:, event_indices].sum()) if event_indices else int(chunk.sum())
            if event_count < min_events:
                continue
            chunk_id = f"{sample_id}_{start:05d}"
            npz_path = split_dir / f"{chunk_id}.npz"
            np.savez_compressed(
                npz_path,
                chart=chunk,
                condition=cond,
                target_channels=np.asarray(target_channels),
                condition_names=np.asarray(condition_names),
                sample_id=np.asarray([sample_id]),
                title=np.asarray([split_row.get("title", "")]),
                start_frame=np.asarray([start], dtype=np.int32),
                duration_frames=np.asarray([duration_frames], dtype=np.int32),
            )
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "sample_id": sample_id,
                    "npz_path": str(npz_path),
                    "title": split_row.get("title", ""),
                    "start_frame": start,
                    "duration_frames": duration_frames,
                    "event_count": event_count,
                }
            )
            conditions.append(cond)
    return rows, conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-window cache for chart-only diffusion.")
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion_v0.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    source_index = Path(data_cfg["source_index"])
    source_split_dir = Path(data_cfg["source_split_dir"])
    output_dir = Path(data_cfg["cache_dir"])
    frame_ms = float(data_cfg.get("frame_ms", 46.4399))
    window_frames = int(data_cfg["window_frames"])
    stride_frames = int(data_cfg["stride_frames"])
    min_events = int(data_cfg.get("min_events_per_window", 1))
    target_channels = [str(name) for name in data_cfg["target_channels"]]
    condition_names = [str(name) for name in data_cfg["condition_names"]]

    source_rows = source_by_sample(source_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_train_conditions: list[np.ndarray] = []
    split_summaries: dict[str, int] = {}

    for split_name in ["train", "val", "test"]:
        split_rows = read_rows(source_split_dir / f"{split_name}.csv")
        chunk_rows, conditions = make_split(
            split_name,
            split_rows,
            source_rows,
            output_dir,
            target_channels,
            condition_names,
            frame_ms,
            window_frames,
            stride_frames,
            min_events,
        )
        write_rows(output_dir / f"{split_name}.csv", chunk_rows)
        split_summaries[split_name] = len(chunk_rows)
        if split_name == "train":
            all_train_conditions.extend(conditions)

    train_cond = np.stack(all_train_conditions, axis=0)
    cond_mean = train_cond.mean(axis=0)
    cond_std = train_cond.std(axis=0)
    cond_std = np.where(cond_std < 1e-6, 1.0, cond_std)
    stats = {
        "target_channels": target_channels,
        "condition_names": condition_names,
        "condition_mean": cond_mean.astype(float).tolist(),
        "condition_std": cond_std.astype(float).tolist(),
        "window_frames": window_frames,
        "frame_ms": frame_ms,
        "splits": split_summaries,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **split_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
