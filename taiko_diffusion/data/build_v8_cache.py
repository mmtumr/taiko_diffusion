from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config
from taiko_diffusion.data.build_v5_cache import (
    event_tracks,
    fit_residual_model,
    load_label_map,
    pattern_tracks,
    predict_baseline,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def collapse_big_notes(x: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}
    merged: list[np.ndarray] = []
    names: list[str] = []
    for name in channels:
        if name == "don":
            merged.append(
                np.clip(
                    x[:, channel_index["don"]] + x[:, channel_index.get("big_don", channel_index["don"])],
                    0.0,
                    1.0,
                )
            )
            names.append("don")
        elif name == "ka":
            merged.append(
                np.clip(
                    x[:, channel_index["ka"]] + x[:, channel_index.get("big_ka", channel_index["ka"])],
                    0.0,
                    1.0,
                )
            )
            names.append("ka")
        elif name in {"big_don", "big_ka"}:
            continue
        else:
            merged.append(x[:, channel_index[name]])
            names.append(name)
    return np.stack(merged, axis=1).astype(np.float32), names


def direct_features(x: np.ndarray, channels: list[str], labels: dict[str, float], frame_ms: float) -> dict[str, float]:
    channel_index = {name: index for index, name in enumerate(channels)}
    note = (
        x[:, channel_index["don"]]
        + x[:, channel_index["ka"]]
        + x[:, channel_index["big_don"]]
        + x[:, channel_index["big_ka"]]
    )
    note_frames = np.where(note > 0)[0]
    total_notes = max(float(len(note_frames)), 1.0)
    active = x[:, channel_index["active"]] > 0 if "active" in channel_index else np.ones(x.shape[0], dtype=bool)
    active_seconds = max(float(active.sum()) * frame_ms / 1000.0, frame_ms / 1000.0)
    window = max(int(round(1000.0 / frame_ms)), 1)
    peak_notes = float(np.convolve((note > 0).astype(np.float32), np.ones(window, dtype=np.float32), mode="same").max())
    big_notes = float(x[:, channel_index["big_don"]].sum() + x[:, channel_index["big_ka"]].sum())

    roll_time = max(float(labels.get("roll_time", 0.0)), 0.0)
    balloon_num = max(float(labels.get("balloon_num", 0.0)), 0.0)
    roll_part = float(np.log1p(roll_time))
    balloon_part = float(np.log1p(balloon_num))
    ratio_denominator = roll_part + balloon_part
    balloon_roll_ratio = balloon_part / ratio_denominator if ratio_denominator > 1e-9 else 0.0

    return {
        "avg_density": total_notes / active_seconds,
        "peak_density": peak_notes,
        "big_note_ratio": big_notes / total_notes,
        "balloon_roll_ratio": balloon_roll_ratio,
    }


def alternating_hand_tracks(x: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}
    if "don" not in channel_index or "ka" not in channel_index:
        return x, channels

    note_values = x[:, channel_index["don"]] + x[:, channel_index["ka"]]
    note_frames = np.where(note_values > 0)[0]
    kinds: list[int] = []
    for frame in note_frames:
        is_ka = x[frame, channel_index["ka"]] > x[frame, channel_index["don"]]
        kinds.append(1 if is_ka else 0)

    left_change = np.zeros(x.shape[0], dtype=np.float32)
    right_change = np.zeros(x.shape[0], dtype=np.float32)
    left_note = np.zeros(x.shape[0], dtype=np.float32)
    right_note = np.zeros(x.shape[0], dtype=np.float32)
    for note_index, frame in enumerate(note_frames):
        if note_index % 2 == 0:
            left_note[frame] = 1.0
        else:
            right_note[frame] = 1.0
        if note_index >= 2 and kinds[note_index] != kinds[note_index - 2]:
            if note_index % 2 == 0:
                left_change[frame] = 1.0
            else:
                right_change[frame] = 1.0

    def smooth(values: np.ndarray, window: int) -> np.ndarray:
        return (np.convolve(values, np.ones(window, dtype=np.float32), mode="same") / window).astype(np.float32)

    left_density = smooth(left_change, 21)
    right_density = smooth(right_change, 21)
    total_density = left_density + right_density
    imbalance = np.abs(left_density - right_density) / (total_density + 1e-6)

    extra_names = [
        "left_hand_note",
        "right_hand_note",
        "left_hand_change_event",
        "right_hand_change_event",
        "left_hand_change_density",
        "right_hand_change_density",
        "hand_change_total_density",
        "hand_change_imbalance",
    ]
    extra = np.stack(
        [
            left_note,
            right_note,
            left_change,
            right_change,
            left_density,
            right_density,
            total_density,
            imbalance.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([x, extra], axis=1), [*channels, *extra_names]


def half_alternating_hand_tracks(
    x: np.ndarray,
    channels: list[str],
    frame_ms: float,
) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}
    if "don" not in channel_index or "ka" not in channel_index:
        return x, channels

    note_values = x[:, channel_index["don"]] + x[:, channel_index["ka"]]
    note_frames = np.where(note_values > 0)[0]
    kinds: list[int] = []
    for frame in note_frames:
        is_ka = x[frame, channel_index["ka"]] > x[frame, channel_index["don"]]
        kinds.append(1 if is_ka else 0)

    if "bpm" in channel_index:
        bpm_values = x[:, channel_index["bpm"]].astype(np.float32) * 300.0
        positive_bpm = bpm_values[bpm_values > 1e-6]
        fallback_bpm = float(np.median(positive_bpm)) if len(positive_bpm) else 120.0
    else:
        bpm_values = np.zeros(x.shape[0], dtype=np.float32)
        fallback_bpm = 120.0

    left_change = np.zeros(x.shape[0], dtype=np.float32)
    right_change = np.zeros(x.shape[0], dtype=np.float32)
    left_note = np.zeros(x.shape[0], dtype=np.float32)
    right_note = np.zeros(x.shape[0], dtype=np.float32)

    current_hand = 0
    last_kind_by_hand: dict[int, int] = {}
    for note_index, frame in enumerate(note_frames):
        kind = kinds[note_index]
        if note_index == 0:
            current_hand = 0
        else:
            previous_frame = int(note_frames[note_index - 1])
            bpm = float(bpm_values[frame]) if float(bpm_values[frame]) > 1e-6 else fallback_bpm
            eighth_frames = (30000.0 / bpm) / frame_ms
            interval_frames = float(frame - previous_frame)
            if interval_frames < eighth_frames * 0.85:
                current_hand = 1 - current_hand
            else:
                current_hand = 0

        if current_hand == 0:
            left_note[frame] = 1.0
            if current_hand in last_kind_by_hand and last_kind_by_hand[current_hand] != kind:
                left_change[frame] = 1.0
        else:
            right_note[frame] = 1.0
            if current_hand in last_kind_by_hand and last_kind_by_hand[current_hand] != kind:
                right_change[frame] = 1.0
        last_kind_by_hand[current_hand] = kind

    def smooth(values: np.ndarray, window: int) -> np.ndarray:
        return (np.convolve(values, np.ones(window, dtype=np.float32), mode="same") / window).astype(np.float32)

    left_density = smooth(left_change, 21)
    right_density = smooth(right_change, 21)
    total_density = left_density + right_density
    imbalance = np.abs(left_density - right_density) / (total_density + 1e-6)

    extra_names = [
        "half_left_hand_note",
        "half_right_hand_note",
        "half_left_hand_change_event",
        "half_right_hand_change_event",
        "half_left_hand_change_density",
        "half_right_hand_change_density",
        "half_hand_change_total_density",
        "half_hand_change_imbalance",
    ]
    extra = np.stack(
        [
            left_note,
            right_note,
            left_change,
            right_change,
            left_density,
            right_density,
            total_density,
            imbalance.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([x, extra], axis=1), [*channels, *extra_names]


def hand_timing_tracks(x: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}

    def smooth(values: np.ndarray, window: int) -> np.ndarray:
        return (np.convolve(values, np.ones(window, dtype=np.float32), mode="same") / window).astype(np.float32)

    def timing_values(total_change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        event_frames = np.where(total_change > 0)[0]
        irregularity = np.zeros(x.shape[0], dtype=np.float32)
        if len(event_frames) >= 4:
            gaps = np.diff(event_frames).astype(np.float32)
            event_values = np.zeros(len(event_frames), dtype=np.float32)
            for event_index in range(len(event_frames)):
                left = max(0, event_index - 3)
                right = min(len(gaps), event_index + 3)
                local_gaps = gaps[left:right]
                if len(local_gaps) >= 2:
                    mean_gap = float(local_gaps.mean())
                    if mean_gap > 1e-6:
                        event_values[event_index] = min(float(local_gaps.std()) / mean_gap, 2.0) / 2.0
            for event_index, frame in enumerate(event_frames):
                if event_index == 0:
                    start = 0
                else:
                    start = int((event_frames[event_index - 1] + frame) // 2)
                if event_index + 1 == len(event_frames):
                    end = x.shape[0]
                else:
                    end = int((frame + event_frames[event_index + 1]) // 2)
                irregularity[start:end] = event_values[event_index]

        short_density = smooth(total_change, 11)
        long_density = smooth(total_change, 65)
        burstiness = np.clip(short_density / (long_density + 1e-6) / 4.0, 0.0, 1.0).astype(np.float32)
        return irregularity, burstiness

    extra_names: list[str] = []
    extra_values: list[np.ndarray] = []
    groups = [
        (
            "hand",
            "left_hand_change_event",
            "right_hand_change_event",
        ),
        (
            "half_hand",
            "half_left_hand_change_event",
            "half_right_hand_change_event",
        ),
    ]
    for prefix, left_name, right_name in groups:
        if left_name not in channel_index or right_name not in channel_index:
            continue
        total_change = np.clip(
            x[:, channel_index[left_name]] + x[:, channel_index[right_name]],
            0.0,
            1.0,
        ).astype(np.float32)
        irregularity, burstiness = timing_values(total_change)
        extra_names.extend([f"{prefix}_change_irregularity", f"{prefix}_change_burstiness"])
        extra_values.extend([irregularity, burstiness])

    if not extra_values:
        return x, channels
    extra = np.stack(extra_values, axis=1).astype(np.float32)
    return np.concatenate([x, extra], axis=1), [*channels, *extra_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8 cache with collapsed note lanes and rhythm-processing bin.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder_v8_rhythm_bin.yaml"))
    parser.add_argument("--source-index", type=Path, default=Path("data/cache/encoder_v1/index.csv"))
    parser.add_argument("--source-train", type=Path, default=Path("data/splits/encoder_v1/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/encoder_v8_rhythm_bin"))
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    target_columns = [str(name) for name in data_config["target_columns"]]
    frame_ms = float(data_config.get("frame_ms", 46.4399))
    add_event_tracks = bool(config.get("derived_grid", {}).get("add_event_tracks", True))
    add_pattern_tracks = bool(config.get("derived_grid", {}).get("add_pattern_tracks", True))
    add_hand_tracks = bool(config.get("derived_grid", {}).get("add_alternating_hand_tracks", False))
    add_half_hand_tracks = bool(config.get("derived_grid", {}).get("add_half_alternating_hand_tracks", False))
    add_hand_timing_tracks = bool(config.get("derived_grid", {}).get("add_hand_timing_tracks", False))
    collapse_big = bool(config.get("derived_grid", {}).get("collapse_big_notes", True))

    source_rows = read_rows(args.source_index)
    train_rows = read_rows(args.source_train)
    note_model = fit_residual_model(train_rows, "note_type")
    rhythm_model = fit_residual_model(train_rows, "rhythm")
    train_note_axis = []
    train_axis = []
    train_raw_rhythm = []
    for row in train_rows:
        labels = load_label_map(row["npz_path"])
        train_note_axis.append(labels["note_type"] - predict_baseline(labels, note_model))
        train_axis.append(labels["rhythm"] - predict_baseline(labels, rhythm_model))
        train_raw_rhythm.append(labels["rhythm"])
    note_thresholds = np.percentile(np.asarray(train_note_axis, dtype=np.float64), [33.333, 66.667])
    thresholds = np.percentile(np.asarray(train_axis, dtype=np.float64), [33.333, 66.667])
    raw_thresholds = np.percentile(np.asarray(train_raw_rhythm, dtype=np.float64), [33.333, 66.667])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []
    final_channels: list[str] = []

    for row in source_rows:
        source = np.load(row["npz_path"], allow_pickle=False)
        source_x = source["x"].astype(np.float32)
        source_channels = [str(name) for name in source["channels"]]
        labels = load_label_map(row["npz_path"])
        note_axis = labels["note_type"] - predict_baseline(labels, note_model)
        labels["note_rhythm_residual_axis"] = note_axis
        labels["note_rhythm_residual_bin"] = float(
            np.searchsorted(note_thresholds, note_axis, side="right")
        )
        rhythm_axis = labels["rhythm"] - predict_baseline(labels, rhythm_model)
        labels["rhythm_processing_axis"] = rhythm_axis
        labels["rhythm_processing_bin"] = float(np.searchsorted(thresholds, rhythm_axis, side="right"))
        labels["rhythm_processing_abs_bin"] = float(
            np.searchsorted(raw_thresholds, labels["rhythm"], side="right")
        )
        labels["rhythm_processing_semantic_bin"] = float(
            np.searchsorted([10.0, 40.0], labels["rhythm"], side="right")
        )
        labels["rhythm_processing_semantic50_bin"] = float(
            np.searchsorted([10.0, 50.0], labels["rhythm"], side="right")
        )
        labels["note_rhythm_bin"] = float(
            np.searchsorted([10.0, 40.0], labels["note_type"], side="right")
        )
        labels["note_rhythm_high25"] = float(labels["note_type"] >= 25.0)
        labels["bpm_rhythm_bin"] = float(
            np.searchsorted([1e-6, 25.0], labels["bpm_change"], side="right")
        )
        labels.update(direct_features(source_x, source_channels, labels, frame_ms))

        x = source_x
        channels = source_channels
        if collapse_big:
            x, channels = collapse_big_notes(x, channels)
        if add_event_tracks:
            x, channels = event_tracks(x, channels)
        if add_pattern_tracks:
            x, channels = pattern_tracks(x, channels)
        if add_hand_tracks:
            x, channels = alternating_hand_tracks(x, channels)
        if add_half_hand_tracks:
            x, channels = half_alternating_hand_tracks(x, channels, frame_ms)
        if add_hand_timing_tracks:
            x, channels = hand_timing_tracks(x, channels)

        y = np.asarray([float(labels[name]) for name in target_columns], dtype=np.float32)
        npz_path = output_dir / f"{row['sample_id']}.npz"
        np.savez_compressed(
            npz_path,
            x=x,
            y=y,
            channels=np.asarray(channels),
            label_names=np.asarray(target_columns),
            duration_frames=source["duration_frames"],
        )
        next_row = dict(row)
        next_row["npz_path"] = str(npz_path)
        index_rows.append(next_row)
        final_channels = channels

    with (output_dir / "index.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "source_index": str(args.source_index),
        "source_train": str(args.source_train),
        "output_dir": str(output_dir),
        "rows": len(index_rows),
        "target_columns": target_columns,
        "note_residual_model": note_model,
        "note_rhythm_residual_bin_thresholds": note_thresholds.astype(float).tolist(),
        "rhythm_residual_model": rhythm_model,
        "rhythm_processing_bin_thresholds": thresholds.astype(float).tolist(),
        "rhythm_processing_abs_bin_thresholds": raw_thresholds.astype(float).tolist(),
        "rhythm_processing_semantic_bin_thresholds": [10.0, 40.0],
        "rhythm_processing_semantic50_bin_thresholds": [10.0, 50.0],
        "note_rhythm_bin_thresholds": [10.0, 40.0],
        "note_rhythm_high25_threshold": 25.0,
        "bpm_rhythm_bin_thresholds": [0.0, 25.0],
        "collapse_big_notes": collapse_big,
        "add_event_tracks": add_event_tracks,
        "add_pattern_tracks": add_pattern_tracks,
        "add_alternating_hand_tracks": add_hand_tracks,
        "add_half_alternating_hand_tracks": add_half_hand_tracks,
        "add_hand_timing_tracks": add_hand_timing_tracks,
        "channels": final_channels,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
