from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config


BASELINE_LABELS = ["const", "combo", "avg_density", "peak_density"]
SOURCE_LABELS = ["note_type", "rhythm", *BASELINE_LABELS]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_label_map(npz_path: str | Path) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=False)
    names = [str(name) for name in data["label_names"]]
    values = data["y"].astype(np.float64)
    return {name: float(values[index]) for index, name in enumerate(names)}


def baseline_matrix(label_maps: list[dict[str, float]]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in BASELINE_LABELS:
        values = np.asarray([labels[name] for labels in label_maps], dtype=np.float64)
        if name == "combo":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    x = np.stack(columns, axis=1)
    return x


def fit_residual_model(train_rows: list[dict[str, str]], target: str) -> dict:
    label_maps = [load_label_map(row["npz_path"]) for row in train_rows]
    y = np.asarray([labels[target] for labels in label_maps], dtype=np.float64)
    x = baseline_matrix(label_maps)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    x_norm = (x - mean) / std
    design = np.concatenate([np.ones((len(x_norm), 1)), x_norm], axis=1)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    residual = y - pred
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
    return {
        "target": target,
        "features": BASELINE_LABELS,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "beta": beta.astype(float).tolist(),
        "train_r2": r2,
    }


def predict_baseline(labels: dict[str, float], model: dict) -> float:
    values = []
    for name in model["features"]:
        value = float(labels[name])
        if name == "combo":
            value = float(np.log1p(max(value, 0.0)))
        values.append(value)
    x = np.asarray(values, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    beta = np.asarray(model["beta"], dtype=np.float64)
    x_norm = (x - mean) / std
    return float(np.concatenate([[1.0], x_norm]) @ beta)


def note_features(x: np.ndarray, channels: list[str]) -> dict[str, float]:
    channel_index = {name: index for index, name in enumerate(channels)}
    note_channels = ["don", "ka", "big_don", "big_ka"]
    notes = x[:, [channel_index[name] for name in note_channels]]
    total = max(float(notes.sum()), 1.0)
    big = float(x[:, channel_index["big_don"]].sum() + x[:, channel_index["big_ka"]].sum())
    ka = float(x[:, channel_index["ka"]].sum() + x[:, channel_index["big_ka"]].sum())

    note_frames = np.where(notes.sum(axis=1) > 0)[0]
    kinds: list[int] = []
    for frame in note_frames:
        is_ka = (
            x[frame, channel_index["ka"]] + x[frame, channel_index["big_ka"]]
            > x[frame, channel_index["don"]] + x[frame, channel_index["big_don"]]
        )
        kinds.append(1 if is_ka else 0)
    alternations = sum(1 for left, right in zip(kinds, kinds[1:]) if left != right)
    return {
        "big_note_ratio": big / total,
        "ka_ratio": ka / total,
        "alternation_rate": alternations / max(len(kinds) - 1, 1),
    }


def event_tracks(x: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}
    note_channel_names = [name for name in ["don", "ka", "big_don", "big_ka"] if name in channel_index]
    note_sum = x[:, [channel_index[name] for name in note_channel_names]].sum(axis=1)
    local_density = np.convolve(note_sum, np.ones(21, dtype=np.float32), mode="same") / 21.0

    extra_names: list[str] = []
    extra_values: list[np.ndarray] = []
    for prefix, event_name, delta_name in [
        ("bpm", "bpm_change_event", "bpm_delta"),
        ("scroll", "scroll_change_event", "scroll_delta"),
    ]:
        event = x[:, channel_index[event_name]].astype(np.float32)
        delta = x[:, channel_index[delta_name]].astype(np.float32)
        extra_names.extend(
            [
                f"{prefix}_abs_delta",
                f"{prefix}_event_local_density",
                f"{prefix}_event_cumsum",
            ]
        )
        extra_values.extend(
            [
                np.abs(delta),
                (event * local_density).astype(np.float32),
                (np.cumsum(event) / 10.0).astype(np.float32),
            ]
        )
    extra = np.stack(extra_values, axis=1).astype(np.float32)
    return np.concatenate([x, extra], axis=1), [*channels, *extra_names]


def pattern_tracks(x: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channel_index = {name: index for index, name in enumerate(channels)}
    note_channel_names = [name for name in ["don", "ka", "big_don", "big_ka"] if name in channel_index]
    note_sum = x[:, [channel_index[name] for name in note_channel_names]].sum(axis=1).astype(np.float32)
    note_frames = np.where(note_sum > 0)[0]

    kind_by_frame = np.full(x.shape[0], -1, dtype=np.int32)
    for frame in note_frames:
        don_value = x[frame, channel_index["don"]]
        ka_value = x[frame, channel_index["ka"]]
        if "big_don" in channel_index:
            don_value += x[frame, channel_index["big_don"]]
        if "big_ka" in channel_index:
            ka_value += x[frame, channel_index["big_ka"]]
        is_ka = ka_value > don_value
        kind_by_frame[frame] = 1 if is_ka else 0

    color_change = np.zeros(x.shape[0], dtype=np.float32)
    prev_kind: int | None = None
    for frame in note_frames:
        kind = int(kind_by_frame[frame])
        if prev_kind is not None and kind != prev_kind:
            color_change[frame] = 1.0
        prev_kind = kind

    prev_interval = np.zeros(x.shape[0], dtype=np.float32)
    next_interval = np.zeros(x.shape[0], dtype=np.float32)
    if len(note_frames) > 1:
        diffs = np.diff(note_frames).astype(np.float32)
        prev_interval[note_frames[1:]] = np.clip(diffs, 0.0, 64.0) / 64.0
        next_interval[note_frames[:-1]] = np.clip(diffs, 0.0, 64.0) / 64.0

    bar_phase = np.zeros(x.shape[0], dtype=np.float32)
    if "barline" in channel_index:
        barlines = np.where(x[:, channel_index["barline"]] > 0)[0]
        if len(barlines) >= 2:
            for left, right in zip(barlines, barlines[1:]):
                if right > left:
                    bar_phase[left:right] = np.linspace(0.0, 1.0, right - left, endpoint=False)
            bar_phase[barlines[-1] :] = 0.0

    def smooth(values: np.ndarray, window: int) -> np.ndarray:
        return (np.convolve(values, np.ones(window, dtype=np.float32), mode="same") / window).astype(np.float32)

    extra_names = [
        "note_density_short",
        "note_density_long",
        "color_change_event",
        "color_change_density",
        "prev_note_interval",
        "next_note_interval",
        "bar_phase",
    ]
    extra = np.stack(
        [
            smooth(note_sum, 11),
            smooth(note_sum, 43),
            color_change,
            smooth(color_change, 21),
            prev_interval,
            next_interval,
            bar_phase,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([x, extra], axis=1), [*channels, *extra_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v5 cache from an existing encoder_v1 tensor cache.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder_v5.yaml"))
    parser.add_argument("--source-index", type=Path, default=Path("data/cache/encoder_v1/index.csv"))
    parser.add_argument("--source-train", type=Path, default=Path("data/splits/encoder_v1/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/encoder_v5"))
    args = parser.parse_args()

    config = load_config(args.config)
    target_columns = [str(name) for name in config["data"]["target_columns"]]
    add_event_tracks = bool(config.get("derived_grid", {}).get("add_event_tracks", True))
    add_pattern_tracks = bool(config.get("derived_grid", {}).get("add_pattern_tracks", False))

    source_rows = read_rows(args.source_index)
    train_rows = read_rows(args.source_train)
    note_model = fit_residual_model(train_rows, "note_type")
    rhythm_model = fit_residual_model(train_rows, "rhythm")

    train_common: list[float] = []
    for row in train_rows:
        labels = load_label_map(row["npz_path"])
        note_residual = labels["note_type"] - predict_baseline(labels, note_model)
        rhythm_residual = labels["rhythm"] - predict_baseline(labels, rhythm_model)
        train_common.append(note_residual)
        train_common.append(rhythm_residual)
    scale = float(np.std(train_common)) or 1.0

    train_axis: list[float] = []
    for row in train_rows:
        labels = load_label_map(row["npz_path"])
        note_residual = labels["note_type"] - predict_baseline(labels, note_model)
        rhythm_residual = labels["rhythm"] - predict_baseline(labels, rhythm_model)
        train_axis.append((note_residual / scale + rhythm_residual / scale) / 2.0)
    thresholds = np.percentile(np.asarray(train_axis, dtype=np.float64), [33.333, 66.667])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []

    for row in source_rows:
        source = np.load(row["npz_path"], allow_pickle=False)
        x = source["x"].astype(np.float32)
        channels = [str(name) for name in source["channels"]]
        if add_event_tracks:
            x, channels = event_tracks(x, channels)
        if add_pattern_tracks:
            x, channels = pattern_tracks(x, channels)

        labels = load_label_map(row["npz_path"])
        labels["speed_change"] = max(labels.get("bpm_change", 0.0), labels.get("hs_change", 0.0))
        labels.update(note_features(source["x"].astype(np.float32), [str(name) for name in source["channels"]]))
        note_residual = labels["note_type"] - predict_baseline(labels, note_model)
        rhythm_residual = labels["rhythm"] - predict_baseline(labels, rhythm_model)
        subjective_axis = (note_residual / scale + rhythm_residual / scale) / 2.0
        labels["subjective_complexity_axis"] = subjective_axis
        labels["subjective_complexity_bin"] = float(np.searchsorted(thresholds, subjective_axis, side="right"))

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
        "rhythm_residual_model": rhythm_model,
        "subjective_axis_scale": scale,
        "subjective_bin_thresholds": thresholds.astype(float).tolist(),
        "add_event_tracks": add_event_tracks,
        "add_pattern_tracks": add_pattern_tracks,
        "channels": channels,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
