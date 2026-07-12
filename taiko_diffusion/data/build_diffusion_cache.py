from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config
from taiko_diffusion.data.build_v5_cache import load_label_map
from taiko_diffusion.data.build_v8_cache import direct_features
from taiko_diffusion.data.diffusion_dataset import local_path


BINNED_V10_NAMES = {
    "complex_bin",
    "hs_change_bin",
    "bpm_rhythm_bin",
    "note_type_bin",
    "avg_density_bin",
    "peak_density_bin",
}


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
    separate_big_notes = "big_don" in target_channels or "big_ka" in target_channels
    don = source_x[:, channel_index["don"]]
    ka = source_x[:, channel_index["ka"]]
    if not separate_big_notes:
        don = np.clip(don + source_x[:, channel_index["big_don"]], 0.0, 1.0)
        ka = np.clip(ka + source_x[:, channel_index["big_ka"]], 0.0, 1.0)
    note = np.clip(don + ka, 0.0, 1.0)
    values["don"] = don
    values["ka"] = ka
    values["big_don"] = source_x[:, channel_index["big_don"]]
    values["big_ka"] = source_x[:, channel_index["big_ka"]]
    values["note_event"] = note
    values["ka_probability"] = ka
    values["hold_start"] = np.clip(
        source_x[:, channel_index["roll_start"]] + source_x[:, channel_index["balloon_start"]], 0.0, 1.0
    )
    values["hold_body"] = np.clip(
        source_x[:, channel_index["roll_body"]] + source_x[:, channel_index["balloon_body"]], 0.0, 1.0
    )
    values["hold_end"] = np.clip(
        source_x[:, channel_index["roll_end"]] + source_x[:, channel_index["balloon_end"]], 0.0, 1.0
    )
    for name in [
        "roll_start",
        "roll_body",
        "roll_end",
        "balloon_start",
        "balloon_body",
        "balloon_end",
    ]:
        values[name] = source_x[:, channel_index[name]]
    values["bpm_change_event"] = source_x[:, channel_index["bpm_change_event"]]
    values["bpm_value"] = np.clip(source_x[:, channel_index["bpm"]], 0.0, 1.0)
    values["scroll_change_event"] = source_x[:, channel_index["scroll_change_event"]]
    values["scroll_value"] = np.clip(source_x[:, channel_index["scroll"]] / 4.0, 0.0, 1.0)
    return np.stack([values[name] for name in target_channels], axis=1).astype(np.float32)


def exact_grid_metadata(
    source_x: np.ndarray,
    channels: list[str],
    duration_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_index = {name: index for index, name in enumerate(channels)}
    active = source_x[:, channel_index["active"]] > 0.5
    barlines = np.flatnonzero(source_x[:duration_frames, channel_index["barline"]] > 0.5).tolist()
    if not barlines or barlines[0] != 0:
        barlines.insert(0, 0)
    boundaries = sorted(set([*barlines, duration_frames]))
    masks = np.zeros((3, source_x.shape[0]), dtype=np.float32)
    measure_indices = np.full((3, source_x.shape[0]), -1, dtype=np.int32)
    slot_indices = np.full((3, source_x.shape[0]), -1, dtype=np.int32)
    medium_slots = sorted(
        set().union(*(range(0, 96, 96 // division) for division in [8, 12, 16, 24, 32]))
    )
    slots_by_bin = [list(range(0, 96, 6)), medium_slots]
    for measure_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        length = max(end - start, 1)
        for bin_index, slots in enumerate(slots_by_bin):
            for slot in slots:
                frame = min(end - 1, start + int(round(slot * length / 96.0)))
                if frame < 0 or frame >= source_x.shape[0] or not active[frame]:
                    continue
                masks[bin_index, frame] = 1.0
                measure_indices[bin_index, frame] = measure_index
                slot_indices[bin_index, frame] = slot
    masks[2] = active.astype(np.float32)
    return masks, measure_indices, slot_indices


def snap_target_to_grid(
    target: np.ndarray,
    legal_mask: np.ndarray,
    target_channels: list[str],
) -> np.ndarray:
    legal_indices = np.flatnonzero(legal_mask > 0.5)
    if legal_indices.size == 0:
        return np.zeros_like(target)
    snapped = target.copy()
    snap_names = {
        "don", "ka", "big_don", "big_ka", "note_event", "roll_start", "roll_end",
        "balloon_start", "balloon_end", "bpm_change_event", "scroll_change_event",
    }
    for channel_index, name in enumerate(target_channels):
        if name not in snap_names:
            continue
        snapped[:, channel_index] = 0.0
        for source_frame in np.flatnonzero(target[:, channel_index] > 0.5):
            nearest = int(legal_indices[np.argmin(np.abs(legal_indices - source_frame))])
            snapped[nearest, channel_index] = max(snapped[nearest, channel_index], target[source_frame, channel_index])
    return snapped


def subdivision_bin_from_target(target: np.ndarray, legal_masks: np.ndarray) -> int:
    note_frames = np.flatnonzero(target.max(axis=1) > 0.5)
    if note_frames.size == 0:
        return 0
    for bin_index in [0, 1]:
        legal_frames = np.flatnonzero(legal_masks[bin_index] > 0.5)
        if legal_frames.size and all(np.min(np.abs(legal_frames - frame)) <= 1 for frame in note_frames):
            return bin_index
    return 2


def condition_values(
    source_x: np.ndarray,
    channels: list[str],
    labels: dict[str, float],
    condition_names: list[str],
    frame_ms: float,
    bin_thresholds: dict[str, list[float]] | None = None,
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
    roll_count = float(source_x[:, channel_index["roll_start"]].sum())
    balloon_count = float(source_x[:, channel_index["balloon_start"]].sum())
    values["balloon_roll_ratio"] = balloon_count / max(roll_count + balloon_count, 1.0)
    values["subdivision_bin"] = 2.0
    if bin_thresholds is not None:
        values["complex_bin"] = float(np.searchsorted(bin_thresholds["complex_bin"], values["complex"], side="right"))
        values["hs_change_bin"] = float(values["hs_change"] > 0.0)
        values["note_type_bin"] = float(np.searchsorted(bin_thresholds["note_type_bin"], values["note_type"], side="right"))
        values["avg_density_bin"] = float(
            np.searchsorted(bin_thresholds["avg_density_bin"], values["avg_density"], side="right")
        )
        peak_ratio = values["peak_density"] / max(values["avg_density"], 1e-6)
        values["peak_density_bin"] = float(
            np.searchsorted(bin_thresholds["peak_density_bin"], peak_ratio, side="right")
        )
    return np.asarray([float(values[name]) for name in condition_names], dtype=np.float32)


def compute_bin_thresholds(
    train_rows: list[dict[str, str]],
    source_rows: dict[str, dict[str, str]],
    frame_ms: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    collected = {"complex_bin": [], "note_type_bin": [], "avg_density_bin": [], "peak_density_bin": []}
    for row in train_rows:
        source_path = local_path(source_rows[row["sample_id"]]["npz_path"])
        source = np.load(source_path, allow_pickle=False)
        source_x = source["x"].astype(np.float32)
        channels = [str(name) for name in source["channels"]]
        channel_index = {name: index for index, name in enumerate(channels)}
        labels = load_label_map(source_path)
        direct = direct_features(source_x, channels, labels, frame_ms)
        collected["complex_bin"].append(float(labels["complex"]))
        collected["note_type_bin"].append(float(labels["note_type"]))
        collected["avg_density_bin"].append(float(direct["avg_density"]))
        collected["peak_density_bin"].append(float(direct["peak_density"] / max(direct["avg_density"], 1e-6)))
    thresholds = {
        name: np.quantile(np.asarray(values, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0]).astype(float).tolist()
        for name, values in collected.items()
    }
    representatives = {}
    for name, values in collected.items():
        array = np.asarray(values, dtype=np.float32)
        bins = np.searchsorted(thresholds[name], array, side="right")
        representatives[name] = [float(array[bins == index].mean()) for index in range(3)]
    return thresholds, representatives


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
    bin_thresholds: dict[str, list[float]] | None = None,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rows: list[dict[str, object]] = []
    conditions: list[np.ndarray] = []
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_row in split_rows:
        sample_id = split_row["sample_id"]
        source_row = source_rows[sample_id]
        source_path = local_path(source_row["npz_path"])
        source = np.load(source_path, allow_pickle=False)
        source_x = source["x"].astype(np.float32)
        channels = [str(name) for name in source["channels"]]
        channel_index = {name: index for index, name in enumerate(channels)}
        labels = load_label_map(source_path)
        duration_frames = int(source["duration_frames"][0])
        target = target_grid(source_x, channels, target_channels)
        cond = condition_values(source_x, channels, labels, condition_names, frame_ms, bin_thresholds)
        if "subdivision_bin" in condition_names or "complex_bin" in condition_names:
            all_legal_masks, all_measure_indices, all_slot_indices = exact_grid_metadata(
                source_x, channels, duration_frames
            )
        else:
            all_legal_masks = all_measure_indices = all_slot_indices = None
        if "subdivision_bin" in condition_names and all_legal_masks is not None:
            subdivision_bin = subdivision_bin_from_target(target, all_legal_masks)
            cond[condition_names.index("subdivision_bin")] = float(subdivision_bin)
        elif "complex_bin" in condition_names:
            subdivision_bin = int(round(float(cond[condition_names.index("complex_bin")])))
        else:
            subdivision_bin = 2
        if all_legal_masks is not None and subdivision_bin < 2:
            target = snap_target_to_grid(target, all_legal_masks[max(0, subdivision_bin)], target_channels)

        for start in chunk_starts(duration_frames, window_frames, stride_frames):
            end = start + window_frames
            chunk = np.zeros((window_frames, len(target_channels)), dtype=np.float32)
            available = max(0, min(end, target.shape[0]) - start)
            if available > 0:
                chunk[:available] = target[start : start + available]
            legal_masks = np.ones((3, window_frames), dtype=np.float32)
            measure_indices = np.full((3, window_frames), -1, dtype=np.int32)
            slot_indices = np.full((3, window_frames), -1, dtype=np.int32)
            bpm_track = np.zeros(window_frames, dtype=np.float32)
            measure_track = np.zeros(window_frames, dtype=np.float32)
            if all_legal_masks is not None and available > 0:
                legal_masks[:, :available] = all_legal_masks[:, start : start + available]
                legal_masks[:, available:] = 0.0
                measure_indices[:, :available] = all_measure_indices[:, start : start + available]
                slot_indices[:, :available] = all_slot_indices[:, start : start + available]
                bpm_track[:available] = source_x[start : start + available, channel_index["bpm"]] * 300.0
                measure_track[:available] = source_x[start : start + available, channel_index["measure"]]
            event_indices = [
                index
                for index, name in enumerate(target_channels)
                if name in {
                    "note_event", "don", "ka", "big_don", "big_ka", "roll_start", "roll_end",
                    "balloon_start", "balloon_end", "bpm_change_event", "scroll_change_event",
                }
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
                legal_masks=legal_masks,
                legal_mask=legal_masks[max(0, min(subdivision_bin, 2))],
                measure_indices=measure_indices,
                slot_indices=slot_indices,
                bpm_track=bpm_track,
                measure_track=measure_track,
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
    align_audio_cache_dir = Path(data_cfg["align_audio_cache_dir"]) if data_cfg.get("align_audio_cache_dir") else None

    source_rows = source_by_sample(source_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(source_split_dir / "train.csv")
    use_binned_v10 = any(name in BINNED_V10_NAMES - {"bpm_rhythm_bin"} for name in condition_names)
    if use_binned_v10:
        bin_thresholds, bin_representatives = compute_bin_thresholds(train_rows, source_rows, frame_ms)
    else:
        bin_thresholds, bin_representatives = None, None
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
            bin_thresholds,
        )
        if align_audio_cache_dir is not None:
            audio_ids = {row["chunk_id"] for row in read_rows(align_audio_cache_dir / f"{split_name}.csv")}
            kept = [(row, condition) for row, condition in zip(chunk_rows, conditions) if row["chunk_id"] in audio_ids]
            chunk_rows = [row for row, _ in kept]
            conditions = [condition for _, condition in kept]
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
    if bin_thresholds is not None:
        stats["bin_thresholds"] = bin_thresholds
        stats["bin_representatives"] = bin_representatives
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **split_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
