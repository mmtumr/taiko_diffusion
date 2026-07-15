from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from taiko_diffusion.data.diffusion_dataset import local_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalized_path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def dense_segments(note: np.ndarray, window: int, minimum_notes: int, minimum_frames: int, maximum_frames: int) -> list[tuple[int, int]]:
    density = np.convolve(note.astype(np.float32), np.ones(window, dtype=np.float32), mode="same")
    dense = density >= minimum_notes
    segments = []
    start = None
    for frame, active in enumerate(dense):
        if active and start is None:
            start = frame
        if start is not None and (not active or frame + 1 == len(dense)):
            end = frame if active and frame + 1 == len(dense) else frame - 1
            if end - start + 1 >= minimum_frames:
                for chunk_start in range(start, end + 1, maximum_frames):
                    chunk_end = min(end, chunk_start + maximum_frames - 1)
                    if chunk_end - chunk_start + 1 >= minimum_frames:
                        segments.append((chunk_start, chunk_end))
            start = None
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build low-const hold hints from paired high-const dense passages.")
    parser.add_argument("--encoder-index", type=Path, default=Path("data/cache/encoder_v1/index.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/strict_matched_dataset.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/diffusion_v13_mug_holds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/hold_substitution_hints_v0"))
    parser.add_argument("--min-const-gap", type=float, default=1.0)
    parser.add_argument("--density-window", type=int, default=16)
    parser.add_argument("--minimum-notes", type=int, default=6)
    parser.add_argument("--minimum-frames", type=int, default=8)
    parser.add_argument("--maximum-frames", type=int, default=40)
    parser.add_argument("--maximum-segments", type=int, default=2)
    args = parser.parse_args()

    manifest = {int(row["rating_index"]): row for row in read_rows(args.manifest)}
    encoder_rows = read_rows(args.encoder_index)
    records = []
    for row in encoder_rows:
        rating_index = int(row["sample_id"].rsplit("_r", 1)[1])
        source = manifest.get(rating_index)
        if source is None or row["clipped"].casefold() == "true":
            continue
        records.append({**row, "const": float(source["const"]), "path_key": normalized_path(row["ese_path"])})
    groups: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for record in records:
        groups[str(record["path_key"])].append(record)

    train_rows = read_rows(args.cache_dir / "train.csv")
    train_samples = {row["sample_id"] for row in train_rows}
    chunks = {(row["sample_id"], int(row["start_frame"])): row for row in train_rows}
    pairs = []
    for group in groups.values():
        candidates = [record for record in group if record["sample_id"] in train_samples]
        if len(candidates) != 2:
            continue
        low, high = sorted(candidates, key=lambda record: float(record["const"]))
        if float(high["const"]) - float(low["const"]) < args.min_const_gap:
            continue
        if int(high["duration_frames"]) != int(low["duration_frames"]):
            continue
        pairs.append((low, high))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    hinted_frames = 0
    for low, high in pairs:
        for (sample_id, start_frame), low_row in chunks.items():
            if sample_id != low["sample_id"]:
                continue
            high_row = chunks.get((str(high["sample_id"]), start_frame))
            if high_row is None:
                continue
            low_data = np.load(local_path(low_row["npz_path"]), allow_pickle=False)
            high_data = np.load(local_path(high_row["npz_path"]), allow_pickle=False)
            names = [str(name) for name in high_data["target_channels"]]
            channel = {name: index for index, name in enumerate(names)}
            high_chart = high_data["chart"].astype(np.float32)
            high_note = np.zeros(high_chart.shape[0], dtype=bool)
            for name in ["don", "ka", "big_don", "big_ka"]:
                high_note |= high_chart[:, channel[name]] > 0.5
            candidates = dense_segments(
                high_note, args.density_window, args.minimum_notes, args.minimum_frames, args.maximum_frames
            )
            candidates.sort(key=lambda segment: int(high_note[segment[0] : segment[1] + 1].sum()), reverse=True)
            selected = []
            for begin, end in candidates:
                if any(begin <= other_end + 1 and end >= other_begin - 1 for other_begin, other_end in selected):
                    continue
                selected.append((begin, end))
                if len(selected) >= args.maximum_segments:
                    break
            hint = np.zeros(high_chart.shape[0], dtype=np.float32)
            for begin, end in selected:
                hint[begin : end + 1] = 1.0
            if not hint.any():
                continue
            np.savez_compressed(
                args.output_dir / f"{low_row['chunk_id']}.npz",
                hold_hint=hint,
                high_sample_id=np.asarray([high["sample_id"]]),
                low_const=np.asarray([low["const"]], dtype=np.float32),
                high_const=np.asarray([high["const"]], dtype=np.float32),
            )
            written += 1
            hinted_frames += int(hint.sum())
    summary = {
        "pairs": len(pairs),
        "hinted_windows": written,
        "hinted_frames": hinted_frames,
        "parameters": {
            "min_const_gap": args.min_const_gap,
            "density_window": args.density_window,
            "minimum_notes": args.minimum_notes,
            "minimum_frames": args.minimum_frames,
            "maximum_frames": args.maximum_frames,
            "maximum_segments": args.maximum_segments,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
