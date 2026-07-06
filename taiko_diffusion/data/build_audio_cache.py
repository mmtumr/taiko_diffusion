from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config
from taiko_diffusion.data.audio import (
    aligned_window,
    audio_features,
    decode_audio_ffmpeg,
    read_rows,
    resolve_audio_path,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_rows_by_sample(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_rows(path)}


def process_split(
    split_name: str,
    chunk_cache_dir: Path,
    output_dir: Path,
    source_rows: dict[str, dict[str, str]],
    sample_rate: int,
    frame_ms: float,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    limit: int,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[dict[str, str]]]:
    chunk_rows = read_rows(chunk_cache_dir / f"{split_name}.csv")
    if limit > 0:
        chunk_rows = chunk_rows[:limit]
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in chunk_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    feature_summaries: list[np.ndarray] = []
    errors: list[dict[str, str]] = []

    total_samples = len(by_sample)
    for sample_index, (sample_id, rows) in enumerate(by_sample.items(), start=1):
        if sample_index == 1 or sample_index % 50 == 0 or sample_index == total_samples:
            print(
                json.dumps(
                    {
                        "split": split_name,
                        "sample": sample_index,
                        "total_samples": total_samples,
                        "written_windows": len(index_rows),
                        "errors": len(errors),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        source = source_rows[sample_id]
        tja_path = Path(source["ese_path"])
        course = source["course"]
        try:
            audio_path, offset_seconds = resolve_audio_path(tja_path, course)
            waveform = decode_audio_ffmpeg(audio_path, sample_rate)
            full_features = audio_features(waveform, sample_rate, frame_ms, n_fft, n_mels, f_min, f_max)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "title": source.get("title", ""), "error": str(exc)})
            continue

        for row in rows:
            chunk_npz = np.load(row["npz_path"], allow_pickle=False)
            window_frames = int(chunk_npz["chart"].shape[0])
            start_frame = int(row["start_frame"])
            features = aligned_window(
                full_features,
                start_frame,
                window_frames,
                offset_seconds,
                sample_rate,
                frame_ms,
            )
            audio_npz_path = split_dir / f"{row['chunk_id']}.npz"
            np.savez_compressed(
                audio_npz_path,
                audio=features,
                feature_names=np.asarray([*[f"mel_{index:02d}" for index in range(n_mels)], "onset", "rms"]),
                sample_id=np.asarray([sample_id]),
                chunk_id=np.asarray([row["chunk_id"]]),
                audio_path=np.asarray([str(audio_path)]),
                offset_seconds=np.asarray([offset_seconds], dtype=np.float32),
                start_frame=np.asarray([start_frame], dtype=np.int32),
            )
            index_rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "sample_id": sample_id,
                    "audio_npz_path": str(audio_npz_path),
                    "chart_npz_path": row["npz_path"],
                    "title": row.get("title", ""),
                    "audio_path": str(audio_path),
                    "offset_seconds": offset_seconds,
                    "start_frame": start_frame,
                    "window_frames": window_frames,
                }
            )
            feature_summaries.append(features.reshape(-1, features.shape[-1]))
    return index_rows, feature_summaries, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio feature cache aligned to diffusion chart windows.")
    parser.add_argument("--config", type=Path, default=Path("configs/audio_v0.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split chunk limit for smoke tests.")
    args = parser.parse_args()

    config = load_config(args.config)
    audio_cfg = config["audio"]
    source_index = Path(audio_cfg["source_index"])
    chunk_cache_dir = Path(audio_cfg["chunk_cache_dir"])
    output_dir = Path(audio_cfg["output_dir"])
    sample_rate = int(audio_cfg.get("sample_rate", 22050))
    frame_ms = float(audio_cfg.get("frame_ms", 46.4399))
    n_fft = int(audio_cfg.get("n_fft", 2048))
    n_mels = int(audio_cfg.get("n_mels", 64))
    f_min = float(audio_cfg.get("f_min", 30.0))
    f_max = float(audio_cfg.get("f_max", sample_rate / 2))
    splits = [str(name) for name in audio_cfg.get("splits", ["train", "val", "test"])]
    source_rows = source_rows_by_sample(source_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_features: list[np.ndarray] = []
    all_errors: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    for split_name in splits:
        rows, summaries, errors = process_split(
            split_name,
            chunk_cache_dir,
            output_dir,
            source_rows,
            sample_rate,
            frame_ms,
            n_fft,
            n_mels,
            f_min,
            f_max,
            args.limit,
        )
        write_rows(output_dir / f"{split_name}.csv", rows)
        split_counts[split_name] = len(rows)
        all_errors.extend(errors)
        if split_name == "train":
            all_train_features.extend(summaries)

    if all_train_features:
        train_features = np.concatenate(all_train_features, axis=0)
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        mean = np.zeros(n_mels + 2, dtype=np.float32)
        std = np.ones(n_mels + 2, dtype=np.float32)

    stats = {
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "hop_length": int(round(sample_rate * frame_ms / 1000.0)),
        "n_fft": n_fft,
        "n_mels": n_mels,
        "feature_names": [*[f"mel_{index:02d}" for index in range(n_mels)], "onset", "rms"],
        "feature_mean": mean.astype(float).tolist(),
        "feature_std": std.astype(float).tolist(),
        "splits": split_counts,
        "errors": len(all_errors),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "errors.json").write_text(json.dumps(all_errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **split_counts, "errors": len(all_errors)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
