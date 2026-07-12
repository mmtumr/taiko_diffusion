from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from taiko_diffusion.data.audio import aligned_window, audio_features, decode_audio_ffmpeg, resolve_audio_path


def pool_time(values: np.ndarray, tokens: int) -> np.ndarray:
    tensor = torch.from_numpy(values.T).unsqueeze(0)
    avg = F.adaptive_avg_pool1d(tensor, tokens)
    maximum = F.adaptive_max_pool1d(tensor, tokens)
    return torch.cat([avg, maximum], dim=1).squeeze(0).numpy().astype(np.float32)


def hand_tracks(x: np.ndarray, channels: list[str]) -> np.ndarray:
    note = (x[:, channels.index("don")] + x[:, channels.index("ka")]) > 0.5
    don = x[:, channels.index("don")] > 0.5
    indices = np.flatnonzero(note)
    tracks = np.zeros((x.shape[0], 12), dtype=np.float32)
    if not len(indices):
        return tracks
    phrase = 0
    local = 0
    previous = -10_000
    for global_index, frame in enumerate(indices):
        if frame - previous > 22:
            phrase += 1
            local = 0
        # 均衡 / 逆半均衡: each separated phrase returns to 正手 / 反手.
        assignments = [local % 2, 1 - local % 2]
        # 半换: odd phrases reverse the phrase-starting hand.
        assignments.append((local + phrase) % 2)
        # 全换: never reset the alternating order.
        assignments.append(global_index % 2)
        # 硬抗: short groups keep their starting hand for pairs.
        assignments.append((local // 2) % 2)
        # 半分工: 咚由正手、咔由反手处理.
        assignments.append(0 if don[frame] else 1)
        for method, hand in enumerate(assignments):
            tracks[frame, method * 2 + hand] = 1.0
        local += 1
        previous = frame
    return tracks


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = {int(row["rating_index"]): row for row in read_rows(root / "data/manifests/v2_regression.csv")}
    source_rows = []
    for split in ("train", "val", "test"):
        for row in read_rows(root / f"data/splits/encoder_v2_regression/{split}.csv"):
            row["split"] = split
            source_rows.append(row)
    output = root / "data/cache/encoder_v2_multimodal"
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[dict[str, object]]] = {name: [] for name in ("train", "val", "test")}
    errors = []
    for index, row in enumerate(source_rows, 1):
        rating_index = int(row["sample_id"].rsplit("_r", 1)[1])
        meta = manifest[rating_index]
        try:
            chart = np.load(row["npz_path"], allow_pickle=False)
            x = chart["x"].astype(np.float32)
            channels = [str(value) for value in chart["channels"]]
            duration = int(chart["duration_frames"][0])
            audio_path, offset = resolve_audio_path(Path(meta["ese_path"]), meta["ese_course"])
            waveform = decode_audio_ffmpeg(audio_path, 22050)
            full_audio = audio_features(waveform, 22050, 46.4399, 2048, 64, 30.0, 11025.0)
            audio = aligned_window(full_audio, 0, x.shape[0], offset, 22050, 46.4399)
            chart_pool = pool_time(x[:duration], args.tokens)
            hand_pool = pool_time(hand_tracks(x[:duration], channels), args.tokens)
            audio_pool = pool_time(audio[:duration], args.tokens)
            target_names = ["v2_main", "v2_stamina", "v2_handspeed", "v2_burst", "v2_complex", "v2_rhythm"]
            y = np.asarray([float(meta[name]) for name in target_names], dtype=np.float32)
            path = output / f"{row['sample_id']}.npz"
            np.savez_compressed(path, chart=chart_pool, hand=hand_pool, audio=audio_pool, y=y)
            written[row["split"]].append({"sample_id": row["sample_id"], "title": row["title"], "npz_path": str(path)})
        except Exception as exc:
            errors.append({"sample_id": row["sample_id"], "error": str(exc)})
        if index == 1 or index % 50 == 0 or index == len(source_rows):
            print(json.dumps({"done": index, "total": len(source_rows), "errors": len(errors)}), flush=True)
    for split, rows in written.items():
        with (output / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["sample_id", "title", "npz_path"])
            writer.writeheader(); writer.writerows(rows)
    train_y = np.stack([np.load(row["npz_path"])["y"] for row in written["train"]])
    stats = {"mean": train_y.mean(0).tolist(), "std": np.maximum(train_y.std(0), 1e-6).tolist(), "errors": errors}
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
