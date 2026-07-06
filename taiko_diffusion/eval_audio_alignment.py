from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def raw_condition_map(sample: np.lib.npyio.NpzFile) -> dict[str, float]:
    names = [str(name) for name in sample["condition_names"]]
    values = sample["raw_condition"].astype(np.float32)
    return {name: float(values[index]) for index, name in enumerate(names)}


def generated_note_mask(
    probability: np.ndarray,
    condition: dict[str, float],
    frame_ms: float,
    onset: np.ndarray | None = None,
    onset_mix: float = 0.0,
    channel_names: list[str] | None = None,
) -> np.ndarray:
    note_count = int(round(max(condition.get("avg_density", 0.0), 0.0) * probability.shape[0] * frame_ms / 1000.0))
    note_count = max(0, min(note_count, probability.shape[0]))
    mask = np.zeros(probability.shape[0], dtype=bool)
    if note_count > 0:
        channel = {name: index for index, name in enumerate(channel_names or [])}
        if "note_event" in channel:
            score = probability[:, channel["note_event"]]
        elif "don" in channel and "ka" in channel:
            score = np.maximum(probability[:, channel["don"]], probability[:, channel["ka"]])
        else:
            score = probability[:, 0]
        if onset is not None and onset_mix > 0.0:
            onset_score = onset
            if onset_score.shape[0] != score.shape[0]:
                x_old = np.linspace(0.0, 1.0, num=onset_score.shape[0], dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, num=score.shape[0], dtype=np.float32)
                onset_score = np.interp(x_new, x_old, onset_score).astype(np.float32)
            onset_score = onset_score - float(onset_score.min())
            onset_score = onset_score / max(float(onset_score.max()), 1e-6)
            score = score + float(onset_mix) * onset_score
        mask[np.argpartition(score, -note_count)[-note_count:]] = True
    return mask


def generated_ka_mask(probability: np.ndarray, note_mask: np.ndarray, channel_names: list[str]) -> np.ndarray:
    channel = {name: index for index, name in enumerate(channel_names)}
    output = np.zeros(probability.shape[0], dtype=bool)
    if "ka_probability" in channel:
        output[note_mask] = probability[note_mask, channel["ka_probability"]] > 0.5
    elif "don" in channel and "ka" in channel:
        output[note_mask] = probability[note_mask, channel["ka"]] > probability[note_mask, channel["don"]]
    elif "ka" in channel:
        output[note_mask] = probability[note_mask, channel["ka"]] > 0.5
    return output


def load_audio(
    sample: np.lib.npyio.NpzFile,
    audio_split: Path | None,
    audio_stats: Path | None,
    chunk_id: str,
) -> np.ndarray:
    if audio_split is not None:
        rows = read_rows_by_chunk(audio_split)
        row = rows[chunk_id]
        data = np.load(row["audio_npz_path"], allow_pickle=False)
        return data["audio"].astype(np.float32).transpose(1, 0)
    if "audio" in sample.files and sample["audio"].size:
        return sample["audio"].astype(np.float32)
    raise ValueError("Sample has no embedded audio; provide --audio-split for cached raw audio.")


def summarize(mask: np.ndarray, onset: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        selected_mean = 0.0
        selected_top25 = 0.0
    else:
        selected = onset[mask]
        selected_mean = float(selected.mean())
        threshold = float(np.quantile(onset, 0.75))
        selected_top25 = float((selected >= threshold).mean())
    return {
        "notes": int(mask.sum()),
        "onset_mean_at_notes": selected_mean,
        "onset_mean_all": float(onset.mean()),
        "onset_top25_hit_rate": selected_top25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate note positions against cached audio onset.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--chart-split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--audio-split", type=Path, default=None)
    parser.add_argument("--audio-stats", type=Path, default=None)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=False)
    chunk_id = str(sample["source_chunk_id"][0])
    condition = raw_condition_map(sample)
    probability = sample["probability"].astype(np.float32)
    channel_names = [str(name) for name in sample["target_channels"]]
    audio = load_audio(sample, args.audio_split, args.audio_stats, chunk_id)
    onset = audio[-2]
    generated_mask = generated_note_mask(
        probability,
        condition,
        float(args.frame_ms),
        onset=onset,
        onset_mix=float(args.onset_mix),
        channel_names=channel_names,
    )

    chart_rows = read_rows_by_chunk(args.chart_split)
    chart_data = np.load(chart_rows[chunk_id]["npz_path"], allow_pickle=False)
    target = chart_data["chart"].astype(np.float32)
    if "note_event" in channel_names:
        target_mask = target[:, channel_names.index("note_event")] > 0.5
    elif "don" in channel_names and "ka" in channel_names:
        target_mask = (target[:, channel_names.index("don")] > 0.5) | (target[:, channel_names.index("ka")] > 0.5)
    else:
        target_mask = target[:, 0] > 0.5
    result = {
        "sample": str(args.sample),
        "chunk_id": chunk_id,
        "generated": summarize(generated_mask, onset),
        "target": summarize(target_mask, onset),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
