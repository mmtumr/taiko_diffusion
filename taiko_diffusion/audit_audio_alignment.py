from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def shifted(values: np.ndarray, shift: int) -> np.ndarray:
    output = np.zeros_like(values)
    if shift >= 0:
        if shift < values.shape[0]:
            output[: values.shape[0] - shift] = values[shift:]
    elif -shift < values.shape[0]:
        output[-shift:] = values[: values.shape[0] + shift]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chart-note alignment against cached raw audio onset.")
    parser.add_argument("--chart-cache", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained"))
    parser.add_argument("--audio-cache", type=Path, default=Path("data/cache/audio_v0"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max-shift", type=int, default=30)
    parser.add_argument("--suspect-improve", type=float, default=0.12)
    parser.add_argument("--output", type=Path, default=Path("eval/audio_alignment_audit.json"))
    args = parser.parse_args()

    stats = json.loads((args.chart_cache / "stats.json").read_text(encoding="utf-8"))
    frame_ms = float(stats["frame_ms"])
    shifts = list(range(-int(args.max_shift), int(args.max_shift) + 1))
    result: dict[str, object] = {"frame_ms": frame_ms, "splits": {}}

    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        chart_rows = {row["chunk_id"]: row for row in read_rows(args.chart_cache / f"{split}.csv")}
        audio_rows = {row["chunk_id"]: row for row in read_rows(args.audio_cache / f"{split}.csv")}
        global_stats = {shift: {"sum": 0.0, "notes": 0, "top": 0} for shift in shifts}
        by_sample: dict[str, dict[int, dict[str, object]]] = {}

        for chunk_id, chart_row in chart_rows.items():
            audio_row = audio_rows.get(chunk_id)
            if audio_row is None:
                continue
            chart = np.load(chart_row["npz_path"], allow_pickle=False)["chart"].astype(np.float32)
            audio = np.load(audio_row["audio_npz_path"], allow_pickle=False)["audio"].astype(np.float32)
            note_mask = chart[:, 0] > 0.5
            if int(note_mask.sum()) == 0:
                continue
            onset = audio[:, -2]
            sample_id = chart_row["sample_id"]
            by_sample.setdefault(
                sample_id,
                {
                    shift: {
                        "sum": 0.0,
                        "notes": 0,
                        "top": 0,
                        "windows": 0,
                        "title": chart_row.get("title", ""),
                        "offset_seconds": audio_row.get("offset_seconds", ""),
                    }
                    for shift in shifts
                },
            )
            for shift in shifts:
                shifted_onset = shifted(onset, shift)
                selected = shifted_onset[note_mask]
                threshold = float(np.quantile(shifted_onset, 0.75))
                top_hits = int((selected >= threshold).sum())
                note_count = int(note_mask.sum())
                global_stats[shift]["sum"] += float(selected.sum())
                global_stats[shift]["notes"] += note_count
                global_stats[shift]["top"] += top_hits
                sample_stats = by_sample[sample_id][shift]
                sample_stats["sum"] = float(sample_stats["sum"]) + float(selected.sum())
                sample_stats["notes"] = int(sample_stats["notes"]) + note_count
                sample_stats["top"] = int(sample_stats["top"]) + top_hits
                sample_stats["windows"] = int(sample_stats["windows"]) + 1

        global_rows = []
        for shift, values in global_stats.items():
            notes = max(int(values["notes"]), 1)
            global_rows.append(
                {
                    "shift_frames": shift,
                    "shift_ms": shift * frame_ms,
                    "onset_mean": float(values["sum"]) / notes,
                    "top25_hit": float(values["top"]) / notes,
                }
            )
        global_best = max(global_rows, key=lambda row: row["onset_mean"])
        sample_bests = []
        for sample_id, shift_stats in by_sample.items():
            rows = []
            for shift, values in shift_stats.items():
                notes = max(int(values["notes"]), 1)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "title": str(values["title"]),
                        "offset_seconds": str(values["offset_seconds"]),
                        "shift_frames": shift,
                        "shift_ms": shift * frame_ms,
                        "onset_mean": float(values["sum"]) / notes,
                        "top25_hit": float(values["top"]) / notes,
                        "windows": int(values["windows"]),
                        "notes": int(values["notes"]),
                    }
                )
            best = max(rows, key=lambda row: row["onset_mean"])
            zero = next(row for row in rows if row["shift_frames"] == 0)
            best["zero_onset_mean"] = zero["onset_mean"]
            best["improvement"] = float(best["onset_mean"]) - float(zero["onset_mean"])
            sample_bests.append(best)

        suspects = [
            row
            for row in sample_bests
            if abs(int(row["shift_frames"])) >= 2
            and float(row["improvement"]) >= float(args.suspect_improve)
            and int(row["notes"]) >= 100
        ]
        split_summary = {
            "windows": len(chart_rows),
            "samples": len(by_sample),
            "global_best": global_best,
            "zero": next(row for row in global_rows if row["shift_frames"] == 0),
            "sample_best_shift_counts": Counter(int(row["shift_frames"]) for row in sample_bests).most_common(),
            "suspects": sorted(suspects, key=lambda row: float(row["improvement"]), reverse=True),
        }
        result["splits"][split] = split_summary
        print(
            json.dumps(
                {
                    "split": split,
                    "windows": split_summary["windows"],
                    "samples": split_summary["samples"],
                    "global_best": split_summary["global_best"],
                    "zero": split_summary["zero"],
                    "suspects": len(suspects),
                    "top_sample_shift_counts": split_summary["sample_best_shift_counts"][:8],
                },
                ensure_ascii=False,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
