from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from taiko_diffusion.data.grid import CHANNELS, chart_to_grid
from taiko_diffusion.data.tja import BranchUnsupportedError, TjaParseError, parse_tja_course
from taiko_diffusion.config import load_config


DEFAULT_LABELS = [
    "const",
    "complex",
    "avg_density",
    "peak_density",
    "bpm_change",
    "hs_change",
    "rhythm",
    "combo",
    "roll_time",
    "balloon_num",
]


def parse_seconds(value: str) -> float:
    if value is None:
        return 0.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


def label_value(row: dict[str, str], name: str) -> float:
    if name == "roll_time":
        return parse_seconds(row.get(name, ""))
    value = row.get(name, "")
    if value == "":
        return 0.0
    return float(value)


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build numpy tensor cache from a matched Taiko manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder_v0.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/encoder_v0"))
    parser.add_argument("--frame-ms", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest-base-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    config = load_config(args.config) if args.config is not None else {}
    data_config = config.get("data", {})
    grid_config = config.get("chart_grid", {})
    manifest = args.manifest or Path(data_config.get("manifest", "data/manifests/strict_matched_dataset.csv"))
    frame_ms = args.frame_ms or float(data_config.get("frame_ms", 46.4399))
    max_frames = args.max_frames or int(data_config.get("max_frames", 8192))
    label_names = list(data_config.get("target_columns", DEFAULT_LABELS))
    channels = list(grid_config.get("channels", CHANNELS))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str | int | float | bool]] = []
    errors: list[dict[str, str]] = []

    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if args.limit > 0:
        rows = rows[: args.limit]

    for sample_index, row in enumerate(rows):
        try:
            chart_path = resolve_path(row["ese_path"], args.manifest_base_dir)
            chart = parse_tja_course(chart_path, row["ese_course"], discard_branch=True)
            grid = chart_to_grid(chart, frame_ms=frame_ms, max_frames=max_frames, channels=channels)
            y = np.asarray([label_value(row, name) for name in label_names], dtype=np.float32)
            sample_id = f"{sample_index:06d}_r{row['rating_index']}"
            npz_path = output_dir / f"{sample_id}.npz"
            np.savez_compressed(
                npz_path,
                x=grid.x,
                y=y,
                channels=np.asarray(channels),
                label_names=np.asarray(label_names),
                duration_frames=np.asarray([grid.duration_frames], dtype=np.int32),
            )
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "npz_path": str(npz_path),
                    "title": row["title"],
                    "ese_path": str(chart_path),
                    "course": row["ese_course"],
                    "duration_frames": grid.duration_frames,
                    "clipped": grid.clipped,
                    "parsed_note_count": chart.playable_note_count,
                    "manifest_note_count": row.get("ese_note_count", ""),
                    "combo": row.get("combo", ""),
                }
            )
        except (BranchUnsupportedError, TjaParseError, OSError, ValueError) as exc:
            errors.append({"title": row.get("title", ""), "ese_path": row.get("ese_path", ""), "error": str(exc)})

    index_path = output_dir / "index.csv"
    if index_rows:
        with index_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
    error_path = output_dir / "errors.json"
    error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "frame_ms": frame_ms,
        "max_frames": max_frames,
        "channels": channels,
        "label_names": label_names,
        "requested_rows": len(rows),
        "written": len(index_rows),
        "errors": len(errors),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
