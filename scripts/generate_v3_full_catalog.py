from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import torch

from taiko_diffusion.config import load_config
from taiko_diffusion.data.audio import (
    aligned_window,
    audio_features,
    decode_audio_ffmpeg,
    parse_offset_seconds,
    resolve_audio_path,
)
from taiko_diffusion.data.build_v2_multimodal_cache import pool_time
from taiko_diffusion.data.grid import chart_to_grid
from taiko_diffusion.data.hand_techniques import technique_tracks_from_grid
from taiko_diffusion.data.tja import (
    BranchUnsupportedError,
    TjaParseError,
    parse_tja_course,
    read_raw_courses,
)
from taiko_diffusion.models.v2_multimodal import V2TechniqueHeadEncoder


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LABELS = ("main", "stamina", "handspeed", "burst", "complex", "rhythm")
TARGET_COURSES = {"Hard", "Oni", "Edit"}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def select_raw_course(path: Path, course: str, level: Any):
    matches = [
        raw for raw in read_raw_courses(path)
        if str(raw.meta.get("COURSE", "")).casefold() == course.casefold()
    ]
    if not matches:
        raise TjaParseError(f"Course {course!r} not found: {path}")
    level_text = str(level or "").strip()
    level_matches = [raw for raw in matches if str(raw.meta.get("LEVEL", "")).strip() == level_text]
    return (level_matches or matches)[0]


def parse_ambiguous(path: Path, course: str, level: Any, discard_branch: bool):
    raw = select_raw_course(path, course, level)
    lines = [f"{key}:{value}" for key, value in raw.meta.items() if key != "COURSE"]
    lines.extend([f"COURSE:{raw.meta.get('COURSE', course)}", "#START", *raw.body, "#END"])
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".tja", encoding="utf-8", delete=False) as file:
            temp_name = file.name
            file.write("\n".join(lines) + "\n")
        try:
            parsed = parse_tja_course(Path(temp_name), course, discard_branch=discard_branch)
        except BranchUnsupportedError:
            parsed = parse_tja_course(Path(temp_name), course, discard_branch=False)
        parsed.path = str(path)
        return parsed, raw
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def parse_chart(row: dict[str, Any], discard_branch: bool):
    path = resolve_workspace_path(str((row.get("ese") or {}).get("path") or ""))
    course = str(row.get("course") or "")
    try:
        try:
            parsed = parse_tja_course(path, course, discard_branch=discard_branch)
        except BranchUnsupportedError:
            parsed = parse_tja_course(path, course, discard_branch=False)
        raw = select_raw_course(path, course, row.get("level"))
    except TjaParseError as error:
        if "ambiguous" not in str(error).casefold():
            raise
        parsed, raw = parse_ambiguous(path, course, row.get("level"), discard_branch)
    return parsed, raw, path


def resolve_audio_from_raw(path: Path, raw) -> tuple[Path, float]:
    wave = str(raw.meta.get("WAVE", "")).strip()
    candidates = []
    if wave:
        candidates.append((path.parent / wave).resolve())
    for extension in ("*.ogg", "*.mp3", "*.wav"):
        candidates.extend(sorted(path.parent.glob(extension)))
    for candidate in candidates:
        if candidate.exists():
            return candidate, parse_offset_seconds(raw.meta.get("OFFSET"))
    raise FileNotFoundError(f"Audio file not found for {path}, WAVE={wave!r}")


def model_grid(parsed, frame_ms: float, max_frames: int, model_channels: list[str]):
    # Reproduce the released v3 checkpoint's training cache exactly.  That
    # cache was built by passing the 51 configured lanes directly to
    # chart_to_grid; its derived lanes are therefore zero.  The improved final
    # encoder cache is appropriate for the separate const model, but feeding it
    # here would be a train/inference mismatch of roughly 1--3 ability points.
    grid = chart_to_grid(parsed, frame_ms=frame_ms, max_frames=max_frames, channels=model_channels)
    return grid.x.astype(np.float32), list(model_channels), int(grid.duration_frames), bool(grid.clipped)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels = checkpoint["channels"]
    model = V2TechniqueHeadEncoder(channels["chart"], channels["hand"], channels["audio"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, np.asarray(checkpoint["mean"], np.float32), np.asarray(checkpoint["std"], np.float32)


def predict_pending(pending: list[dict[str, Any]], model, mean, std, device, output) -> int:
    if not pending:
        return 0
    chart = torch.from_numpy(np.stack([item["chart"] for item in pending])).to(device)
    hand = torch.from_numpy(np.stack([item["hand"] for item in pending])).to(device)
    audio = torch.from_numpy(np.stack([item["audio"] for item in pending])).to(device)
    with torch.no_grad():
        normalized = model(chart, hand, audio).cpu().numpy()
    predictions = np.clip(normalized * std + mean, 0.0, 15.5)
    for item, prediction in zip(pending, predictions):
        output[item["id"]] = {
            **{name: round(float(prediction[index]), 4) for index, name in enumerate(LABELS)},
            "course": item["course"],
            "title": item["title"],
            "clipped": item["clipped"],
            "model": "encoder_custom_abilities_v3_physical",
        }
    count = len(pending)
    pending.clear()
    return count


def prepare_group(
    rows: list[dict[str, Any]],
    *,
    discard_branch: bool,
    frame_ms: float,
    max_frames: int,
    expected_channels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        first_parsed, first_raw, tja_path = parse_chart(rows[0], discard_branch)
        try:
            audio_path, _ = resolve_audio_path(tja_path, str(rows[0].get("course") or ""))
        except TjaParseError:
            audio_path, _ = resolve_audio_from_raw(tja_path, first_raw)
        waveform = decode_audio_ffmpeg(audio_path, 22050)
        full_audio = audio_features(waveform, 22050, frame_ms, 2048, 64, 30.0, 11025.0)
        del waveform
        for row_index, row in enumerate(rows):
            parsed, raw, _ = (first_parsed, first_raw, tja_path) if row_index == 0 else parse_chart(row, discard_branch)
            _audio_path, offset = resolve_audio_from_raw(tja_path, raw)
            x, channels, duration, clipped = model_grid(parsed, frame_ms, max_frames, expected_channels)
            if channels != expected_channels:
                raise ValueError(f"derived channels differ from checkpoint config: {len(channels)} != {len(expected_channels)}")
            aligned = aligned_window(full_audio, 0, x.shape[0], offset, 22050, frame_ms)
            hand_tracks, _ = technique_tracks_from_grid(x[:duration], channels, frame_ms)
            prepared.append(
                {
                    "id": str(row.get("id")),
                    "title": str(row.get("title") or ""),
                    "course": str(row.get("course") or ""),
                    "clipped": clipped,
                    "chart": pool_time(x[:duration], 256),
                    "hand": pool_time(hand_tracks, 256),
                    "audio": pool_time(aligned[:duration], 256),
                }
            )
        del full_audio
    except Exception as error:
        for row in rows:
            errors.append(
                {
                    "id": str(row.get("id")),
                    "title": str(row.get("title") or ""),
                    "course": str(row.get("course") or ""),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        prepared.clear()
    return prepared, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v3 abilities for the complete rating chart catalog.")
    parser.add_argument("--chart-data", type=Path, default=WORKSPACE / "taiko_rating/data/chart_data.json")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "taiko_rating/data/v3_abilities.json")
    parser.add_argument("--summary", type=Path, default=WORKSPACE / "taiko_rating/data/v3_abilities_summary.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/encoder_custom_abilities_v3_physical/best.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/encoder_final_main.yaml")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all-courses", action="store_true", help="Include Easy and every Normal chart too.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    frame_ms = float(config["data"].get("frame_ms", 46.4399))
    max_frames = int(config["data"].get("max_frames", 8192))
    discard_branch = bool(config["data"].get("discard_branch", True))
    expected_channels = [str(value) for value in config["chart_grid"]["channels"]]
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    model, mean, std = load_model(args.checkpoint, device)

    charts = read_json(args.chart_data, [])
    selected = [
        row for row in charts
        if args.all_courses or str(row.get("course")) in TARGET_COURSES or bool(row.get("force_included"))
    ]
    if args.limit:
        selected = selected[: args.limit]
    existing = {} if args.overwrite else read_json(args.output, {})
    output = dict(existing)
    todo = [row for row in selected if str(row.get("id")) not in output]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in todo:
        grouped[str((row.get("ese") or {}).get("path") or "")].append(row)

    pending: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    generated = 0
    start = time.time()
    group_iterator = iter(grouped.values())
    workers = max(1, int(args.workers))
    completed_paths = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v3-prepare") as executor:
        futures = {}
        for _ in range(min(workers * 2, len(grouped))):
            rows = next(group_iterator, None)
            if rows is None:
                break
            future = executor.submit(
                prepare_group,
                rows,
                discard_branch=discard_branch,
                frame_ms=frame_ms,
                max_frames=max_frames,
                expected_channels=expected_channels,
            )
            futures[future] = True
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                prepared, group_errors = future.result()
                pending.extend(prepared)
                errors.extend(group_errors)
                completed_paths += 1
                if len(pending) >= args.batch_size:
                    generated += predict_pending(pending, model, mean, std, device, output)
                rows = next(group_iterator, None)
                if rows is not None:
                    next_future = executor.submit(
                        prepare_group,
                        rows,
                        discard_branch=discard_branch,
                        frame_ms=frame_ms,
                        max_frames=max_frames,
                        expected_channels=expected_channels,
                    )
                    futures[next_future] = True
            if completed_paths % 25 == 0:
                generated += predict_pending(pending, model, mean, std, device, output)
                write_json(args.output, output)
                print(json.dumps({"paths": completed_paths, "total_paths": len(grouped), "generated": generated, "errors": len(errors)}, ensure_ascii=False), flush=True)

    generated += predict_pending(pending, model, mean, std, device, output)
    write_json(args.output, output)
    covered = [row for row in selected if str(row.get("id")) in output]
    summary = {
        "chart_rows": len(charts),
        "selected": len(selected),
        "already_present": len(selected) - len(todo),
        "generated": generated,
        "covered": len(covered),
        "coverage": len(covered) / max(len(selected), 1),
        "by_course": dict(Counter(str(row.get("course")) for row in covered)),
        "forced_normal": [
            {"id": row.get("id"), "title": row.get("title"), "covered": str(row.get("id")) in output}
            for row in selected if row.get("force_included")
        ],
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.time() - start, 2),
        "errors": errors,
    }
    write_json(args.summary, summary)
    print(json.dumps({**summary, "errors": errors[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
