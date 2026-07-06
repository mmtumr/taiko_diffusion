from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from taiko_diffusion.eval_audio_alignment import generated_ka_mask, generated_note_mask, summarize


DON = (230, 60, 55)
KA = (68, 140, 230)
TARGET = (235, 235, 235)
GRID = (52, 58, 68)
BG = (14, 16, 20)
PANEL = (24, 28, 34)
TEXT = (230, 234, 241)
MUTED = (142, 151, 166)
GREEN = (80, 210, 145)
YELLOW = (245, 190, 80)


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def read_audio_row(path: Path, chunk_id: str) -> dict[str, str]:
    rows = read_rows_by_chunk(path)
    if chunk_id not in rows:
        raise KeyError(f"chunk_id not found in audio split: {chunk_id}")
    return rows[chunk_id]


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    vmax = float(values.max())
    if vmax <= 1e-6:
        return np.zeros_like(values)
    return values / vmax


def target_masks(chart: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel = {name: index for index, name in enumerate(channel_names)}
    if "don" in channel:
        don = chart[:, channel["don"]] > 0.5
    else:
        don = np.zeros(chart.shape[0], dtype=bool)
    if "ka" in channel:
        ka = chart[:, channel["ka"]] > 0.5
    else:
        ka = np.zeros(chart.shape[0], dtype=bool)
    if "note_event" in channel:
        note = chart[:, channel["note_event"]] > 0.5
    else:
        note = don | ka
    return note, don, ka


def generated_masks(
    probability: np.ndarray,
    condition: dict[str, float],
    channel_names: list[str],
    frame_ms: float,
    onset: np.ndarray,
    onset_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    note = generated_note_mask(
        probability,
        condition,
        frame_ms,
        onset=onset,
        onset_mix=onset_mix,
        channel_names=channel_names,
    )
    ka = generated_ka_mask(probability, note, channel_names)
    don = note & ~ka
    return note, don, ka


def condition_map(sample: np.lib.npyio.NpzFile) -> dict[str, float]:
    names = [str(name) for name in sample["condition_names"]]
    values = sample["raw_condition"].astype(np.float32)
    return {name: float(values[index]) for index, name in enumerate(names)}


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def draw_note(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int = 9) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(250, 250, 250), width=1)


def lane_y(base: int, lane_index: int) -> int:
    return base + lane_index * 86


def draw_frame(
    output: Path,
    frame_index: int,
    total_video_frames: int,
    probability: np.ndarray,
    target_don: np.ndarray,
    target_ka: np.ndarray,
    gen_don: np.ndarray,
    gen_ka: np.ndarray,
    onset: np.ndarray,
    title: str,
    chunk_id: str,
    metrics: dict[str, float],
    frame_ms: float,
    view_frames: int,
) -> None:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    frames = probability.shape[0]
    current = frame_index / max(total_video_frames - 1, 1) * (frames - 1)
    start = max(0.0, min(current - view_frames / 2.0, frames - view_frames))
    end = min(frames, start + view_frames)

    draw.rectangle((0, 0, width, 82), fill=(20, 24, 30))
    draw_text(draw, (28, 20), f"Taiko Diffusion v9 don/ka preview | {title} | {chunk_id}")
    draw_text(
        draw,
        (28, 46),
        "red=don  blue=ka  white rows=target  colored rows=generated  green=audio onset",
        MUTED,
    )
    draw_text(
        draw,
        (1060, 20),
        f"time {current * frame_ms / 1000.0:05.2f}s / {frames * frame_ms / 1000.0:05.2f}s",
        MUTED,
    )
    draw_text(
        draw,
        (1060, 46),
        (
            f"gen top25 {metrics['gen_top25']:.3f} / target {metrics['target_top25']:.3f} | "
            f"ka ratio {metrics['gen_ka_ratio']:.3f} / {metrics['target_ka_ratio']:.3f}"
        ),
        MUTED,
    )

    plot_left, plot_right = 92, 1536
    main_top, main_bottom = 130, 700
    timeline_top, timeline_bottom = 750, 836
    draw.rectangle((56, 104, 1568, 724), fill=PANEL, outline=GRID)
    draw.rectangle((56, 740, 1568, 852), fill=PANEL, outline=GRID)

    for sec in range(0, int(math.ceil(frames * frame_ms / 1000.0)) + 1):
        chart_frame = sec * 1000.0 / frame_ms
        if start <= chart_frame <= end:
            x = plot_left + (chart_frame - start) / max(end - start, 1.0) * (plot_right - plot_left)
            draw.line((x, main_top, x, main_bottom), fill=(38, 44, 54), width=1)
            draw_text(draw, (int(x) + 3, main_bottom + 6), f"{sec}s", (90, 98, 112))

    labels = ["onset", "target don", "target ka", "gen don", "gen ka", "confidence"]
    for i, label in enumerate(labels):
        y = lane_y(main_top + 24, i)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw_text(draw, (14, y - 7), label, MUTED)

    cursor_x = plot_left + (current - start) / max(end - start, 1.0) * (plot_right - plot_left)
    draw.line((cursor_x, main_top - 18, cursor_x, main_bottom + 22), fill=YELLOW, width=2)

    onset_norm = normalize(onset)
    visible_indices = np.arange(int(max(0, math.floor(start))), int(min(frames, math.ceil(end))))
    if visible_indices.size > 1:
        points = []
        y_base = lane_y(main_top + 24, 0)
        for idx in visible_indices:
            x = plot_left + (idx - start) / max(end - start, 1.0) * (plot_right - plot_left)
            y = y_base + 26 - float(onset_norm[idx]) * 52
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=GREEN, width=2)

    note_rows = [
        (target_don, lane_y(main_top + 24, 1), DON, TARGET),
        (target_ka, lane_y(main_top + 24, 2), KA, TARGET),
        (gen_don, lane_y(main_top + 24, 3), DON, DON),
        (gen_ka, lane_y(main_top + 24, 4), KA, KA),
    ]
    for mask, y, color, outline_color in note_rows:
        for idx in visible_indices[mask[visible_indices]]:
            x = plot_left + (idx - start) / max(end - start, 1.0) * (plot_right - plot_left)
            draw_note(draw, x, y, color if outline_color != TARGET else TARGET, radius=8)
            if outline_color == TARGET:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    channel_count = probability.shape[1]
    note_score = probability.max(axis=1) if channel_count > 1 else probability[:, 0]
    score_norm = normalize(note_score)
    y_conf = lane_y(main_top + 24, 5)
    for idx in visible_indices:
        x = plot_left + (idx - start) / max(end - start, 1.0) * (plot_right - plot_left)
        bar_h = float(score_norm[idx]) * 44
        color = (80, 105, 140)
        draw.line((x, y_conf + 24, x, y_conf + 24 - bar_h), fill=color, width=2)

    for idx in range(frames):
        x = plot_left + idx / max(frames - 1, 1) * (plot_right - plot_left)
        if onset_norm[idx] > 0.72:
            draw.line((x, timeline_top + 4, x, timeline_top + 22), fill=(40, 98, 76), width=1)
        if target_don[idx] or target_ka[idx]:
            draw.line((x, timeline_top + 36, x, timeline_top + 52), fill=TARGET, width=1)
        if gen_don[idx]:
            draw.line((x, timeline_top + 66, x, timeline_top + 82), fill=DON, width=1)
        if gen_ka[idx]:
            draw.line((x, timeline_top + 66, x, timeline_top + 82), fill=KA, width=1)
    draw.line((cursor_x, timeline_top, cursor_x, timeline_bottom), fill=YELLOW, width=2)
    draw_text(draw, (14, timeline_top + 4), "full")
    draw_text(draw, (14, timeline_top + 34), "target")
    draw_text(draw, (14, timeline_top + 64), "generated")

    image.save(output)


def run_ffmpeg(frames_dir: Path, output: Path, fps: int, audio_row: dict[str, str] | None, duration_sec: float, frame_ms: float) -> None:
    pattern = str(frames_dir / "%05d.png")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
    ]
    if audio_row is not None:
        start_frame = int(audio_row.get("start_frame", 0))
        offset_seconds = float(audio_row.get("offset_seconds", 0.0))
        audio_start = max(0.0, start_frame * frame_ms / 1000.0 - offset_seconds)
        command.extend(["-ss", f"{audio_start:.6f}", "-t", f"{duration_sec:.6f}", "-i", audio_row["audio_path"]])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if audio_row is not None:
        command.extend(["-c:a", "aac", "-shortest"])
    command.append(str(output))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an MP4 preview for a latent diffusion sample.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--chart-split", type=Path, default=Path("data/cache/diffusion_v9_donka/test.csv"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view-seconds", type=float, default=5.5)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=False)
    probability = sample["probability"].astype(np.float32)
    channel_names = [str(name) for name in sample["target_channels"]]
    condition = condition_map(sample)
    chunk_id = str(sample["source_chunk_id"][0])
    title = str(sample["source_title"][0]) if "source_title" in sample.files else chunk_id

    chart_rows = read_rows_by_chunk(args.chart_split)
    chart_data = np.load(chart_rows[chunk_id]["npz_path"], allow_pickle=False)
    chart = chart_data["chart"].astype(np.float32)
    target_note, target_don, target_ka = target_masks(chart, channel_names)

    audio_row = read_audio_row(args.audio_split, chunk_id)
    audio = np.load(audio_row["audio_npz_path"], allow_pickle=False)["audio"].astype(np.float32)
    onset = audio[:, -2]
    gen_note, gen_don, gen_ka = generated_masks(
        probability,
        condition,
        channel_names,
        float(args.frame_ms),
        onset,
        float(args.onset_mix),
    )

    gen_summary = summarize(gen_note, onset)
    target_summary = summarize(target_note, onset)
    metrics = {
        "gen_top25": float(gen_summary["onset_top25_hit_rate"]),
        "target_top25": float(target_summary["onset_top25_hit_rate"]),
        "gen_ka_ratio": float(gen_ka.sum() / max(gen_note.sum(), 1)),
        "target_ka_ratio": float(target_ka.sum() / max(target_note.sum(), 1)),
    }

    frames = probability.shape[0]
    duration_sec = frames * float(args.frame_ms) / 1000.0
    video_frames = max(1, int(round(duration_sec * int(args.fps))))
    view_frames = max(8, int(round(float(args.view_seconds) * 1000.0 / float(args.frame_ms))))
    frames_dir = args.output.with_suffix("").parent / f"{args.output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index in range(video_frames):
        draw_frame(
            frames_dir / f"{index:05d}.png",
            index,
            video_frames,
            probability,
            target_don,
            target_ka,
            gen_don,
            gen_ka,
            onset,
            title,
            chunk_id,
            metrics,
            float(args.frame_ms),
            view_frames,
        )
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == video_frames:
            print(json.dumps({"frame": index + 1, "total": video_frames}, ensure_ascii=False), flush=True)

    ffmpeg_audio_row = None if args.no_audio else audio_row
    run_ffmpeg(frames_dir, args.output, int(args.fps), ffmpeg_audio_row, duration_sec, float(args.frame_ms))
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(
        json.dumps(
            {
                "sample": str(args.sample),
                "output": str(args.output),
                "chunk_id": chunk_id,
                "title": title,
                "duration_sec": duration_sec,
                "fps": int(args.fps),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
