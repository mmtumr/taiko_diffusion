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

from taiko_diffusion.data.diffusion_dataset import local_path
from taiko_diffusion.eval_audio_alignment import generated_ka_mask, generated_note_mask


BG = (16, 18, 22)
LANE = (42, 45, 52)
LANE_EDGE = (72, 76, 86)
DON = (224, 54, 48)
KA = (64, 132, 222)
WHITE = (245, 246, 248)
MUTED = (152, 160, 174)
YELLOW = (248, 198, 82)
HIT_RING = (250, 238, 210)


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def local_audio_path(value: str) -> Path:
    path = local_path(value)
    if path.exists():
        return path
    marker = "ESE-master/ese/"
    normalized = value.replace("\\", "/")
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = Path(__file__).resolve().parents[2] / "ESE-master_extracted" / "ese" / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Audio file not found: {value}")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path(r"C:\Windows\Fonts\msgothic.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def condition_map(sample: np.lib.npyio.NpzFile) -> dict[str, float]:
    names = [str(name) for name in sample["condition_names"]]
    values = sample["raw_condition"].astype(np.float32)
    return {name: float(values[index]) for index, name in enumerate(names)}


def masks_from_sample(
    sample: np.lib.npyio.NpzFile,
    audio_split: Path,
    frame_ms: float,
    onset_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str], np.ndarray]:
    chunk_id = str(sample["source_chunk_id"][0])
    audio_rows = read_rows_by_chunk(audio_split)
    audio_row = audio_rows[chunk_id]
    audio_data = np.load(local_path(audio_row["audio_npz_path"]), allow_pickle=False)
    onset = audio_data["audio"].astype(np.float32)[:, -2]

    probability = sample["probability"].astype(np.float32)
    channel_names = [str(name) for name in sample["target_channels"]]
    note = generated_note_mask(
        probability,
        condition_map(sample),
        frame_ms,
        onset=onset,
        onset_mix=onset_mix,
        channel_names=channel_names,
    )
    ka = generated_ka_mask(probability, note, channel_names)
    don = note & ~ka
    return note, don, ka, audio_row, onset


def draw_note(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    shadow = (0, 0, 0)
    draw.ellipse((x - radius + 3, y - radius + 4, x + radius + 3, y + radius + 4), fill=shadow)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=4)
    inner = tuple(min(255, c + 28) for c in color)
    draw.ellipse((x - radius * 0.48, y - radius * 0.48, x + radius * 0.48, y + radius * 0.48), fill=inner)


def draw_frame(
    path: Path,
    frame_index: int,
    total_frames: int,
    don_times: np.ndarray,
    ka_times: np.ndarray,
    onset: np.ndarray,
    title: str,
    duration_sec: float,
    fps: int,
    approach_sec: float,
    frame_ms: float,
) -> None:
    width, height = 1600, 900
    t = frame_index / fps
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(38)
    small_font = load_font(24)
    count_font = load_font(30)

    # Background bands.
    draw.rectangle((0, 0, width, height), fill=BG)
    for y, alpha in [(0, 28), (760, 34)]:
        draw.rectangle((0, y, width, y + 140), fill=(22 + alpha // 8, 25 + alpha // 8, 30 + alpha // 8))

    lane_y = 430
    lane_h = 148
    hit_x = 260
    spawn_x = 1510
    note_radius = 40

    draw.rounded_rectangle((86, lane_y - lane_h // 2, 1538, lane_y + lane_h // 2), radius=26, fill=LANE, outline=LANE_EDGE, width=4)
    draw.line((hit_x, lane_y - lane_h // 2 + 8, hit_x, lane_y + lane_h // 2 - 8), fill=YELLOW, width=5)
    draw.ellipse((hit_x - 58, lane_y - 58, hit_x + 58, lane_y + 58), outline=HIT_RING, width=7)
    draw.ellipse((hit_x - 34, lane_y - 34, hit_x + 34, lane_y + 34), outline=(140, 128, 96), width=3)

    # Beat/onset decoration near the lane floor.
    onset_norm = onset - float(onset.min())
    onset_norm = onset_norm / max(float(onset_norm.max()), 1e-6)
    current_frame = t * 1000.0 / frame_ms
    for offset in range(-40, 220):
        idx = int(current_frame + offset)
        if idx < 0 or idx >= onset_norm.shape[0]:
            continue
        x = hit_x + (offset * frame_ms / 1000.0) / approach_sec * (spawn_x - hit_x)
        if 96 <= x <= 1538 and onset_norm[idx] > 0.5:
            h = 10 + int(onset_norm[idx] * 28)
            draw.line((x, lane_y + lane_h // 2 - h, x, lane_y + lane_h // 2 - 8), fill=(77, 146, 116), width=2)

    # Notes.
    visible_notes: list[tuple[float, bool]] = []
    for note_time in don_times:
        dt = float(note_time - t)
        if -0.12 <= dt <= approach_sec:
            visible_notes.append((note_time, False))
    for note_time in ka_times:
        dt = float(note_time - t)
        if -0.12 <= dt <= approach_sec:
            visible_notes.append((note_time, True))
    visible_notes.sort(key=lambda item: item[0], reverse=True)

    for note_time, is_ka in visible_notes:
        dt = float(note_time - t)
        x = hit_x + dt / approach_sec * (spawn_x - hit_x)
        if dt < 0:
            scale = max(0.15, 1.0 + dt / 0.12)
            color = KA if is_ka else DON
            draw_note(draw, x, lane_y, color, max(8, int(note_radius * scale)))
        else:
            draw_note(draw, x, lane_y, KA if is_ka else DON, note_radius)

    # Hit flash.
    near_hits = np.concatenate([don_times, ka_times])
    if near_hits.size:
        nearest = float(np.min(np.abs(near_hits - t)))
        if nearest < 0.055:
            r = int(78 + (0.055 - nearest) / 0.055 * 50)
            draw.ellipse((hit_x - r, lane_y - r, hit_x + r, lane_y + r), outline=(255, 232, 124), width=5)

    # Header/status.
    shown_title = title if title else "Generated Taiko Chart"
    draw.text((72, 54), shown_title, font=title_font, fill=WHITE)
    draw.text((74, 104), "Taiko Diffusion v9 generated play preview", font=small_font, fill=MUTED)
    draw.text((1220, 62), f"{t:05.2f}s / {duration_sec:05.2f}s", font=count_font, fill=WHITE)

    combo = int((np.concatenate([don_times, ka_times]) <= t).sum()) if near_hits.size else 0
    total = int(don_times.size + ka_times.size)
    draw.text((90, 660), f"COMBO {combo:03d} / {total:03d}", font=count_font, fill=WHITE)
    draw.text((90, 706), f"DON {don_times.size:03d}   KA {ka_times.size:03d}", font=small_font, fill=MUTED)

    # Progress bar.
    bar_x0, bar_y0, bar_x1, bar_y1 = 90, 812, 1510, 832
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=10, fill=(44, 48, 58))
    progress = min(1.0, max(0.0, t / max(duration_sec, 1e-6)))
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * progress), bar_y1), radius=10, fill=YELLOW)

    image.save(path)


def run_ffmpeg(frames_dir: Path, output: Path, fps: int, audio_row: dict[str, str], duration_sec: float, frame_ms: float) -> None:
    start_frame = int(audio_row.get("start_frame", 0))
    offset_seconds = float(audio_row.get("offset_seconds", 0.0))
    audio_start = max(0.0, start_frame * frame_ms / 1000.0 - offset_seconds)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-ss",
        f"{audio_start:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        str(local_audio_path(audio_row["audio_path"])),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Taiko gameplay-style MP4 from a generated sample.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--approach-sec", type=float, default=2.15)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=False)
    note, don, ka, audio_row, onset = masks_from_sample(sample, args.audio_split, float(args.frame_ms), float(args.onset_mix))
    frames = note.shape[0]
    duration_sec = frames * float(args.frame_ms) / 1000.0
    total_video_frames = max(1, int(math.ceil(duration_sec * int(args.fps))))
    frame_times = np.arange(frames, dtype=np.float32) * float(args.frame_ms) / 1000.0
    don_times = frame_times[don]
    ka_times = frame_times[ka]
    title = str(sample["source_title"][0]) if "source_title" in sample.files else str(sample["source_chunk_id"][0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output.with_suffix("").parent / f"{args.output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for index in range(total_video_frames):
        draw_frame(
            frames_dir / f"{index:05d}.png",
            index,
            total_video_frames,
            don_times,
            ka_times,
            onset,
            title,
            duration_sec,
            int(args.fps),
            float(args.approach_sec),
            float(args.frame_ms),
        )
        if index == 0 or (index + 1) % 120 == 0 or index + 1 == total_video_frames:
            print(json.dumps({"frame": index + 1, "total": total_video_frames}, ensure_ascii=False), flush=True)

    run_ffmpeg(frames_dir, args.output, int(args.fps), audio_row, duration_sec, float(args.frame_ms))
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(
        json.dumps(
            {
                "sample": str(args.sample),
                "output": str(args.output),
                "title": title,
                "duration_sec": duration_sec,
                "notes": int(note.sum()),
                "don": int(don.sum()),
                "ka": int(ka.sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
