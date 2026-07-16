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
from taiko_diffusion.export_sample_tja import balloon_span_indices, density_topk_binary, hold_spans


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
    condition = {name: float(values[index]) for index, name in enumerate(names)}
    if "avg_density" not in condition and "decode_avg_density" in sample.files:
        condition["avg_density"] = float(sample["decode_avg_density"][0])
    return condition


def masks_from_sample(
    sample: np.lib.npyio.NpzFile,
    audio_split: Path | None,
    audio_path: Path | None,
    frame_ms: float,
    onset_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, bool]], dict[str, str], np.ndarray]:
    if audio_path is not None:
        if "audio" not in sample.files:
            raise ValueError("Sample has no audio features for direct-audio rendering")
        onset = sample["audio"].astype(np.float32)[:, -2]
        audio_row = {"audio_path": str(audio_path), "start_frame": "0", "offset_seconds": "0"}
    else:
        if audio_split is None or "source_chunk_id" not in sample.files:
            raise ValueError("--audio is required for samples without source_chunk_id")
        chunk_id = str(sample["source_chunk_id"][0])
        audio_rows = read_rows_by_chunk(audio_split)
        audio_row = audio_rows[chunk_id]
        audio_data = np.load(local_path(audio_row["audio_npz_path"]), allow_pickle=False)
        onset = audio_data["audio"].astype(np.float32)[:, -2]

    probability = sample["probability"].astype(np.float32)
    channel_names = [str(name) for name in sample["target_channels"]]
    binary = density_topk_binary(probability, channel_names, sample, frame_ms, None, onset_mix)
    channel = {name: index for index, name in enumerate(channel_names)}
    note = np.zeros(probability.shape[0], dtype=bool)
    for name in ["don", "ka", "big_don", "big_ka"]:
        if name in channel:
            note |= binary[:, channel[name]] > 0.5
    ka = np.zeros_like(note)
    for name in ["ka", "big_ka"]:
        if name in channel:
            ka |= binary[:, channel[name]] > 0.5
    big = np.zeros_like(note)
    for name in ["big_don", "big_ka"]:
        if name in channel:
            big |= binary[:, channel[name]] > 0.5
    don = note & ~ka
    spans = []
    unified_spans = hold_spans(binary, channel_names)
    balloon_indices = balloon_span_indices(
        len(unified_spans), condition_map(sample).get("balloon_roll_ratio", 0.0)
    )
    spans.extend((start, end, index in balloon_indices) for index, (start, end) in enumerate(unified_spans))
    for prefix, is_balloon in [("roll", False), ("balloon", True)]:
        if f"{prefix}_start" not in channel or f"{prefix}_end" not in channel:
            continue
        starts = np.flatnonzero(binary[:, channel[f"{prefix}_start"]] > 0.5)
        ends = np.flatnonzero(binary[:, channel[f"{prefix}_end"]] > 0.5)
        for start in starts:
            later = ends[ends > start]
            if later.size:
                spans.append((int(start), int(later[0]), is_balloon))
    return note, don, ka, big, binary, spans, audio_row, onset


def draw_note(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    shadow = (0, 0, 0)
    draw.ellipse((x - radius + 3, y - radius + 4, x + radius + 3, y + radius + 4), fill=shadow)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=4)
    inner = tuple(min(255, c + 28) for c in color)
    draw.ellipse((x - radius * 0.48, y - radius * 0.48, x + radius * 0.48, y + radius * 0.48), fill=inner)


def tempo_beats(times: np.ndarray, tempo_map: np.ndarray, frame_ms: float) -> np.ndarray:
    """Convert chart seconds to cumulative beats using the generated tempo map."""
    starts = tempo_map[:, 0].astype(np.float32) * frame_ms / 1000.0
    bpms = tempo_map[:, 1].astype(np.float32)
    base = np.zeros_like(starts)
    if starts.size > 1:
        base[1:] = np.cumsum((starts[1:] - starts[:-1]) * bpms[:-1] / 60.0)
    indices = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
    return base[indices] + (times - starts[indices]) * bpms[indices] / 60.0


def tempo_bpm(times: np.ndarray, tempo_map: np.ndarray, frame_ms: float) -> np.ndarray:
    starts = tempo_map[:, 0].astype(np.float32) * frame_ms / 1000.0
    indices = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
    return tempo_map[:, 1].astype(np.float32)[indices]


def draw_frame(
    path: Path,
    frame_index: int,
    total_frames: int,
    don_beats: np.ndarray,
    ka_beats: np.ndarray,
    big_beats: np.ndarray,
    spans: list[tuple[float, float, bool]],
    onset: np.ndarray,
    onset_beats: np.ndarray,
    title: str,
    duration_sec: float,
    fps: int,
    approach_beats: float,
    current_beat: float,
    current_bpm: float,
    frame_ms: float,
    condition_text: str,
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
        x = hit_x + (float(onset_beats[idx]) - current_beat) / approach_beats * (spawn_x - hit_x)
        if 96 <= x <= 1538 and onset_norm[idx] > 0.5:
            h = 10 + int(onset_norm[idx] * 28)
            draw.line((x, lane_y + lane_h // 2 - h, x, lane_y + lane_h // 2 - 8), fill=(77, 146, 116), width=2)

    # Notes.
    visible_notes: list[tuple[float, bool]] = []
    for note_beat in don_beats:
        delta = float(note_beat - current_beat)
        if -0.3 <= delta <= approach_beats:
            visible_notes.append((note_beat, False))
    for note_beat in ka_beats:
        delta = float(note_beat - current_beat)
        if -0.3 <= delta <= approach_beats:
            visible_notes.append((note_beat, True))
    visible_notes.sort(key=lambda item: item[0], reverse=True)

    for span_start, span_end, is_balloon in spans:
        if span_end < current_beat - 0.3 or span_start > current_beat + approach_beats:
            continue
        start_x = hit_x + (span_start - current_beat) / approach_beats * (spawn_x - hit_x)
        end_x = hit_x + (span_end - current_beat) / approach_beats * (spawn_x - hit_x)
        left, right = sorted((max(hit_x, start_x), min(spawn_x, end_x)))
        if right > left:
            color = (244, 176, 55) if not is_balloon else (234, 112, 55)
            draw.rounded_rectangle((left, lane_y - 24, right, lane_y + 24), radius=20, fill=color, outline=WHITE, width=3)

    for note_beat, is_ka in visible_notes:
        delta = float(note_beat - current_beat)
        x = hit_x + delta / approach_beats * (spawn_x - hit_x)
        is_big = bool(np.any(np.isclose(big_beats, note_beat, atol=1e-5)))
        radius = 52 if is_big else note_radius
        if delta < 0:
            scale = max(0.15, 1.0 + delta / 0.3)
            color = KA if is_ka else DON
            draw_note(draw, x, lane_y, color, max(8, int(radius * scale)))
        else:
            draw_note(draw, x, lane_y, KA if is_ka else DON, radius)

    # Hit flash.
    near_hits = np.concatenate([don_beats, ka_beats])
    if near_hits.size:
        nearest = float(np.min(np.abs(near_hits - current_beat)))
        if nearest < 0.15:
            r = int(78 + (0.15 - nearest) / 0.15 * 50)
            draw.ellipse((hit_x - r, lane_y - r, hit_x + r, lane_y + r), outline=(255, 232, 124), width=5)

    # Header/status.
    shown_title = title if title else "Generated Taiko Chart"
    draw.text((72, 54), shown_title, font=title_font, fill=WHITE)
    draw.text((74, 104), "Taiko Diffusion generated play preview", font=small_font, fill=MUTED)
    draw.text((74, 142), condition_text, font=small_font, fill=(190, 198, 210))
    draw.text((1170, 62), f"BPM {current_bpm:06.2f}  {t:05.2f}s / {duration_sec:05.2f}s", font=count_font, fill=WHITE)

    combo = int((np.concatenate([don_beats, ka_beats]) <= current_beat).sum()) if near_hits.size else 0
    total = int(don_beats.size + ka_beats.size)
    draw.text((90, 660), f"COMBO {combo:03d} / {total:03d}", font=count_font, fill=WHITE)
    draw.text((90, 706), f"DON {don_beats.size:03d}   KA {ka_beats.size:03d}", font=small_font, fill=MUTED)

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
    parser.add_argument("--audio", type=Path, default=None, help="Original audio for a standalone one-click sample.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--approach-sec", type=float, default=None)
    parser.add_argument("--sixteenth-spacing", type=float, default=80.0)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=False)
    note, don, ka, big, _, span_frames, audio_row, onset = masks_from_sample(sample, args.audio_split, args.audio, float(args.frame_ms), float(args.onset_mix))
    active_frames = np.flatnonzero(sample["legal_mask"] > 0.5) if "legal_mask" in sample.files else np.arange(note.shape[0])
    frames = int(active_frames[-1]) + 1 if active_frames.size else note.shape[0]
    note, don, ka, big, onset = (values[:frames] for values in (note, don, ka, big, onset))
    duration_sec = frames * float(args.frame_ms) / 1000.0
    total_video_frames = max(1, int(math.ceil(duration_sec * int(args.fps))))
    frame_times = np.arange(frames, dtype=np.float32) * float(args.frame_ms) / 1000.0
    bpm = float(sample["bpm_track"][0]) if "bpm_track" in sample.files and sample["bpm_track"].size else 120.0
    tempo_map = sample["tempo_map"].astype(np.float32) if "tempo_map" in sample.files else np.asarray([[0.0, bpm]], dtype=np.float32)
    don_beats = tempo_beats(frame_times[don], tempo_map, float(args.frame_ms))
    ka_beats = tempo_beats(frame_times[ka], tempo_map, float(args.frame_ms))
    big_beats = tempo_beats(frame_times[big], tempo_map, float(args.frame_ms))
    spans = [
        (float(tempo_beats(np.asarray([frame_times[start]]), tempo_map, float(args.frame_ms))[0]), float(tempo_beats(np.asarray([frame_times[end]]), tempo_map, float(args.frame_ms))[0]), is_balloon)
        for start, end, is_balloon in span_frames
    ]
    onset_beats = tempo_beats(frame_times, tempo_map, float(args.frame_ms))
    title = str(sample["source_title"][0]) if "source_title" in sample.files else str(sample["source_chunk_id"][0])
    lane_distance = 1510.0 - 260.0
    approach_beats = float(args.approach_sec) * bpm / 60.0 if args.approach_sec is not None else lane_distance / float(args.sixteenth_spacing) / 4.0
    conditions = condition_map(sample)
    condition_text = (
        f"const {conditions.get('const', 0):.1f} | complex {conditions.get('complex_bin', 0):.0f} | "
        f"note {conditions.get('note_type_bin', 0):.0f} | density {conditions.get('avg_density_bin', 0):.0f} | "
        f"peak {conditions.get('peak_density_bin', 0):.0f} | ka {conditions.get('ka_ratio', 0):.2f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output.with_suffix("").parent / f"{args.output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for index in range(total_video_frames):
        current_time = index / int(args.fps)
        current_beat = float(tempo_beats(np.asarray([current_time]), tempo_map, float(args.frame_ms))[0])
        current_bpm = float(tempo_bpm(np.asarray([current_time]), tempo_map, float(args.frame_ms))[0])
        draw_frame(
            frames_dir / f"{index:05d}.png",
            index,
            total_video_frames,
            don_beats,
            ka_beats,
            big_beats,
            spans,
            onset,
            onset_beats,
            title,
            duration_sec,
            int(args.fps),
            approach_beats,
            current_beat,
            current_bpm,
            float(args.frame_ms),
            condition_text,
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
                "bpm": bpm,
                "tempo_map": tempo_map.tolist(),
                "approach_beats": approach_beats,
                "sixteenth_spacing_px": float(args.sixteenth_spacing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
