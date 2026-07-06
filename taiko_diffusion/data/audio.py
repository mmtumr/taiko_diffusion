from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

from taiko_diffusion.data.tja import read_raw_courses, select_course


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_offset_seconds(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def resolve_audio_path(tja_path: Path, course_name: str) -> tuple[Path, float]:
    raw_course = select_course(read_raw_courses(tja_path), course_name)
    wave = raw_course.meta.get("WAVE", "").strip()
    offset_seconds = parse_offset_seconds(raw_course.meta.get("OFFSET"))
    candidates: list[Path] = []
    if wave:
        candidates.append((tja_path.parent / wave).resolve())
    candidates.extend(sorted(tja_path.parent.glob("*.ogg")))
    candidates.extend(sorted(tja_path.parent.glob("*.mp3")))
    candidates.extend(sorted(tja_path.parent.glob("*.wav")))
    for candidate in candidates:
        if candidate.exists():
            return candidate, offset_seconds
    raise FileNotFoundError(f"Audio file not found for {tja_path} [{course_name}], WAVE={wave!r}")


def decode_audio_ffmpeg(path: Path, sample_rate: int) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise ValueError(f"Decoded audio is empty: {path}")
    return torch.from_numpy(samples).unsqueeze(0)


def audio_features(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    hop_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    waveform = waveform.float()
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    mel = mel_transform(waveform)
    mel_db = amp_to_db(mel).squeeze(0)
    mel_db = mel_db - mel_db.max()
    mel_norm = torch.clamp((mel_db + 80.0) / 80.0, 0.0, 1.0)

    onset = torch.relu(mel_norm[:, 1:] - mel_norm[:, :-1]).mean(dim=0)
    onset = torch.cat([torch.zeros(1, dtype=onset.dtype), onset], dim=0)
    onset_scale = torch.quantile(onset, 0.95).clamp_min(1e-6)
    onset = torch.clamp(onset / onset_scale, 0.0, 1.0)

    power = waveform.square()
    rms = torch.nn.functional.avg_pool1d(
        power,
        kernel_size=n_fft,
        stride=hop_length,
        padding=n_fft // 2,
        count_include_pad=False,
    ).sqrt().squeeze(0)
    if rms.shape[0] < mel_norm.shape[1]:
        rms = torch.nn.functional.pad(rms, (0, mel_norm.shape[1] - rms.shape[0]))
    rms = rms[: mel_norm.shape[1]]
    rms = torch.log1p(10.0 * rms) / np.log(11.0)
    rms = torch.clamp(rms, 0.0, 1.0)

    features = torch.cat([mel_norm, onset.unsqueeze(0), rms.unsqueeze(0)], dim=0)
    return features.transpose(0, 1).cpu().numpy().astype(np.float32)


def aligned_window(
    full_features: np.ndarray,
    start_frame: int,
    window_frames: int,
    offset_seconds: float,
    sample_rate: int,
    frame_ms: float,
) -> np.ndarray:
    hop_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    offset_frames = int(round((-offset_seconds * sample_rate) / hop_length))
    source_start = start_frame + offset_frames
    source_end = source_start + window_frames
    output = np.zeros((window_frames, full_features.shape[1]), dtype=np.float32)
    copy_start = max(source_start, 0)
    copy_end = min(source_end, full_features.shape[0])
    if copy_end <= copy_start:
        return output
    output_start = copy_start - source_start
    output_end = output_start + (copy_end - copy_start)
    output[output_start:output_end] = full_features[copy_start:copy_end]
    return output
