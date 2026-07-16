from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.data.audio import audio_features, decode_audio_ffmpeg
from taiko_diffusion.export_sample_tja import density_topk_binary
from taiko_diffusion.hold_span_inference import load_hold_span_model, predict_hold_spans, spans_to_hold_channels
from taiko_diffusion.sample_latent_diffusion import ddim_sample, infer_latent_shape, set_seed
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


DEFAULT_CONDITIONS = {
    "const": 7.0,
    "complex_bin": 1.0,
    "subdivision_bin": 1.0,
    "hs_change_bin": 0.0,
    "bpm_rhythm_bin": 0.0,
    "note_type_bin": 1.0,
    "avg_density_bin": 1.0,
    "peak_density_bin": 1.0,
    "big_note_ratio": 0.06,
    "balloon_roll_ratio": 0.2,
    "ka_ratio": 0.35,
}


def parse_condition_assignments(assignments: list[str]) -> dict[str, float]:
    values = dict(DEFAULT_CONDITIONS)
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or name not in values:
            raise ValueError(f"Invalid --set-condition assignment: {assignment}")
        values[name] = float(value)
    return values


def estimate_bpm(onset: np.ndarray, frame_ms: float) -> float:
    centered = onset.astype(np.float32) - float(np.mean(onset))
    if float(np.std(centered)) < 1e-6:
        return 120.0
    minimum_lag = max(1, int(round(60000.0 / 240.0 / frame_ms)))
    maximum_lag = max(minimum_lag, int(round(60000.0 / 60.0 / frame_ms)))
    scores = np.asarray(
        [float(np.dot(centered[:-lag], centered[lag:])) for lag in range(minimum_lag, maximum_lag + 1)],
        dtype=np.float32,
    )
    lag = minimum_lag + int(np.argmax(scores))
    return float(np.clip(60000.0 / (lag * frame_ms), 60.0, 240.0))


def tempo_evidence(onset: np.ndarray, frame_ms: float) -> float:
    centered = onset.astype(np.float32) - float(np.mean(onset))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-6:
        return 0.0
    minimum_lag = max(1, int(round(60_000.0 / 240.0 / frame_ms)))
    maximum_lag = max(minimum_lag, int(round(60_000.0 / 60.0 / frame_ms)))
    return float(
        np.clip(
            max(float(np.dot(centered[:-lag], centered[lag:])) / energy for lag in range(minimum_lag, maximum_lag + 1)),
            0.0,
            1.0,
        )
    )


def estimate_tempo_map(onset: np.ndarray, frame_ms: float) -> tuple[np.ndarray, float]:
    """Estimate stable tempo sections with multi-candidate continuity decoding."""
    window_frames = max(8, int(round(8_000.0 / frame_ms)))
    hop_frames = max(1, int(round(4_000.0 / frame_ms)))
    if onset.shape[0] < window_frames * 2:
        return np.asarray([[0.0, estimate_bpm(onset, frame_ms)]], dtype=np.float32), 0.0
    starts = list(range(0, onset.shape[0] - window_frames + 1, hop_frames))
    local_onsets = [onset[start : start + window_frames] for start in starts]
    minimum_lag = max(1, int(round(60_000.0 / 240.0 / frame_ms)))
    maximum_lag = max(minimum_lag, int(round(60_000.0 / 60.0 / frame_ms)))
    candidates: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for values in local_onsets:
        centered = values.astype(np.float32) - float(np.mean(values))
        energy = max(float(np.dot(centered, centered)), 1e-6)
        correlation = np.asarray(
            [float(np.dot(centered[:-lag], centered[lag:])) / energy for lag in range(minimum_lag, maximum_lag + 1)],
            dtype=np.float32,
        )
        selected: list[int] = []
        for index in np.argsort(correlation)[::-1]:
            if all(abs(int(index) - other) > 2 for other in selected):
                selected.append(int(index))
            if len(selected) == 5:
                break
        candidate_lags = np.asarray([minimum_lag + index for index in selected], dtype=np.float32)
        candidates.append(60_000.0 / (candidate_lags * frame_ms))
        scores.append(correlation[np.asarray(selected, dtype=np.int32)])

    # Decode a smooth trajectory across the local BPM candidates. This prevents
    # the strongest local half-tempo peak from replacing a consistent beat track.
    path_scores = scores[0].copy()
    backpointers: list[np.ndarray] = []
    for index in range(1, len(candidates)):
        previous = candidates[index - 1]
        current = candidates[index]
        transition = 0.30 * np.abs(np.log(current[:, None] / previous[None, :]))
        choices = np.argmax(path_scores[None, :] - transition, axis=1)
        path_scores = scores[index] + path_scores[choices] - transition[np.arange(len(current)), choices]
        backpointers.append(choices)
    path = [int(np.argmax(path_scores))]
    for choices in reversed(backpointers):
        path.append(int(choices[path[-1]]))
    path.reverse()
    local_bpms = np.asarray([candidates[index][choice] for index, choice in enumerate(path)], dtype=np.float32)
    evidence = float(np.median([tempo_evidence(values, frame_ms) for values in local_onsets]))
    # A three-window median suppresses isolated half/double-tempo estimates.
    smoothed = np.asarray(
        [np.median(local_bpms[max(0, index - 1) : min(len(local_bpms), index + 2)]) for index in range(len(local_bpms))],
        dtype=np.float32,
    )
    relative_spread = float((np.percentile(smoothed, 90) - np.percentile(smoothed, 10)) / max(np.median(smoothed), 1e-6))
    if relative_spread < 0.03:
        return np.asarray([[0.0, float(np.median(smoothed))]], dtype=np.float32), evidence

    sections: list[tuple[int, float]] = [(0, float(smoothed[0]))]
    index = 1
    while index < len(smoothed):
        current_bpm = sections[-1][1]
        if abs(float(smoothed[index]) - current_bpm) / max(current_bpm, 1e-6) < 0.03:
            index += 1
            continue
        # A change must persist for two adjacent analysis windows.
        end = index + 1
        while end < len(smoothed) and abs(float(smoothed[end]) - float(smoothed[index])) / max(float(smoothed[index]), 1e-6) < 0.03:
            end += 1
        if end - index >= 2:
            sections.append((starts[index], float(np.median(smoothed[index:end]))))
            index = end
        else:
            index += 1

    if len(sections) == 1:
        return np.asarray(sections, dtype=np.float32), max(0.0, 1.0 - relative_spread / 0.25)
    confidence = evidence * min(1.0, relative_spread / 0.03)
    return np.asarray(sections, dtype=np.float32), confidence


def estimate_beat_tempo_map(audio_path: Path, duration_frames: int, frame_ms: float) -> tuple[np.ndarray, float] | None:
    """Return only sustained beat-tracker tempo changes from the original audio."""
    try:
        import essentia.standard as essentia
    except ImportError:
        return None
    audio = essentia.MonoLoader(filename=str(audio_path), sampleRate=44_100)()
    beats, confidence = essentia.BeatTrackerMultiFeature()(audio)
    if len(beats) < 8:
        return None
    beat_bpms = 60.0 / np.diff(np.asarray(beats, dtype=np.float32))
    window_seconds = 4.0
    windows: list[tuple[float, float]] = []
    for start in np.arange(0.0, duration_frames * frame_ms / 1000.0, window_seconds):
        selected = beat_bpms[(beats[:-1] >= start) & (beats[:-1] < start + window_seconds)]
        if selected.size >= 2:
            windows.append((float(start), float(np.median(selected))))
    if len(windows) < 3:
        return None
    sections: list[tuple[float, float]] = [(0.0, windows[0][1])]
    index = 1
    while index < len(windows):
        current_bpm = sections[-1][1]
        candidate_bpm = windows[index][1]
        if abs(candidate_bpm - current_bpm) / max(current_bpm, 1e-6) < 0.03:
            index += 1
            continue
        end = index + 1
        while end < len(windows) and abs(windows[end][1] - candidate_bpm) / max(candidate_bpm, 1e-6) < 0.03:
            end += 1
        # Require three 4-second windows, so short ramps and song tails remain static.
        if end - index >= 3:
            sections.append((windows[index][0] * 1000.0 / frame_ms, float(np.median([value for _, value in windows[index:end]]))))
            index = end
        else:
            index += 1
    coverage = len(windows) * window_seconds / max(duration_frames * frame_ms / 1000.0, 1e-6)
    return np.asarray(sections, dtype=np.float32), float(min(1.0, max(0.0, confidence) * coverage))


def bar_aligned_tempo_map(tempo_map: np.ndarray, frame_ms: float) -> np.ndarray:
    """Snap detected tempo changes to preceding whole-bar boundaries."""
    aligned: list[tuple[int, float]] = [(0, float(tempo_map[0, 1]))]
    cursor = 0
    bpm = aligned[0][1]
    for target, next_bpm in tempo_map[1:]:
        target_frame = int(round(float(target)))
        bar_frames = max(1, int(round(4.0 * 60_000.0 / bpm / frame_ms)))
        boundary = cursor + max(1, int(round((target_frame - cursor) / bar_frames))) * bar_frames
        if boundary <= cursor:
            continue
        aligned.append((boundary, float(next_bpm)))
        cursor = boundary
        bpm = float(next_bpm)
    return np.asarray(aligned, dtype=np.float32)


def legal_grid(
    window_frames: int,
    subdivision_bin: int,
    tempo_map: np.ndarray,
    frame_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legal = np.ones(window_frames, dtype=np.float32)
    measure_indices = np.full(window_frames, -1, dtype=np.int32)
    slot_indices = np.full(window_frames, -1, dtype=np.int32)
    if subdivision_bin >= 2:
        return legal, measure_indices, slot_indices
    slots = (
        list(range(0, 96, 6))
        if subdivision_bin == 0
        else sorted(set().union(*(range(0, 96, 96 // division) for division in [8, 12, 16, 24, 32])))
    )
    legal.fill(0.0)
    measure = 0
    for index, (start_value, bpm_value) in enumerate(tempo_map):
        start = int(round(float(start_value)))
        end = int(round(float(tempo_map[index + 1, 0]))) if index + 1 < len(tempo_map) else window_frames
        bar_frames = max(1, int(round(4.0 * 60_000.0 / float(bpm_value) / frame_ms)))
        for bar_start in range(start, min(end, window_frames), bar_frames):
            bar_end = min(bar_start + bar_frames, end, window_frames)
            for slot in slots:
                frame = min(bar_end - 1, bar_start + int(round(slot * (bar_end - bar_start) / 96.0)))
                legal[frame] = 1.0
                measure_indices[frame] = measure
                slot_indices[frame] = slot
            measure += 1
    return legal, measure_indices, slot_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a complete TJA from an arbitrary audio file.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latent_diffusion_v15_full_catalog/best.pt"))
    parser.add_argument("--hold-span-checkpoint", type=Path, default=Path("checkpoints/hold_span_v2_deploy/best.pt"))
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--hold-span-threshold", type=float, default=0.8)
    parser.add_argument("--bpm", type=float, default=None, help="Override audio tempo detection with a static BPM.")
    parser.add_argument("--set-condition", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {args.audio}")
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    cache_dir = Path(config["data"]["cache_dir"])
    stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8"))
    audio_stats = json.loads((Path(config["data"]["audio_cache_dir"]) / "stats.json").read_text(encoding="utf-8"))
    condition_names = [str(name) for name in stats["condition_names"]]
    if set(condition_names) != set(DEFAULT_CONDITIONS):
        raise ValueError("This one-click entry currently supports the v15 condition schema only")
    condition_map = parse_condition_assignments(args.set_condition)
    raw_condition = np.asarray([condition_map[name] for name in condition_names], dtype=np.float32)
    condition = (raw_condition - np.asarray(stats["condition_mean"], dtype=np.float32)) / np.asarray(stats["condition_std"], dtype=np.float32)
    frame_ms = float(stats["frame_ms"])
    window_frames = int(stats["window_frames"])
    sample_rate = int(audio_stats["sample_rate"])
    waveform = decode_audio_ffmpeg(args.audio, sample_rate)
    features = audio_features(
        waveform,
        sample_rate,
        frame_ms,
        int(audio_stats["n_fft"]),
        int(audio_stats["n_mels"]),
        30.0,
        float(audio_stats["sample_rate"]) / 2.0,
    )
    duration_frames = min(features.shape[0], window_frames)
    raw_audio = np.zeros((window_frames, features.shape[1]), dtype=np.float32)
    raw_audio[:duration_frames] = features[:duration_frames]
    normalized_audio = (raw_audio - np.asarray(audio_stats["feature_mean"], dtype=np.float32)) / np.asarray(audio_stats["feature_std"], dtype=np.float32)
    subdivision_bin = int(round(condition_map["subdivision_bin"]))
    if args.bpm is not None:
        tempo_map = np.asarray([[0.0, float(args.bpm)]], dtype=np.float32)
        tempo_confidence = 1.0
    elif condition_map["bpm_rhythm_bin"] > 0.0:
        beat_map = estimate_beat_tempo_map(args.audio, duration_frames, frame_ms)
        if beat_map is None:
            tempo_map, tempo_confidence = estimate_tempo_map(features[:, -2], frame_ms)
        else:
            tempo_map, tempo_confidence = beat_map
        if tempo_confidence >= 0.12:
            tempo_map = bar_aligned_tempo_map(tempo_map, frame_ms)
        else:
            tempo_map = np.asarray([[0.0, estimate_bpm(features[:, -2], frame_ms)]], dtype=np.float32)
    else:
        tempo_map = np.asarray([[0.0, estimate_bpm(features[:, -2], frame_ms)]], dtype=np.float32)
        tempo_confidence = 1.0
    bpm = float(tempo_map[0, 1])
    legal_mask, measure_indices, slot_indices = legal_grid(window_frames, subdivision_bin, tempo_map, frame_ms)
    active_mask = np.zeros(window_frames, dtype=np.float32)
    active_mask[:duration_frames] = 1.0
    legal_mask *= active_mask
    measure_indices[active_mask <= 0.5] = -1
    slot_indices[active_mask <= 0.5] = -1
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    latent_mean, latent_std = load_latent_stats(Path(config["autoencoder"]["latent_stats"]), device)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    audio_for_diffusion = np.concatenate([normalized_audio.T, legal_mask[None]], axis=0)
    latent = ddim_sample(
        model,
        torch.from_numpy(condition).unsqueeze(0).to(device),
        torch.from_numpy(audio_for_diffusion).unsqueeze(0).to(device),
        diffusion_schedule(config["diffusion"], device),
        int(config["diffusion"]["timesteps"]),
        int(args.sample_steps),
        infer_latent_shape(autoencoder, len(stats["target_channels"]), window_frames, device),
        float(args.guidance_scale),
        device,
    )
    latent = latent * latent_std + latent_mean
    probability = torch.sigmoid(autoencoder.decode(latent)).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    hold_model, hold_config, duration_log_scale = load_hold_span_model(args.hold_span_checkpoint, device)
    if hold_config["data"].get("context_channels"):
        raise ValueError("Hold span checkpoint requires unavailable chart context")
    spans = predict_hold_spans(
        hold_model,
        normalized_audio.T,
        condition,
        legal_mask,
        active_mask,
        duration_log_scale,
        float(args.hold_span_threshold),
        device,
    )
    channel = {name: index for index, name in enumerate(stats["target_channels"])}
    hold_channels = [channel[name] for name in ["hold_start", "hold_body", "hold_end"]]
    probability[:, hold_channels] = spans_to_hold_channels(window_frames, spans)
    sample_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        sample_path,
        probability=probability,
        target_channels=np.asarray(stats["target_channels"]),
        condition_names=np.asarray(condition_names),
        raw_condition=raw_condition,
        audio=raw_audio,
        legal_mask=legal_mask,
        measure_indices=measure_indices,
        slot_indices=slot_indices,
        bpm_track=np.asarray([bpm], dtype=np.float32),
        tempo_map=tempo_map,
        tempo_map_confidence=np.asarray([tempo_confidence], dtype=np.float32),
        duration_frames=np.asarray([duration_frames], dtype=np.int32),
        decode_avg_density=np.asarray([stats["bin_representatives"]["avg_density_bin"][int(condition_map["avg_density_bin"])]], dtype=np.float32),
        source_title=np.asarray([args.audio.stem]),
        neural_hold_spans=np.asarray(spans, dtype=np.float32),
    )
    command = ["export_sample_tja", "--sample", str(sample_path), "--output", str(args.output), "--density-topk", "--frame-ms", str(frame_ms)]
    from taiko_diffusion.export_sample_tja import main as export_main
    import sys

    old_argv = sys.argv
    try:
        sys.argv = command
        export_main()
    finally:
        sys.argv = old_argv
    print(json.dumps({"audio": str(args.audio), "output": str(args.output), "sample": str(sample_path), "duration_sec": duration_frames * frame_ms / 1000.0, "estimated_bpm": bpm, "tempo_map": tempo_map.tolist(), "tempo_map_confidence": tempo_confidence, "truncated": features.shape[0] > window_frames, "neural_hold_spans": len(spans), "condition": condition_map}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
