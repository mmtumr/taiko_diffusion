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


def legal_grid(window_frames: int, subdivision_bin: int, bpm: float, frame_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(window_frames, dtype=np.float32)
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
    measure_frames = max(1, int(round(4.0 * 60000.0 / bpm / frame_ms)))
    legal.fill(0.0)
    for measure, start in enumerate(range(0, window_frames, measure_frames)):
        end = min(start + measure_frames, window_frames)
        for slot in slots:
            frame = min(end - 1, start + int(round(slot * (end - start) / 96.0)))
            legal[frame] = 1.0
            measure_indices[frame] = measure
            slot_indices[frame] = slot
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
    parser.add_argument("--bpm", type=float, default=None, help="Override onset-estimated static BPM.")
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
    bpm = float(args.bpm) if args.bpm is not None else estimate_bpm(features[:, -2], frame_ms)
    legal_mask, measure_indices, slot_indices = legal_grid(window_frames, subdivision_bin, bpm, frame_ms)
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
    print(json.dumps({"audio": str(args.audio), "output": str(args.output), "sample": str(sample_path), "duration_sec": duration_frames * frame_ms / 1000.0, "estimated_bpm": bpm, "truncated": features.shape[0] > window_frames, "neural_hold_spans": len(spans), "condition": condition_map}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
