from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.eval_audio_alignment import generated_note_mask, summarize
from taiko_diffusion.sample_diffusion import load_audio_from_row, load_condition_from_row, read_first_row
from taiko_diffusion.sample_latent_diffusion import ddim_sample, infer_latent_shape, set_seed
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample latent diffusion with several seeds and evaluate raw-onset alignment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    row = read_first_row(args.split)
    chunk_id = row["chunk_id"]
    condition_np, raw_condition = load_condition_from_row(row, stats)
    audio_np = load_audio_from_row(row, args.audio_split, args.audio_stats)
    raw_audio = np.load(read_rows_by_chunk(args.audio_split)[chunk_id]["audio_npz_path"], allow_pickle=False)["audio"].astype(np.float32)
    onset = raw_audio[:, -2]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    latent_stats_path = config["autoencoder"].get("latent_stats")
    latent_mean, latent_std = load_latent_stats(Path(latent_stats_path), device) if latent_stats_path else (None, None)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
    audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    latent_shape = infer_latent_shape(autoencoder, len(stats["target_channels"]), int(stats["window_frames"]), device)

    chart_rows = read_rows_by_chunk(args.split)
    target = np.load(chart_rows[chunk_id]["npz_path"], allow_pickle=False)["chart"].astype(np.float32)
    target_channels = [str(name) for name in stats["target_channels"]]
    if "note_event" in target_channels:
        target_mask = target[:, target_channels.index("note_event")] > 0.5
    elif "don" in target_channels and "ka" in target_channels:
        target_mask = (target[:, target_channels.index("don")] > 0.5) | (target[:, target_channels.index("ka")] > 0.5)
    else:
        target_mask = target[:, 0] > 0.5
    seed_values = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    records = []
    for seed in seed_values:
        set_seed(seed)
        latent = ddim_sample(
            model,
            condition,
            audio,
            schedule,
            int(config["diffusion"]["timesteps"]),
            int(args.sample_steps),
            latent_shape,
            float(args.guidance_scale),
            device,
        )
        if latent_mean is not None and latent_std is not None:
            latent = latent * latent_std + latent_mean
        probability = torch.sigmoid(autoencoder.decode(latent)).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
        mask = generated_note_mask(
            probability,
            raw_condition,
            float(stats["frame_ms"]),
            onset=onset,
            onset_mix=float(args.onset_mix),
            channel_names=target_channels,
        )
        records.append({"seed": seed, **summarize(mask, onset)})

    mean_generated = {
        key: float(np.mean([record[key] for record in records]))
        for key in ["notes", "onset_mean_at_notes", "onset_top25_hit_rate"]
    }
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "chunk_id": chunk_id,
                "sample_steps": int(args.sample_steps),
                "guidance_scale": float(args.guidance_scale),
                "onset_mix": float(args.onset_mix),
                "records": records,
                "mean_generated": mean_generated,
                "target": summarize(target_mask, onset),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
