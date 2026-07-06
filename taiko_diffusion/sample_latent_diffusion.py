from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.models.latent_diffusion import ChartAutoencoder1D, encode_chart_latent
from taiko_diffusion.sample_diffusion import load_audio_from_row, load_condition_from_row, read_selected_row
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


@torch.no_grad()
def predict_guided(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    audio: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x, t, condition, audio)
    pred_uncond = model(x, t, torch.zeros_like(condition), torch.zeros_like(audio))
    pred_cond = model(x, t, condition, audio)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    audio: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    sample_steps: int,
    latent_shape: tuple[int, int],
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((1, latent_shape[0], latent_shape[1]), device=device)
    steps = np.linspace(timesteps - 1, 0, num=min(sample_steps, timesteps), dtype=np.int64)
    steps = np.unique(steps)[::-1]
    if steps[0] != timesteps - 1:
        steps = np.concatenate([[timesteps - 1], steps])
    if steps[-1] != 0:
        steps = np.concatenate([steps, [0]])

    for index, step in enumerate(steps):
        t = torch.full((1,), int(step), dtype=torch.long, device=device)
        pred_noise = predict_guided(model, x, t, condition, audio, guidance_scale)
        alpha_bar = schedule["alpha_bar"][int(step)]
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_om = torch.sqrt(1.0 - alpha_bar)
        pred_x0 = (x - sqrt_om * pred_noise) / torch.clamp(sqrt_ab, min=1e-6)
        if index + 1 == len(steps):
            x = pred_x0
            break
        next_step = int(steps[index + 1])
        alpha_bar_next = schedule["alpha_bar"][next_step]
        x = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
    return x


def infer_latent_shape(autoencoder: ChartAutoencoder1D, chart_channels: int, window_frames: int, device: torch.device) -> tuple[int, int]:
    dummy = torch.zeros((1, chart_channels, window_frames), device=device)
    with torch.no_grad():
        latent = encode_chart_latent(autoencoder, dummy, sample_posterior=False)
    return int(latent.shape[1]), int(latent.shape[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a chart window from latent Taiko diffusion.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/latent_diffusion_v7_sample.npz"))
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--chunk-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    row = read_selected_row(args.split, int(args.row_index), args.chunk_id)
    condition_np, raw_condition = load_condition_from_row(row, stats)
    audio_np = load_audio_from_row(row, args.audio_split, args.audio_stats)
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
    latent_shape = infer_latent_shape(
        autoencoder,
        len(stats["target_channels"]),
        int(stats["window_frames"]),
        device,
    )
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
    logits = autoencoder.decode(latent)
    probability = torch.sigmoid(logits).squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probability=probability,
        binary=(probability >= 0.5).astype(np.float32),
        target_channels=np.asarray(stats["target_channels"]),
        condition_names=np.asarray(stats["condition_names"]),
        raw_condition=np.asarray([raw_condition[name] for name in stats["condition_names"]], dtype=np.float32),
        audio=audio_np,
        source_chunk_id=np.asarray([row["chunk_id"]]),
        source_sample_id=np.asarray([row["sample_id"]]),
        source_title=np.asarray([row.get("title", "")]),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
                "source_chunk_id": row["chunk_id"],
                "sample_steps": int(args.sample_steps),
                "guidance_scale": float(args.guidance_scale),
                "probability_shape": list(probability.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
