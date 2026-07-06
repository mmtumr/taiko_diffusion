from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from taiko_diffusion.data.diffusion_dataset import TaikoDiffusionDataset
from taiko_diffusion.models.latent_diffusion import encode_chart_latent
from taiko_diffusion.train_latent_diffusion import load_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-channel latent normalization stats for chart autoencoder.")
    parser.add_argument("--autoencoder", type=Path, default=Path("checkpoints/autoencoder_v7/best.pt"))
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/train.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/autoencoder_v7/latent_stats.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    autoencoder = load_autoencoder(args.autoencoder, device)
    dataset = TaikoDiffusionDataset(args.split, args.stats)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    total_count = 0
    total_sum = None
    total_square = None
    global_min = float("inf")
    global_max = float("-inf")
    with torch.no_grad():
        for batch in loader:
            chart = batch["chart"].to(device, non_blocking=True)
            latent = encode_chart_latent(autoencoder, chart, sample_posterior=False).cpu()
            flat = latent.permute(1, 0, 2).reshape(latent.shape[1], -1)
            if total_sum is None:
                total_sum = flat.sum(dim=1)
                total_square = (flat * flat).sum(dim=1)
            else:
                total_sum += flat.sum(dim=1)
                total_square += (flat * flat).sum(dim=1)
            total_count += flat.shape[1]
            global_min = min(global_min, float(latent.min()))
            global_max = max(global_max, float(latent.max()))
    assert total_sum is not None and total_square is not None
    mean = total_sum / max(total_count, 1)
    std = (total_square / max(total_count, 1) - mean * mean).clamp_min(1e-12).sqrt()
    stats = {
        "autoencoder": str(args.autoencoder),
        "split": str(args.split),
        "count": int(total_count),
        "mean": mean.numpy().astype(float).tolist(),
        "std": std.numpy().astype(float).tolist(),
        "global_min": global_min,
        "global_max": global_max,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
