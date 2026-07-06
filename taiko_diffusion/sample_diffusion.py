from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.train_diffusion import diffusion_schedule, make_model


def read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return next(csv.DictReader(file))


def read_selected_row(path: Path, row_index: int = 0, chunk_id: str | None = None) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if chunk_id is not None:
        for row in rows:
            if row["chunk_id"] == chunk_id:
                return row
        raise KeyError(f"chunk_id not found in {path}: {chunk_id}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index out of range for {path}: {row_index}")
    return rows[row_index]


def load_condition_from_row(row: dict[str, str], stats: dict) -> tuple[np.ndarray, dict[str, float]]:
    data = np.load(row["npz_path"], allow_pickle=False)
    raw = data["condition"].astype(np.float32)
    mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    std = np.asarray(stats["condition_std"], dtype=np.float32)
    normalized = (raw - mean) / std
    names = [str(name) for name in stats["condition_names"]]
    return normalized.astype(np.float32), {name: float(raw[index]) for index, name in enumerate(names)}


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def load_audio_from_row(row: dict[str, str], audio_csv: Path, audio_stats_path: Path) -> np.ndarray:
    audio_rows = read_rows_by_chunk(audio_csv)
    audio_row = audio_rows[row["chunk_id"]]
    audio_stats = json.loads(audio_stats_path.read_text(encoding="utf-8"))
    mean = np.asarray(audio_stats["feature_mean"], dtype=np.float32)
    std = np.asarray(audio_stats["feature_std"], dtype=np.float32)
    data = np.load(audio_row["audio_npz_path"], allow_pickle=False)
    audio = data["audio"].astype(np.float32)
    audio = (audio - mean) / std
    return audio.transpose(1, 0).astype(np.float32)


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    chart_channels: int,
    window_frames: int,
    device: torch.device,
    audio: torch.Tensor | None = None,
) -> torch.Tensor:
    x = torch.randn((1, chart_channels, window_frames), device=device)
    for step in range(timesteps - 1, -1, -1):
        t = torch.full((1,), step, dtype=torch.long, device=device)
        pred_noise = model(x, t, condition, audio)
        beta = schedule["betas"][step]
        alpha = schedule["alphas"][step]
        alpha_bar = schedule["alpha_bar"][step]
        mean = (x - beta * pred_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one chart window from a chart-only diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/diffusion_v0/best.pt"))
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v0/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v0/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=None)
    parser.add_argument("--audio-stats", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/diffusion_v0_sample.npz"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--chunk-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    row = read_selected_row(args.split, int(args.row_index), args.chunk_id)
    condition_np, raw_condition = load_condition_from_row(row, stats)
    audio_tensor = None
    audio_np = None
    if args.audio_split is not None or int(config.get("model", {}).get("audio_channels", 0)) > 0:
        if args.audio_split is None or args.audio_stats is None:
            raise ValueError("--audio-split and --audio-stats are required for an audio-conditioned checkpoint")
        audio_np = load_audio_from_row(row, args.audio_split, args.audio_stats)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
    if audio_np is not None:
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    window_frames = int(stats["window_frames"])
    chart_channels = len(stats["target_channels"])
    generated = sample(
        model,
        condition,
        schedule,
        int(config["diffusion"]["timesteps"]),
        chart_channels,
        window_frames,
        device,
        audio_tensor,
    )
    probability = ((generated.clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).transpose(0, 1).cpu().numpy()
    binary = (probability >= float(args.threshold)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probability=probability,
        binary=binary,
        target_channels=np.asarray(stats["target_channels"]),
        condition_names=np.asarray(stats["condition_names"]),
        raw_condition=np.asarray([raw_condition[name] for name in stats["condition_names"]], dtype=np.float32),
        audio=audio_np if audio_np is not None else np.asarray([], dtype=np.float32),
        source_chunk_id=np.asarray([row["chunk_id"]]),
        source_sample_id=np.asarray([row["sample_id"]]),
        source_title=np.asarray([row.get("title", "")]),
    )
    channel_sums = {
        str(name): int(binary[:, index].sum()) for index, name in enumerate(stats["target_channels"])
    }
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
                "source_chunk_id": row["chunk_id"],
                "source_title": row.get("title", ""),
                "threshold": float(args.threshold),
                "binary_channel_sums": channel_sums,
                "raw_condition": raw_condition,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
