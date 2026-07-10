from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.sample_latent_diffusion import infer_latent_shape, set_seed
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


@torch.no_grad()
def sample_variants(
    model: torch.nn.Module,
    condition: torch.Tensor,
    audio: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    sample_steps: int,
    initial_noise: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    batch_size = condition.shape[0]
    x = initial_noise.expand(batch_size, -1, -1).clone()
    steps = np.unique(np.linspace(timesteps - 1, 0, min(sample_steps, timesteps), dtype=np.int64))[::-1]
    for index, step in enumerate(steps):
        t = torch.full((batch_size,), int(step), dtype=torch.long, device=x.device)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        condition_in = torch.cat([torch.zeros_like(condition), condition], dim=0)
        audio_in = torch.cat([torch.zeros_like(audio), audio], dim=0)
        pred_uncond, pred_cond = model(x_in, t_in, condition_in, audio_in).chunk(2, dim=0)
        pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        alpha_bar = schedule["alpha_bar"][int(step)]
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar).clamp_min(1e-6)
        if index + 1 == len(steps):
            return pred_x0
        next_alpha_bar = schedule["alpha_bar"][int(steps[index + 1])]
        x = torch.sqrt(next_alpha_bar) * pred_x0 + torch.sqrt(1.0 - next_alpha_bar) * pred_noise
    return x


def intervention_values(name: str, base: float, mean: float, std: float) -> tuple[float, float]:
    if name in {"complex_bin", "bpm_rhythm_bin", "note_type_bin", "avg_density_bin", "peak_density_bin"}:
        return 0.0, 2.0
    if name in {"hs_change_bin", "note_type_high"}:
        return 0.0, 1.0
    low = mean - std
    high = mean + std
    if name in {"const", "complex", "hs_change", "note_type", "avg_density", "peak_density"}:
        low = max(0.0, low)
    if name in {"big_note_ratio", "balloon_roll_ratio", "ka_ratio"}:
        low = max(0.0, low)
        high = min(1.0, high)
    return float(low), float(high)


def probability_metrics(
    probability: np.ndarray,
    fixed_note_count: int,
    legal_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    raw_note_score = probability.max(axis=0)
    note_score = raw_note_score.copy()
    valid = np.ones(note_score.shape[0], dtype=bool)
    if legal_mask is not None:
        valid = legal_mask > 0.5
        note_score = np.where(valid, note_score, -np.inf)
        fixed_note_count = min(fixed_note_count, int(valid.sum()))
    fixed_note_count = min(max(int(fixed_note_count), 1), note_score.size)
    selected = np.argpartition(note_score, -fixed_note_count)[-fixed_note_count:]
    selected_ka = probability[1, selected] > probability[0, selected]
    return {
        "don_probability_mean": float(probability[0].mean()),
        "ka_probability_mean": float(probability[1].mean()),
        "note_probability_mean": float(raw_note_score[valid].mean()) if valid.any() else 0.0,
        "threshold_note_count": int((raw_note_score[valid] >= 0.5).sum()),
        "fixed_count_ka_ratio": float(selected_ka.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure v9 response to isolated chart-condition changes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v9_donka/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v9_donka/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/condition_interventions_v9.json"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=str, default="const,complex,note_type,avg_density,peak_density,ka_ratio")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    names = [str(name) for name in stats["condition_names"]]
    selected_names = [name.strip() for name in args.conditions.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in names]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    latent_stats_path = config["autoencoder"].get("latent_stats")
    latent_mean, latent_std = load_latent_stats(Path(latent_stats_path), device) if latent_stats_path else (None, None)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    dataset = TaikoAudioDiffusionDataset(args.split, args.stats, args.audio_split, args.audio_stats)
    latent_shape = infer_latent_shape(autoencoder, len(stats["target_channels"]), int(stats["window_frames"]), device)
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)

    records = []
    for sample_index in range(min(args.samples, len(dataset))):
        item = dataset[sample_index]
        base_raw = item["condition_raw"].numpy().astype(np.float32)
        audio = item["audio"].unsqueeze(0).to(device)
        legal_masks = item.get("legal_masks")
        fixed_note_count = int((item["chart"].max(dim=0).values > 0.5).sum())
        initial_noise = torch.randn((1, latent_shape[0], latent_shape[1]), device=device)
        for name in selected_names:
            condition_index = names.index(name)
            low, high = intervention_values(
                name,
                float(base_raw[condition_index]),
                float(condition_mean[condition_index]),
                float(condition_std[condition_index]),
            )
            raw_variants = np.repeat(base_raw[None, :], 3, axis=0)
            raw_variants[:, condition_index] = [low, base_raw[condition_index], high]
            normalized = (raw_variants - condition_mean) / condition_std
            condition = torch.from_numpy(normalized).to(device)
            audio_batch = audio.expand(3, -1, -1)
            variant_legal_masks = None
            if legal_masks is not None:
                if "complex_bin" in names:
                    complex_index = names.index("complex_bin")
                    bins = np.rint(raw_variants[:, complex_index]).astype(np.int64).clip(0, 2)
                    variant_legal_masks = torch.stack([legal_masks[int(value)] for value in bins])
                else:
                    variant_legal_masks = legal_masks[-1].unsqueeze(0).expand(3, -1)
                if bool(config["model"].get("use_legal_mask_channel", False)):
                    audio_batch = torch.cat([audio_batch, variant_legal_masks.to(device).unsqueeze(1)], dim=1)
            latent = sample_variants(
                model,
                condition,
                audio_batch,
                schedule,
                int(config["diffusion"]["timesteps"]),
                int(args.sample_steps),
                initial_noise,
                float(args.guidance_scale),
            )
            if latent_mean is not None and latent_std is not None:
                latent = latent * latent_std + latent_mean
            probability = torch.sigmoid(autoencoder.decode(latent)).cpu().numpy()
            metrics = [
                probability_metrics(
                    value,
                    fixed_note_count,
                    variant_legal_masks[index].numpy() if variant_legal_masks is not None else None,
                )
                for index, value in enumerate(probability)
            ]
            records.append(
                {
                    "chunk_id": str(item["chunk_id"]),
                    "condition": name,
                    "values": {"low": low, "base": float(base_raw[condition_index]), "high": high},
                    "metrics": {"low": metrics[0], "base": metrics[1], "high": metrics[2]},
                    "low_high_probability_mae": float(np.abs(probability[2] - probability[0]).mean()),
                }
            )
        print(json.dumps({"sample": sample_index + 1, "total": min(args.samples, len(dataset))}), flush=True)

    summary: dict[str, dict[str, float]] = {}
    for name in selected_names:
        rows = [record for record in records if record["condition"] == name]
        summary[name] = {
            "low_high_probability_mae": float(np.mean([row["low_high_probability_mae"] for row in rows])),
            "note_probability_change": float(
                np.mean([row["metrics"]["high"]["note_probability_mean"] - row["metrics"]["low"]["note_probability_mean"] for row in rows])
            ),
            "threshold_note_count_change": float(
                np.mean([row["metrics"]["high"]["threshold_note_count"] - row["metrics"]["low"]["threshold_note_count"] for row in rows])
            ),
            "fixed_count_ka_ratio_change": float(
                np.mean([row["metrics"]["high"]["fixed_count_ka_ratio"] - row["metrics"]["low"]["fixed_count_ka_ratio"] for row in rows])
            ),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "samples": min(args.samples, len(dataset)),
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
