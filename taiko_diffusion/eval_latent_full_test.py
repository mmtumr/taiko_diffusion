from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.eval_audio_alignment import generated_ka_mask, generated_note_mask, summarize
from taiko_diffusion.sample_latent_diffusion import ddim_sample, infer_latent_shape, set_seed
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def load_raw_onset(audio_npz_path: str) -> np.ndarray:
    return np.load(audio_npz_path, allow_pickle=False)["audio"].astype(np.float32)[:, -2]


@torch.no_grad()
def ddim_sample_batch(
    model: torch.nn.Module,
    condition: torch.Tensor,
    audio: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    sample_steps: int,
    latent_shape: tuple[int, int],
    guidance_scale: float,
    device: torch.device,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = condition.shape[0]
    x = (
        initial_noise.to(device)
        if initial_noise is not None
        else torch.randn((batch_size, latent_shape[0], latent_shape[1]), device=device)
    )
    steps = np.linspace(timesteps - 1, 0, num=min(sample_steps, timesteps), dtype=np.int64)
    steps = np.unique(steps)[::-1]
    if steps[0] != timesteps - 1:
        steps = np.concatenate([[timesteps - 1], steps])
    if steps[-1] != 0:
        steps = np.concatenate([steps, [0]])
    for index, step in enumerate(steps):
        t = torch.full((batch_size,), int(step), dtype=torch.long, device=device)
        if guidance_scale == 1.0:
            pred_noise = model(x, t, condition, audio)
        else:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            condition_in = torch.cat([torch.zeros_like(condition), condition], dim=0)
            audio_in = torch.cat([torch.zeros_like(audio), audio], dim=0)
            pred_uncond, pred_cond = model(x_in, t_in, condition_in, audio_in).chunk(2, dim=0)
            pred_noise = pred_uncond + float(guidance_scale) * (pred_cond - pred_uncond)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion checkpoint on the full audio test split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/full_test_latent_v8.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(int(args.seed))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    condition_names = [str(name) for name in stats["condition_names"]]
    target_channels = [str(name) for name in stats["target_channels"]]
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
    dataset = TaikoAudioDiffusionDataset(args.split, args.stats, args.audio_split, args.audio_stats)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    latent_shape = infer_latent_shape(
        autoencoder,
        len(stats["target_channels"]),
        int(stats["window_frames"]),
        device,
    )
    records = []
    start_time = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        condition = batch["condition"].to(device)
        audio = batch["audio"].to(device)
        legal_mask = batch.get("legal_mask")
        if bool(config["model"].get("use_legal_mask_channel", False)):
            if legal_mask is None:
                raise ValueError("Model requires legal_mask but the cache does not contain it")
            audio = torch.cat([audio, legal_mask.to(device).unsqueeze(1)], dim=1)
        latent = ddim_sample_batch(
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
        probability = torch.sigmoid(autoencoder.decode(latent)).transpose(1, 2).cpu().numpy()
        target = batch["chart"].transpose(1, 2).numpy()
        raw_condition = batch["condition_raw"].numpy()
        for index in range(probability.shape[0]):
            condition_map = {
                name: float(raw_condition[index, condition_index])
                for condition_index, name in enumerate(condition_names)
            }
            if "avg_density" not in condition_map and "avg_density_bin" in condition_map:
                representatives = stats.get("bin_representatives", {}).get("avg_density_bin")
                if representatives:
                    density_bin = int(round(condition_map["avg_density_bin"]))
                    condition_map["avg_density"] = float(representatives[max(0, min(density_bin, 2))])
            onset = load_raw_onset(str(batch["audio_npz_path"][index]))
            generated_mask = generated_note_mask(
                probability[index],
                condition_map,
                float(stats["frame_ms"]),
                onset=onset,
                onset_mix=float(args.onset_mix),
                channel_names=target_channels,
                legal_mask=legal_mask[index].numpy() if legal_mask is not None else None,
            )
            if "note_event" in target_channels:
                target_note = target[index, :, target_channels.index("note_event")] > 0.5
            elif "don" in target_channels and "ka" in target_channels:
                target_note = (target[index, :, target_channels.index("don")] > 0.5) | (
                    target[index, :, target_channels.index("ka")] > 0.5
                )
            else:
                target_note = target[index, :, 0] > 0.5
            if "note_event" in target_channels and "ka_probability" in target_channels:
                target_ka = target[index, :, target_channels.index("ka_probability")] > 0.5
            elif "ka" in target_channels:
                target_ka = target[index, :, target_channels.index("ka")] > 0.5
            else:
                target_ka = np.zeros_like(target_note)
            generated_ka = generated_ka_mask(probability[index], generated_mask, target_channels)
            generated_summary = summarize(generated_mask, onset)
            target_summary = summarize(target_note, onset)
            sample_legal_mask = legal_mask[index].numpy() > 0.5 if legal_mask is not None else np.ones_like(generated_mask)
            records.append(
                {
                    "chunk_id": str(batch["chunk_id"][index]),
                    "generated_notes": int(generated_mask.sum()),
                    "target_notes": int(target_note.sum()),
                    "generated_ka": int(generated_ka.sum()),
                    "target_ka": int(target_ka.sum()),
                    "generated_onset_mean": generated_summary["onset_mean_at_notes"],
                    "target_onset_mean": target_summary["onset_mean_at_notes"],
                    "generated_top25_hit": generated_summary["onset_top25_hit_rate"],
                    "target_top25_hit": target_summary["onset_top25_hit_rate"],
                    "generated_legal_rate": float(sample_legal_mask[generated_mask].mean()) if generated_mask.any() else 1.0,
                }
            )
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
            elapsed = time.perf_counter() - start_time
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "total_batches": len(loader),
                        "samples": len(records),
                        "elapsed_sec": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - start_time
    note_errors = np.asarray([record["generated_notes"] - record["target_notes"] for record in records], dtype=np.float32)
    ka_ratio_errors = []
    for record in records:
        gen_ratio = record["generated_ka"] / max(record["generated_notes"], 1)
        target_ratio = record["target_ka"] / max(record["target_notes"], 1)
        ka_ratio_errors.append(gen_ratio - target_ratio)
    ka_ratio_errors_np = np.asarray(ka_ratio_errors, dtype=np.float32)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "samples": len(records),
        "batch_size": int(args.batch_size),
        "sample_steps": int(args.sample_steps),
        "guidance_scale": float(args.guidance_scale),
        "onset_mix": float(args.onset_mix),
        "seed": int(args.seed),
        "elapsed_sec": elapsed,
        "samples_per_sec": len(records) / max(elapsed, 1e-9),
        "note_count_mae": float(np.abs(note_errors).mean()),
        "note_count_bias": float(note_errors.mean()),
        "ka_ratio_mae": float(np.abs(ka_ratio_errors_np).mean()),
        "ka_ratio_bias": float(ka_ratio_errors_np.mean()),
        "generated_onset_mean": float(np.mean([record["generated_onset_mean"] for record in records])),
        "target_onset_mean": float(np.mean([record["target_onset_mean"] for record in records])),
        "generated_top25_hit": float(np.mean([record["generated_top25_hit"] for record in records])),
        "target_top25_hit": float(np.mean([record["target_top25_hit"] for record in records])),
        "generated_legal_rate": float(np.mean([record["generated_legal_rate"] for record in records])),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
