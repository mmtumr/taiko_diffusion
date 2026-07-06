from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.train_diffusion import diffusion_schedule, make_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def ddpm_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    audio: torch.Tensor | None,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    chart_channels: int,
    window_frames: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = condition.shape[0]
    x = torch.randn((batch_size, chart_channels, window_frames), device=device)
    for step in range(timesteps - 1, -1, -1):
        t = torch.full((batch_size,), step, dtype=torch.long, device=device)
        pred_noise = model(x, t, condition, audio)
        beta = schedule["betas"][step]
        alpha = schedule["alphas"][step]
        alpha_bar = schedule["alpha_bar"][step]
        mean = (x - beta * pred_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        if step > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    audio: torch.Tensor | None,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    sample_steps: int,
    chart_channels: int,
    window_frames: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = condition.shape[0]
    x = torch.randn((batch_size, chart_channels, window_frames), device=device)
    steps = np.linspace(timesteps - 1, 0, num=min(sample_steps, timesteps), dtype=np.int64)
    steps = np.unique(steps)[::-1]
    if steps[0] != timesteps - 1:
        steps = np.concatenate([[timesteps - 1], steps])
    if steps[-1] != 0:
        steps = np.concatenate([steps, [0]])

    for index, step in enumerate(steps):
        t = torch.full((batch_size,), int(step), dtype=torch.long, device=device)
        pred_noise = model(x, t, condition, audio)
        alpha_bar = schedule["alpha_bar"][int(step)]
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_om = torch.sqrt(1.0 - alpha_bar)
        pred_x0 = (x - sqrt_om * pred_noise) / torch.clamp(sqrt_ab, min=1e-6)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)
        if index + 1 == len(steps):
            x = pred_x0
            break
        next_step = int(steps[index + 1])
        alpha_bar_next = schedule["alpha_bar"][next_step]
        x = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
    return x


def select_notes(
    probability: np.ndarray,
    condition_raw: np.ndarray,
    condition_names: list[str],
    frame_ms: float,
    onset: np.ndarray | None,
    onset_mix: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = {name: float(condition_raw[index]) for index, name in enumerate(condition_names)}
    note_count = int(round(max(values.get("avg_density", 0.0), 0.0) * probability.shape[0] * frame_ms / 1000.0))
    note_count = max(0, min(note_count, probability.shape[0]))
    score = probability[:, 0].copy()
    if onset is not None and onset_mix > 0.0:
        onset_score = onset.astype(np.float32)
        if onset_score.shape[0] != score.shape[0]:
            x_old = np.linspace(0.0, 1.0, num=onset_score.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, num=score.shape[0], dtype=np.float32)
            onset_score = np.interp(x_new, x_old, onset_score).astype(np.float32)
        onset_score = onset_score - float(onset_score.min())
        onset_score = onset_score / max(float(onset_score.max()), 1e-6)
        score = score + float(onset_mix) * onset_score

    note_mask = np.zeros(probability.shape[0], dtype=bool)
    if note_count > 0:
        note_mask[np.argpartition(score, -note_count)[-note_count:]] = True

    ka_mask = np.zeros(probability.shape[0], dtype=bool)
    ka_count = int(round(note_count * min(max(values.get("ka_ratio", 0.0), 0.0), 1.0)))
    selected = np.where(note_mask)[0]
    if ka_count > 0 and selected.size > 0:
        ka_count = min(ka_count, selected.size)
        scores = probability[selected, 1]
        ka_indices = selected[np.argpartition(scores, -ka_count)[-ka_count:]]
        ka_mask[ka_indices] = True
    return note_mask, ka_mask


def onset_summary(mask: np.ndarray, onset: np.ndarray) -> tuple[float, float]:
    if mask.sum() == 0:
        return 0.0, 0.0
    selected = onset[mask]
    threshold = float(np.quantile(onset, 0.75))
    return float(selected.mean()), float((selected >= threshold).mean())


def load_raw_onset(audio_npz_path: str) -> np.ndarray:
    data = np.load(audio_npz_path, allow_pickle=False)
    audio = data["audio"].astype(np.float32)
    return audio[:, -2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a diffusion checkpoint on the full audio test split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/full_test_v4_summary.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--sample-steps", type=int, default=50)
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
    frame_ms = float(stats["frame_ms"])
    window_frames = int(stats["window_frames"])
    chart_channels = len(target_channels)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    dataset = TaikoAudioDiffusionDataset(args.split, args.stats, args.audio_split, args.audio_stats)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    timesteps = int(config["diffusion"]["timesteps"])

    records = []
    start_time = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        condition = batch["condition"].to(device)
        audio = batch["audio"].to(device)
        if args.sampler == "ddpm":
            generated = ddpm_sample(
                model,
                condition,
                audio,
                schedule,
                timesteps,
                chart_channels,
                window_frames,
                device,
            )
        else:
            generated = ddim_sample(
                model,
                condition,
                audio,
                schedule,
                timesteps,
                int(args.sample_steps),
                chart_channels,
                window_frames,
                device,
            )
        probability = ((generated.clamp(-1.0, 1.0) + 1.0) * 0.5).transpose(1, 2).cpu().numpy()
        target = batch["chart"].transpose(1, 2).numpy()
        raw_condition = batch["condition_raw"].numpy()
        for index in range(probability.shape[0]):
            onset = load_raw_onset(str(batch["audio_npz_path"][index]))
            note_mask, ka_mask = select_notes(
                probability[index],
                raw_condition[index],
                condition_names,
                frame_ms,
                onset,
                float(args.onset_mix),
            )
            target_note = target[index, :, 0] > 0.5
            target_ka = target[index, :, 1] > 0.5
            gen_onset_mean, gen_top25 = onset_summary(note_mask, onset)
            target_onset_mean, target_top25 = onset_summary(target_note, onset)
            records.append(
                {
                    "chunk_id": str(batch["chunk_id"][index]),
                    "generated_notes": int(note_mask.sum()),
                    "target_notes": int(target_note.sum()),
                    "generated_ka": int(ka_mask.sum()),
                    "target_ka": int(target_ka.sum()),
                    "generated_onset_mean": gen_onset_mean,
                    "target_onset_mean": target_onset_mean,
                    "generated_top25_hit": gen_top25,
                    "target_top25_hit": target_top25,
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
        "sampler": str(args.sampler),
        "sample_steps": int(args.sample_steps),
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
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
