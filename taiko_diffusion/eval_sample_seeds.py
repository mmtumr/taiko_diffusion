from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.eval_audio_alignment import generated_note_mask, summarize
from taiko_diffusion.sample_diffusion import load_condition_from_row, read_first_row, sample
from taiko_diffusion.train_diffusion import diffusion_schedule, make_model


def read_rows_by_chunk(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["chunk_id"]: row for row in csv.DictReader(file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample several seeds and evaluate onset alignment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/test.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/cache/diffusion_v2_notes_constrained/stats.json"))
    parser.add_argument("--audio-split", type=Path, default=Path("data/cache/audio_v0/test.csv"))
    parser.add_argument("--audio-stats", type=Path, default=Path("data/cache/audio_v0/stats.json"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--onset-mix", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    row = read_first_row(args.split)
    chunk_id = row["chunk_id"]
    condition_np, raw_condition = load_condition_from_row(row, stats)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
    # The model consumes normalized audio, but alignment metrics should use raw onset values.
    audio_tensor = None
    audio_rows = read_rows_by_chunk(args.audio_split)
    audio_stats = json.loads(args.audio_stats.read_text(encoding="utf-8"))
    mean = np.asarray(audio_stats["feature_mean"], dtype=np.float32)
    std = np.asarray(audio_stats["feature_std"], dtype=np.float32)
    audio_npz = np.load(audio_rows[chunk_id]["audio_npz_path"], allow_pickle=False)
    raw_audio = audio_npz["audio"].astype(np.float32)
    audio = (raw_audio - mean) / std
    audio_for_model = audio.transpose(1, 0)
    if int(config.get("model", {}).get("audio_channels", 0)) > 0:
        audio_tensor = torch.from_numpy(audio_for_model).unsqueeze(0).to(device)

    chart_rows = read_rows_by_chunk(args.split)
    chart_data = np.load(chart_rows[chunk_id]["npz_path"], allow_pickle=False)
    target_mask = chart_data["chart"].astype(np.float32)[:, 0] > 0.5
    onset = raw_audio[:, -2]
    seed_values = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    records = []
    for seed in seed_values:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generated = sample(
            model,
            condition,
            schedule,
            int(config["diffusion"]["timesteps"]),
            len(stats["target_channels"]),
            int(stats["window_frames"]),
            device,
            audio_tensor,
        )
        probability = ((generated.clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).transpose(0, 1).cpu().numpy()
        mask = generated_note_mask(
            probability,
            raw_condition,
            46.4399,
            onset=onset,
            onset_mix=float(args.onset_mix),
        )
        record = {"seed": seed, **summarize(mask, onset)}
        records.append(record)

    target = summarize(target_mask, onset)
    mean_generated = {
        key: float(np.mean([record[key] for record in records]))
        for key in ["notes", "onset_mean_at_notes", "onset_top25_hit_rate"]
    }
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "chunk_id": chunk_id,
                "records": records,
                "mean_generated": mean_generated,
                "target": target,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
