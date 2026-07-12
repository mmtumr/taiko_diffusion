from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.eval_latent_full_test import ddim_sample_batch
from taiko_diffusion.export_sample_tja import balloon_span_indices
from taiko_diffusion.sample_latent_diffusion import infer_latent_shape, set_seed
from taiko_diffusion.train_diffusion import diffusion_schedule
from taiko_diffusion.train_latent_diffusion import load_autoencoder, load_latent_stats, make_model


def spans_from_channels(array: np.ndarray, names: list[str], threshold: float = 0.5) -> list[tuple[int, int]]:
    channel = {name: index for index, name in enumerate(names)}
    starts = array[:, channel["hold_start"]] > threshold
    body = array[:, channel["hold_body"]] > threshold
    spans = []
    for start in np.flatnonzero(starts):
        end = int(start)
        cursor = end + 1
        while cursor < len(body) and body[cursor] and not starts[cursor]:
            end = cursor
            cursor += 1
        if end > start:
            spans.append((int(start), end))
    return spans


def event_matches(generated: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[int, int, int]:
    generated_indices = np.flatnonzero(generated)
    unmatched = list(np.flatnonzero(target))
    matches = 0
    for frame in generated_indices:
        candidates = [index for index, value in enumerate(unmatched) if abs(int(value) - int(frame)) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda index: abs(int(unmatched[index]) - int(frame)))
            unmatched.pop(best)
            matches += 1
    return matches, len(generated_indices) - matches, len(unmatched)


def f1_counts(counts: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn = (int(value) for value in counts.sum(axis=0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v13 neural holds, balloons, BPM changes, and HS changes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--audio-split", type=Path, required=True)
    parser.add_argument("--audio-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval/v13_events.json"))
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    names = [str(name) for name in stats["target_channels"]]
    conditions = [str(name) for name in stats["condition_names"]]
    condition_index = {name: index for index, name in enumerate(conditions)}
    target_index = {name: index for index, name in enumerate(names)}
    dataset = TaikoAudioDiffusionDataset(args.split, args.stats, args.audio_split, args.audio_stats)
    ranked = []
    for index in range(len(dataset)):
        item = dataset[index]
        chart = item["chart"].numpy()
        score = (
            4.0 * chart[target_index["hold_start"]].sum()
            + chart[target_index["hold_body"]].sum()
            + 3.0 * chart[target_index["bpm_change_event"]].sum()
            + 3.0 * chart[target_index["scroll_change_event"]].sum()
        )
        ranked.append((float(score), index))
    selected = [index for _, index in sorted(ranked, reverse=True)[: args.samples]]

    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    latent_mean, latent_std = load_latent_stats(Path(config["autoencoder"]["latent_stats"]), device)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = diffusion_schedule(config["diffusion"], device)
    latent_shape = infer_latent_shape(autoencoder, len(names), int(stats["window_frames"]), device)
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)
    variants = ["base", "bpm_off", "bpm_high", "hs_off", "hs_on"]
    records = []

    jobs = []
    for row_index in selected:
        item = dataset[row_index]
        raw = item["condition_raw"].numpy()
        for seed_offset in range(args.seeds):
            generator = torch.Generator().manual_seed(args.seed + row_index * 1009 + seed_offset)
            noise = torch.randn(latent_shape, generator=generator)
            for variant in variants:
                changed = raw.copy()
                if variant == "bpm_off":
                    changed[condition_index["bpm_rhythm_bin"]] = 0.0
                elif variant == "bpm_high":
                    changed[condition_index["bpm_rhythm_bin"]] = 2.0
                elif variant == "hs_off":
                    changed[condition_index["hs_change_bin"]] = 0.0
                elif variant == "hs_on":
                    changed[condition_index["hs_change_bin"]] = 1.0
                jobs.append((row_index, seed_offset, variant, changed, item, noise))

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        condition = torch.from_numpy(np.stack([(job[3] - condition_mean) / condition_std for job in batch])).to(device)
        audio = torch.stack([job[4]["audio"] for job in batch]).to(device)
        if bool(config["model"].get("use_legal_mask_channel", False)):
            audio = torch.cat([audio, torch.stack([job[4]["legal_mask"] for job in batch]).to(device).unsqueeze(1)], dim=1)
        noise = torch.stack([job[5] for job in batch]).to(device)
        latent = ddim_sample_batch(
            model, condition, audio, schedule, int(config["diffusion"]["timesteps"]), args.sample_steps,
            latent_shape, args.guidance_scale, device, initial_noise=noise,
        )
        latent = latent * latent_std + latent_mean
        probability = torch.sigmoid(autoencoder.decode(latent)).transpose(1, 2).cpu().numpy()
        for job, generated in zip(batch, probability):
            row_index, seed_offset, variant, raw, item, _ = job
            target = item["chart"].transpose(0, 1).numpy()
            generated_spans = spans_from_channels(generated, names)
            target_spans = spans_from_channels(target, names)
            hold_counts = event_matches(
                generated[:, target_index["hold_start"]] > 0.5,
                target[:, target_index["hold_start"]] > 0.5,
            )
            bpm_mask = generated[:, target_index["bpm_change_event"]] > 0.5
            target_bpm = target[:, target_index["bpm_change_event"]] > 0.5
            hs_mask = generated[:, target_index["scroll_change_event"]] > 0.5
            target_hs = target[:, target_index["scroll_change_event"]] > 0.5
            balloon_ids = balloon_span_indices(len(generated_spans), float(raw[condition_index["balloon_roll_ratio"]]))
            balloon_hits = [
                max(1, int(np.floor((generated_spans[index][1] - generated_spans[index][0]) * stats["frame_ms"] / 1000.0 * 15.0)))
                for index in sorted(balloon_ids)
            ]
            records.append({
                "row": row_index,
                "seed": seed_offset,
                "variant": variant,
                "hold_counts": hold_counts,
                "generated_holds": len(generated_spans),
                "target_holds": len(target_spans),
                "generated_hold_frames": int(sum(end - begin for begin, end in generated_spans)),
                "target_hold_frames": int(sum(end - begin for begin, end in target_spans)),
                "balloons": len(balloon_ids),
                "balloon_hits": balloon_hits,
                "bpm_counts": event_matches(bpm_mask, target_bpm),
                "bpm_events": int(bpm_mask.sum()),
                "bpm_value_mae": float(np.abs(generated[target_bpm, target_index["bpm_value"]] - target[target_bpm, target_index["bpm_value"]]).mean() * 300.0) if target_bpm.any() else None,
                "hs_counts": event_matches(hs_mask, target_hs),
                "hs_events": int(hs_mask.sum()),
                "hs_value_mae": float(np.abs(generated[target_hs, target_index["scroll_value"]] - target[target_hs, target_index["scroll_value"]]).mean() * 4.0) if target_hs.any() else None,
            })
        print(json.dumps({"completed": min(start + len(batch), len(jobs)), "total": len(jobs)}), flush=True)

    base = [record for record in records if record["variant"] == "base"]
    by_variant = {variant: [record for record in records if record["variant"] == variant] for variant in variants}
    summary = {
        "checkpoint": str(args.checkpoint),
        "samples": len(selected),
        "seeds": args.seeds,
        "hold_start": f1_counts(np.asarray([record["hold_counts"] for record in base])),
        "generated_hold_rate": float(np.mean([record["generated_holds"] > 0 for record in base])),
        "generated_holds_mean": float(np.mean([record["generated_holds"] for record in base])),
        "target_holds_mean": float(np.mean([record["target_holds"] for record in base])),
        "hold_frame_count_mae": float(np.mean([abs(record["generated_hold_frames"] - record["target_hold_frames"]) for record in base])),
        "bpm_event": f1_counts(np.asarray([record["bpm_counts"] for record in base])),
        "bpm_value_mae": float(np.mean([record["bpm_value_mae"] for record in base if record["bpm_value_mae"] is not None])),
        "hs_event": f1_counts(np.asarray([record["hs_counts"] for record in base])),
        "hs_value_mae": float(np.mean([record["hs_value_mae"] for record in base if record["hs_value_mae"] is not None])),
        "condition_response": {
            variant: {
                "bpm_events_mean": float(np.mean([record["bpm_events"] for record in rows])),
                "hs_events_mean": float(np.mean([record["hs_events"] for record in rows])),
            }
            for variant, rows in by_variant.items()
        },
        "balloon_formula_valid": all(all(value >= 1 for value in record["balloon_hits"]) for record in base),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
