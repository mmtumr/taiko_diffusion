from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from taiko_diffusion.config import load_config
from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.models.latent_diffusion import AudioNoteEventPretrainer
from taiko_diffusion.train_autoencoder import set_seed


def make_model(config: dict) -> AudioNoteEventPretrainer:
    model = config["model"]
    return AudioNoteEventPretrainer(
        audio_channels=int(model["audio_channels"]),
        base_channels=int(model["base_channels"]),
        channel_mults=[int(value) for value in model["channel_mults"]],
        num_res_blocks=int(model.get("audio_scale_blocks", 2)),
        dropout=float(model.get("dropout", 0.0)),
        head_channels=int(model.get("head_channels", 64)),
    )


def make_loader(config: dict, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    chart_dir = Path(config["data"]["cache_dir"])
    audio_dir = Path(config["data"]["audio_cache_dir"])
    dataset = TaikoAudioDiffusionDataset(
        chart_dir / f"{split}.csv",
        chart_dir / "stats.json",
        audio_dir / f"{split}.csv",
        audio_dir / "stats.json",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    positive_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        audio = batch["audio"].to(device, non_blocking=True)
        chart = batch["chart"].to(device, non_blocking=True)
        target = (chart.amax(dim=1) > 0.5).float()
        with torch.set_grad_enabled(is_train):
            logits = model(audio)
            weights = torch.where(target > 0.5, positive_weight, 1.0)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        prediction = torch.sigmoid(logits) >= 0.5
        truth = target > 0.5
        true_positive = (prediction & truth).sum().float()
        precision = true_positive / prediction.sum().clamp_min(1)
        recall = true_positive / truth.sum().clamp_min(1)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        batch_size = audio.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["precision"] += float(precision) * batch_size
        totals["recall"] += float(recall) * batch_size
        totals["f1"] += float(f1) * batch_size
        total_count += batch_size
    return {name: value / max(total_count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the Mug-style audio encoder on frame-level Taiko note events.")
    parser.add_argument("--config", type=Path, default=Path("configs/audio_encoder_pretrain_v0.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    set_seed(int(training.get("seed", 20260625)))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    batch_size = int(args.batch_size or training["batch_size"])
    train_loader = make_loader(config, "train", batch_size, int(training.get("num_workers", 0)), True)
    val_loader = make_loader(config, "val", batch_size, int(training.get("num_workers", 0)), False)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    log_dir = Path(training.get("log_dir", checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    epochs = int(args.epochs or training["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            float(training.get("positive_loss_weight", 8.0)),
            device,
            optimizer,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            float(training.get("positive_loss_weight", 8.0)),
            device,
            max_batches=args.max_val_batches,
        )
        record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "audio_scale_encoder": model.audio_scale_encoder.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                },
                checkpoint_dir / "best.pt",
            )
    (log_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
