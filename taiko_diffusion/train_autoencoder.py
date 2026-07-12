from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.config import load_config
from taiko_diffusion.data.diffusion_dataset import TaikoDiffusionDataset
from taiko_diffusion.models.latent_diffusion import ChartAutoencoder1D, ChartAutoencoderKL1D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict) -> ChartAutoencoder1D:
    model_cfg = config["autoencoder"]
    if str(model_cfg.get("type", "deterministic")) == "kl":
        return ChartAutoencoderKL1D(
            chart_channels=int(model_cfg.get("chart_channels", 2)),
            latent_channels=int(model_cfg.get("latent_channels", 16)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            scale=float(model_cfg.get("scale", 1.0)),
        )
    return ChartAutoencoder1D(
        chart_channels=int(model_cfg.get("chart_channels", 2)),
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def make_loader(data_cfg: dict, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    cache_dir = Path(data_cfg["cache_dir"])
    dataset = TaikoDiffusionDataset(cache_dir / f"{split}.csv", cache_dir / "stats.json")
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
    positive_loss_weight: float,
    channel_positive_weights: torch.Tensor | None,
    count_loss_weight: float,
    latent_l2_weight: float,
    kl_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_bce = 0.0
    total_count_loss = 0.0
    total_kl_loss = 0.0
    total_count = 0
    for batch in loader:
        chart = batch["chart"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            output = model(chart, sample_posterior=is_train) if isinstance(model, ChartAutoencoderKL1D) else model(chart)
            logits, latent_or_posterior = output
            if hasattr(latent_or_posterior, "kl"):
                kl_loss = latent_or_posterior.kl()
                latent_l2 = latent_or_posterior.mode().square().mean()
            else:
                kl_loss = torch.zeros((), device=device)
                latent_l2 = latent_or_posterior.square().mean()
            positive_weights = positive_loss_weight if channel_positive_weights is None else channel_positive_weights
            weights = torch.where(chart > 0.5, positive_weights, 1.0)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, chart, reduction="none")
            bce = (bce * weights).mean()
            prob = torch.sigmoid(logits)
            count_loss = torch.nn.functional.smooth_l1_loss(
                prob.sum(dim=-1) / prob.shape[-1],
                chart.sum(dim=-1) / chart.shape[-1],
            )
            loss = bce + count_loss_weight * count_loss + latent_l2_weight * latent_l2 + kl_weight * kl_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = chart.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_bce += float(bce.detach().cpu()) * batch_size
        total_count_loss += float(count_loss.detach().cpu()) * batch_size
        total_kl_loss += float(kl_loss.detach().cpu()) * batch_size
        total_count += batch_size
    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "count_loss": total_count_loss / denom,
        "kl_loss": total_kl_loss / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chart autoencoder for latent Taiko diffusion.")
    parser.add_argument("--config", type=Path, default=Path("configs/autoencoder_v7.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    set_seed(int(training.get("seed", 20260625)))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    batch_size = int(args.batch_size or training["batch_size"])
    epochs = int(args.epochs or training["epochs"])
    num_workers = int(training.get("num_workers", 0))
    train_loader = make_loader(config["data"], "train", batch_size, num_workers, True)
    val_loader = make_loader(config["data"], "val", batch_size, num_workers, False)
    model = make_model(config).to(device)
    resume_epoch = 0
    best_val = float("inf")
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("val_loss", best_val))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    log_dir = Path(training.get("log_dir", checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume_checkpoint and history_path.exists() else []
    positive_loss_weight = float(training.get("positive_loss_weight", 4.0))
    configured_weights = training.get("channel_positive_weights")
    channel_positive_weights = None
    if configured_weights is not None:
        channel_positive_weights = torch.tensor(configured_weights, dtype=torch.float32, device=device).view(1, -1, 1)
    count_loss_weight = float(training.get("count_loss_weight", 0.25))
    latent_l2_weight = float(training.get("latent_l2_weight", 0.0001))
    kl_weight = float(training.get("kl_weight", 0.0))
    for epoch in range(resume_epoch + 1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            positive_loss_weight,
            channel_positive_weights,
            count_loss_weight,
            latent_l2_weight,
            kl_weight,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            positive_loss_weight,
            channel_positive_weights,
            count_loss_weight,
            latent_l2_weight,
            kl_weight,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "val_bce": val_metrics["bce"],
            "train_count_loss": train_metrics["count_loss"],
            "val_count_loss": val_metrics["count_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                },
                checkpoint_dir / "best.pt",
            )
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
