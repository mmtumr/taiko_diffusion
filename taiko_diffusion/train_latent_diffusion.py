from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.config import load_config
from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset
from taiko_diffusion.models.latent_diffusion import (
    ChartAutoencoder1D,
    ChartAutoencoderKL1D,
    LatentUNet1D,
    encode_chart_latent,
)
from taiko_diffusion.train_diffusion import diffusion_schedule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_autoencoder(path: Path, device: torch.device) -> ChartAutoencoder1D:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["autoencoder"]
    if str(cfg.get("type", "deterministic")) == "kl":
        model = ChartAutoencoderKL1D(
            chart_channels=int(cfg.get("chart_channels", 2)),
            latent_channels=int(cfg.get("latent_channels", 16)),
            base_channels=int(cfg.get("base_channels", 64)),
            dropout=float(cfg.get("dropout", 0.0)),
            scale=float(cfg.get("scale", 1.0)),
        ).to(device)
    else:
        model = ChartAutoencoder1D(
            chart_channels=int(cfg.get("chart_channels", 2)),
            latent_channels=int(cfg.get("latent_channels", 16)),
            base_channels=int(cfg.get("base_channels", 64)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_latent_stats(path: Path | None, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if path is None:
        return None, None
    stats = json.loads(path.read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, -1, 1).clamp_min(1e-6)
    return mean, std


def make_model(config: dict) -> LatentUNet1D:
    model_cfg = config["model"]
    return LatentUNet1D(
        latent_channels=int(model_cfg["latent_channels"]),
        cond_dim=int(model_cfg["cond_dim"]),
        audio_channels=int(model_cfg["audio_channels"]),
        base_channels=int(model_cfg.get("base_channels", 128)),
        channel_mults=[int(value) for value in model_cfg.get("channel_mults", [1, 2, 2])],
        dropout=float(model_cfg.get("dropout", 0.1)),
        audio_context_dim=int(model_cfg.get("audio_context_dim", 128)),
        audio_context_tokens=int(model_cfg.get("audio_context_tokens", 128)),
        audio_attention_heads=int(model_cfg.get("audio_attention_heads", 4)),
        audio_fusion=str(model_cfg.get("audio_fusion", "token")),
        audio_scale_blocks=int(model_cfg.get("audio_scale_blocks", 2)),
    )


def make_loader(data_cfg: dict, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    cache_dir = Path(data_cfg["cache_dir"])
    audio_cache_dir = Path(data_cfg["audio_cache_dir"])
    dataset = TaikoAudioDiffusionDataset(
        cache_dir / f"{split}.csv",
        cache_dir / "stats.json",
        audio_cache_dir / f"{split}.csv",
        audio_cache_dir / "stats.json",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def apply_condition_dropout(
    condition: torch.Tensor,
    audio: torch.Tensor,
    condition_dropout: float,
    audio_dropout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition_dropout > 0.0:
        keep = (torch.rand(condition.shape[0], 1, device=condition.device) >= condition_dropout).float()
        condition = condition * keep
    if audio_dropout > 0.0:
        keep = (torch.rand(audio.shape[0], 1, 1, device=audio.device) >= audio_dropout).float()
        audio = audio * keep
    return condition, audio


def run_epoch(
    model: torch.nn.Module,
    autoencoder: ChartAutoencoder1D,
    loader: DataLoader,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    condition_dropout: float,
    audio_dropout: float,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    decoded_x0_loss_weight: float,
    decoded_onset_loss_weight: float,
    decoded_positive_loss_weight: float,
    onset_weight_scale: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        chart = batch["chart"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        raw_onset = batch.get("raw_onset")
        raw_onset = raw_onset.to(device, non_blocking=True) if raw_onset is not None else None
        with torch.no_grad():
            latent = encode_chart_latent(autoencoder, chart, sample_posterior=False)
            if latent_mean is not None and latent_std is not None:
                latent = (latent - latent_mean) / latent_std
        t = torch.randint(0, timesteps, (latent.shape[0],), device=device)
        noise = torch.randn_like(latent)
        sqrt_ab = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1)
        sqrt_om = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1)
        xt = sqrt_ab * latent + sqrt_om * noise
        condition_in, audio_in = apply_condition_dropout(
            condition,
            audio,
            condition_dropout if is_train else 0.0,
            audio_dropout if is_train else 0.0,
        )
        with torch.set_grad_enabled(is_train):
            pred = model(xt, t, condition_in, audio_in)
            loss = torch.nn.functional.mse_loss(pred, noise)
            if decoded_x0_loss_weight > 0.0 or decoded_onset_loss_weight > 0.0:
                pred_x0 = (xt - sqrt_om * pred) / torch.clamp(sqrt_ab, min=1e-6)
                decode_latent = pred_x0
                if latent_mean is not None and latent_std is not None:
                    decode_latent = decode_latent * latent_std + latent_mean
                decoded_prob = torch.sigmoid(autoencoder.decode(decode_latent))
                event_weight = torch.where(chart > 0.5, decoded_positive_loss_weight, 1.0)
                decoded_loss = torch.nn.functional.mse_loss(decoded_prob, chart, reduction="none")
                if decoded_x0_loss_weight > 0.0:
                    loss = loss + decoded_x0_loss_weight * (decoded_loss * event_weight).mean()
                if decoded_onset_loss_weight > 0.0 and raw_onset is not None:
                    onset = raw_onset
                    if onset.shape[-1] != decoded_prob.shape[-1]:
                        onset = torch.nn.functional.interpolate(
                            onset.unsqueeze(1),
                            size=decoded_prob.shape[-1],
                            mode="linear",
                            align_corners=False,
                        ).squeeze(1)
                    onset = onset / torch.clamp(onset.amax(dim=-1, keepdim=True), min=1e-6)
                    note_error = torch.nn.functional.mse_loss(decoded_prob[:, 0, :], chart[:, 0, :], reduction="none")
                    onset_weight = 1.0 + onset_weight_scale * onset
                    loss = loss + decoded_onset_loss_weight * (note_error * onset_weight).mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = latent.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent audio-conditioned Taiko diffusion.")
    parser.add_argument("--config", type=Path, default=Path("configs/latent_diffusion_v7.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
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
    autoencoder = load_autoencoder(Path(config["autoencoder"]["checkpoint"]), device)
    latent_stats_path = config["autoencoder"].get("latent_stats")
    latent_mean, latent_std = load_latent_stats(Path(latent_stats_path), device) if latent_stats_path else (None, None)
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
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(json.dumps({"init_checkpoint": str(args.init_checkpoint)}, ensure_ascii=False), flush=True)
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
    schedule = diffusion_schedule(config["diffusion"], device)
    timesteps = int(config["diffusion"]["timesteps"])
    condition_dropout = float(training.get("condition_dropout", 0.1))
    audio_dropout = float(training.get("audio_dropout", 0.1))
    decoded_x0_loss_weight = float(training.get("decoded_x0_loss_weight", 0.0))
    decoded_onset_loss_weight = float(training.get("decoded_onset_loss_weight", 0.0))
    decoded_positive_loss_weight = float(training.get("decoded_positive_loss_weight", 2.0))
    onset_weight_scale = float(training.get("onset_weight_scale", 2.0))
    for epoch in range(resume_epoch + 1, epochs + 1):
        train_loss = run_epoch(
            model,
            autoencoder,
            train_loader,
            schedule,
            timesteps,
            condition_dropout,
            audio_dropout,
            latent_mean,
            latent_std,
            decoded_x0_loss_weight,
            decoded_onset_loss_weight,
            decoded_positive_loss_weight,
            onset_weight_scale,
            device,
            optimizer,
        )
        val_loss = run_epoch(
            model,
            autoencoder,
            val_loader,
            schedule,
            timesteps,
            condition_dropout,
            audio_dropout,
            latent_mean,
            latent_std,
            decoded_x0_loss_weight,
            decoded_onset_loss_weight,
            decoded_positive_loss_weight,
            onset_weight_scale,
            device,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "autoencoder_checkpoint": str(config["autoencoder"]["checkpoint"]),
                },
                checkpoint_dir / "best.pt",
            )
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
