from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.config import load_config
from taiko_diffusion.data.diffusion_dataset import TaikoAudioDiffusionDataset, TaikoDiffusionDataset
from taiko_diffusion.models.chart_diffusion import ChartUNet1D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diffusion_schedule(config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    timesteps = int(config["timesteps"])
    betas = torch.linspace(float(config["beta_start"]), float(config["beta_end"]), timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


def make_model(config: dict) -> ChartUNet1D:
    model_cfg = config["model"]
    return ChartUNet1D(
        chart_channels=int(model_cfg["chart_channels"]),
        cond_dim=int(model_cfg["cond_dim"]),
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=[int(value) for value in model_cfg.get("channel_mults", [1, 2, 4])],
        dropout=float(model_cfg.get("dropout", 0.1)),
        audio_channels=int(model_cfg.get("audio_channels", 0)),
        audio_multiscale=bool(model_cfg.get("audio_multiscale", False)),
        audio_fusion=model_cfg.get("audio_fusion"),
        audio_context_dim=int(model_cfg.get("audio_context_dim", 128)),
        audio_context_tokens=int(model_cfg.get("audio_context_tokens", 128)),
        audio_attention_heads=int(model_cfg.get("audio_attention_heads", 4)),
    )


def make_loader(
    data_cfg: dict,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    cache_dir = Path(data_cfg["cache_dir"])
    if "audio_cache_dir" in data_cfg:
        audio_cache_dir = Path(data_cfg["audio_cache_dir"])
        dataset = TaikoAudioDiffusionDataset(
            cache_dir / f"{split}.csv",
            cache_dir / "stats.json",
            audio_cache_dir / f"{split}.csv",
            audio_cache_dir / "stats.json",
        )
    else:
        dataset = TaikoDiffusionDataset(cache_dir / f"{split}.csv", cache_dir / "stats.json")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_matching_weights(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    matched = {
        name: tensor
        for name, tensor in source.items()
        if name in target and tuple(tensor.shape) == tuple(target[name].shape)
    }
    target.update(matched)
    model.load_state_dict(target)
    return {
        "loaded": len(matched),
        "source": len(source),
        "target": len(target),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    schedule: dict[str, torch.Tensor],
    timesteps: int,
    positive_loss_weight: float,
    x0_loss_weight: float,
    count_loss_weight: float,
    onset_loss_weight: float,
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
        audio = batch.get("audio")
        audio = audio.to(device, non_blocking=True) if audio is not None else None
        raw_onset = batch.get("raw_onset")
        raw_onset = raw_onset.to(device, non_blocking=True) if raw_onset is not None else None
        x0 = chart * 2.0 - 1.0
        t = torch.randint(0, timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1)
        sqrt_om = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_om * noise
        with torch.set_grad_enabled(is_train):
            pred = model(xt, t, condition, audio)
            loss = torch.nn.functional.mse_loss(pred, noise, reduction="none")
            event_weight = torch.where(chart > 0.5, positive_loss_weight, 1.0)
            noise_loss = (loss * event_weight).mean()
            loss = noise_loss
            if x0_loss_weight > 0.0 or count_loss_weight > 0.0:
                pred_x0 = (xt - sqrt_om * pred) / torch.clamp(sqrt_ab, min=1e-6)
                pred_prob = ((pred_x0.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
                if x0_loss_weight > 0.0:
                    x0_loss = torch.nn.functional.mse_loss(pred_prob, chart, reduction="none")
                    x0_loss = (x0_loss * event_weight).mean()
                    loss = loss + x0_loss_weight * x0_loss
                if onset_loss_weight > 0.0 and (raw_onset is not None or audio is not None):
                    onset = raw_onset if raw_onset is not None else audio[:, -2, :].clamp_min(0.0)
                    if onset.shape[-1] != pred_prob.shape[-1]:
                        onset = torch.nn.functional.interpolate(
                            onset.unsqueeze(1),
                            size=pred_prob.shape[-1],
                            mode="linear",
                            align_corners=False,
                        ).squeeze(1)
                    onset = onset / torch.clamp(onset.amax(dim=-1, keepdim=True), min=1e-6)
                    note_error = torch.nn.functional.mse_loss(pred_prob[:, 0, :], chart[:, 0, :], reduction="none")
                    onset_weight = 1.0 + onset_weight_scale * onset
                    onset_loss = (note_error * onset_weight).mean()
                    loss = loss + onset_loss_weight * onset_loss
                if count_loss_weight > 0.0:
                    pred_count = pred_prob.sum(dim=-1) / pred_prob.shape[-1]
                    target_count = chart.sum(dim=-1) / chart.shape[-1]
                    count_loss = torch.nn.functional.smooth_l1_loss(pred_count, target_count)
                    loss = loss + count_loss_weight * count_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * x0.shape[0]
        total_count += x0.shape[0]
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chart-only Taiko diffusion v0.")
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion_v0.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    data_cfg = config["data"]
    set_seed(int(training.get("seed", 20260624)))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    batch_size = int(args.batch_size or training["batch_size"])
    epochs = int(args.epochs or training["epochs"])
    num_workers = int(training.get("num_workers", 0))
    timesteps = int(config["diffusion"]["timesteps"])
    schedule = diffusion_schedule(config["diffusion"], device)

    train_loader = make_loader(data_cfg, "train", batch_size, num_workers, True)
    val_loader = make_loader(data_cfg, "val", batch_size, num_workers, False)
    model = make_model(config).to(device)
    resume_epoch = 0
    resume_val = None
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        resume_epoch = int(resume.get("epoch", 0))
        resume_val = float(resume["val_loss"]) if "val_loss" in resume else None
    elif args.init_checkpoint is not None:
        loaded = load_matching_weights(model, args.init_checkpoint)
        print(json.dumps({"init_checkpoint": str(args.init_checkpoint), **loaded}, ensure_ascii=False))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(training.get("log_dir", checkpoint_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = resume_val if resume_val is not None else float("inf")
    history_path = log_dir / "history.json"
    if args.resume_checkpoint is not None and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    positive_loss_weight = float(training.get("positive_loss_weight", 1.0))
    x0_loss_weight = float(training.get("x0_loss_weight", 0.0))
    count_loss_weight = float(training.get("count_loss_weight", 0.0))
    onset_loss_weight = float(training.get("onset_loss_weight", 0.0))
    onset_weight_scale = float(training.get("onset_weight_scale", 1.0))
    for epoch in range(resume_epoch + 1, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            schedule,
            timesteps,
            positive_loss_weight,
            x0_loss_weight,
            count_loss_weight,
            onset_loss_weight,
            onset_weight_scale,
            device,
            optimizer,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            schedule,
            timesteps,
            positive_loss_weight,
            x0_loss_weight,
            count_loss_weight,
            onset_loss_weight,
            onset_weight_scale,
            device,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_dir / "best.pt",
            )
    (log_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
