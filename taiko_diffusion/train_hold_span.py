from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.config import load_config
from taiko_diffusion.data.hold_span_dataset import HoldSpanDataset, hold_span_features
from taiko_diffusion.models.hold_span import HoldSpanHead


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: dict, split: str) -> HoldSpanDataset:
    data = config["data"]
    cache_dir = Path(data["cache_dir"])
    audio_dir = Path(data["audio_cache_dir"])
    return HoldSpanDataset(
        str(cache_dir / f"{split}.csv"),
        str(cache_dir / "stats.json"),
        str(audio_dir / f"{split}.csv"),
        str(audio_dir / "stats.json"),
        int(config["model"]["query_count"]),
        [str(name) for name in data["context_channels"]],
    )


def make_model(config: dict) -> HoldSpanHead:
    model = config["model"]
    data = config["data"]
    input_channels = int(model["audio_channels"]) + len(data["context_channels"]) + 2
    return HoldSpanHead(
        input_channels=input_channels,
        condition_dim=int(model["condition_dim"]),
        query_count=int(model["query_count"]),
        hidden_dim=int(model.get("hidden_dim", 192)),
        context_tokens=int(model.get("context_tokens", 256)),
        encoder_layers=int(model.get("encoder_layers", 3)),
        decoder_layers=int(model.get("decoder_layers", 3)),
        attention_heads=int(model.get("attention_heads", 6)),
        dropout=float(model.get("dropout", 0.1)),
    )


def span_loss(output: dict[str, torch.Tensor], batch: dict, config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    target_exists = batch["span_exists"]
    target_start = batch["span_start"]
    target_duration = batch["span_duration_log"]
    positive = target_exists > 0.5
    positive_weight = torch.as_tensor(float(config.get("exist_positive_weight", 8.0)), device=target_exists.device)
    existence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["exist_logits"], target_exists, pos_weight=positive_weight
    )
    if positive.any():
        start_index = torch.clamp(
            torch.round(target_start * float(output["start_logits"].shape[-1] - 1)).long(),
            min=0,
            max=output["start_logits"].shape[-1] - 1,
        )
        start_class_loss = torch.nn.functional.cross_entropy(output["start_logits"][positive], start_index[positive])
        start_regression_loss = torch.nn.functional.smooth_l1_loss(output["start"][positive], target_start[positive])
        start_loss = start_class_loss + float(config.get("start_regression_weight", 1.0)) * start_regression_loss
        duration_loss = torch.nn.functional.smooth_l1_loss(
            output["duration_log"][positive], target_duration[positive]
        )
    else:
        start_loss = output["start"].sum() * 0.0
        duration_loss = output["duration_log"].sum() * 0.0
    probabilities = torch.sigmoid(output["exist_logits"])
    target_count = target_exists.sum(dim=1)
    query_count = float(target_exists.shape[1])
    count_loss = torch.nn.functional.smooth_l1_loss(
        probabilities.sum(dim=1) / query_count,
        target_count / query_count,
    )

    window_frames = batch["window_frames"].float().unsqueeze(1)
    duration_frames = torch.expm1(output["duration_log"] * float(config["duration_log_scale"]))
    predicted_end = output["start"] + duration_frames / torch.clamp(window_frames - 1.0, min=1.0)
    adjacent_positive = positive[:, :-1] & positive[:, 1:]
    if adjacent_positive.any():
        overlap = torch.relu(predicted_end[:, :-1] - output["start"][:, 1:])
        order_loss = overlap[adjacent_positive].mean()
    else:
        order_loss = output["start"].sum() * 0.0

    total = (
        float(config.get("exist_weight", 1.0)) * existence_loss
        + float(config.get("start_weight", 4.0)) * start_loss
        + float(config.get("duration_weight", 4.0)) * duration_loss
        + float(config.get("count_weight", 0.5)) * count_loss
        + float(config.get("order_weight", 1.0)) * order_loss
    )
    return total, {
        "exist": float(existence_loss.detach()),
        "start": float(start_loss.detach()),
        "duration": float(duration_loss.detach()),
        "count": float(count_loss.detach()),
        "order": float(order_loss.detach()),
    }


def run_epoch(
    model: HoldSpanHead,
    loader: DataLoader,
    loss_config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ["loss", "exist", "start", "duration", "count", "order"]}
    count = 0
    for batch in loader:
        tensor_batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.set_grad_enabled(training):
            output = model(hold_span_features(tensor_batch), tensor_batch["condition"])
            loss, parts = span_loss(output, tensor_batch, loss_config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = int(tensor_batch["condition"].shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += value * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an explicit neural hold span prediction head.")
    parser.add_argument("--config", type=Path, default=Path("configs/hold_span_v0.yaml"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    set_seed(int(training.get("seed", 20260719)))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    config["loss"]["duration_log_scale"] = train_dataset.duration_log_scale
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or training["epochs"])
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(model, train_loader, config["loss"], device, optimizer)
        val_metrics = run_epoch(model, val_loader, config["loss"], device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if device.type == "cuda":
            record["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            record["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024**3)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "duration_log_scale": train_dataset.duration_log_scale,
                },
                checkpoint_dir / "best.pt",
            )
    (checkpoint_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
