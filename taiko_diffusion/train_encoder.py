from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "PyTorch is not installed in this Miniforge environment. "
        "Install torch before running training."
    ) from exc

from taiko_diffusion.data.dataset import TaikoTensorDataset
from taiko_diffusion.models.encoder import (
    TaikoEncoderBranchedSparse,
    TaikoEncoderGroupedHeads,
    TaikoEncoderSparseHeads,
    TaikoEncoderV0,
)
from taiko_diffusion.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split_csv: Path, stats_path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TaikoTensorDataset(split_csv, stats_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def parse_target_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_map(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): int(val) for key, val in value.items()}
    result: dict[str, int] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, raw_val = item.split(":", 1)
        result[key.strip()] = int(raw_val.strip())
    return result


def build_head_groups(model_cfg: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, value in model_cfg.items():
        if key.endswith("_targets"):
            labels = parse_target_list(value)
            if labels:
                groups[key[: -len("_targets")]] = labels
    return groups


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    chart_grid = config["chart_grid"]
    data_cfg = config["data"]
    common = {
        "input_channels": len(chart_grid["channels"]),
        "conv_channels": int(model_cfg["conv_channels"]),
        "downsample_layers": int(model_cfg.get("downsample_layers", 4)),
        "transformer_layers": int(model_cfg.get("transformer_layers", 3)),
        "transformer_heads": int(model_cfg["transformer_heads"]),
        "latent_dim": int(model_cfg["latent_dim"]),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    if bool(model_cfg.get("grouped_heads", False)):
        return TaikoEncoderGroupedHeads(
            target_columns=list(data_cfg["target_columns"]),
            head_groups=build_head_groups(model_cfg),
            **common,
        )
    if bool(model_cfg.get("sparse_heads", False)):
        return TaikoEncoderSparseHeads(
            output_dim=len(data_cfg["target_columns"]),
            sparse_targets=parse_target_list(model_cfg.get("sparse_targets")),
            **common,
        )
    if bool(model_cfg.get("branched_sparse", False)):
        return TaikoEncoderBranchedSparse(
            target_columns=list(data_cfg["target_columns"]),
            core_targets=parse_target_list(model_cfg.get("core_targets")),
            sparse_targets=parse_target_list(model_cfg.get("sparse_targets")),
            input_channels=len(chart_grid["channels"]),
            conv_channels=int(model_cfg["conv_channels"]),
            shared_downsample_layers=int(model_cfg.get("shared_downsample_layers", 2)),
            branch_downsample_layers=int(model_cfg.get("branch_downsample_layers", 2)),
            core_transformer_layers=int(model_cfg.get("core_transformer_layers", 2)),
            event_transformer_layers=int(model_cfg.get("event_transformer_layers", 1)),
            transformer_heads=int(model_cfg["transformer_heads"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            class_targets=parse_target_list(model_cfg.get("class_targets")),
            class_dims=parse_int_map(model_cfg.get("class_dims")),
            class_detach=bool(model_cfg.get("class_detach", False)),
            class_branch_downsample_layers=int(model_cfg.get("class_branch_downsample_layers", 0)),
            class_transformer_layers=int(model_cfg.get("class_transformer_layers", 1)),
        )
    return TaikoEncoderV0(
        output_dim=len(data_cfg["target_columns"]),
        **common,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        y_raw = batch["y_raw"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            output = model(x)
            loss = criterion(output, y, y_raw)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.size(0)
        total_count += x.size(0)
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Taiko encoder v0.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder_v0.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    model_cfg = config["model"]
    chart_grid = config["chart_grid"]
    data_cfg = config["data"]

    seed = int(training.get("seed", 20260619))
    set_seed(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    split_dir = Path(training["split_dir"])
    stats_path = Path(training["stats_path"])
    batch_size = int(args.batch_size or training["batch_size"])
    epochs = int(args.epochs or training["epochs"])

    train_loader = make_loader(split_dir / "train.csv", stats_path, batch_size, shuffle=True)
    val_loader = make_loader(split_dir / "val.csv", stats_path, batch_size, shuffle=False)

    model = build_model(config).to(device)

    loss_weights_config = training.get("loss_weights", {})
    positive_weights_config = training.get("positive_loss_weights", {})
    positive_threshold = float(training.get("positive_loss_threshold", 0.0))
    label_names = list(data_cfg["target_columns"])
    loss_weights = torch.tensor(
        [float(loss_weights_config.get(name, 1.0)) for name in label_names],
        dtype=torch.float32,
        device=device,
    )
    positive_loss_weights = torch.tensor(
        [float(positive_weights_config.get(name, 1.0)) for name in label_names],
        dtype=torch.float32,
        device=device,
    )

    sparse_targets = parse_target_list(model_cfg.get("sparse_targets"))
    sparse_indices = [label_names.index(name) for name in sparse_targets if name in label_names]
    sparse_class_weight = float(training.get("sparse_class_loss_weight", 0.0))
    sparse_regression_weight = float(training.get("sparse_regression_weight", 1.0))
    sparse_pos_weights_config = training.get("sparse_pos_weights", {})
    sparse_pos_weights = torch.tensor(
        [float(sparse_pos_weights_config.get(name, 1.0)) for name in sparse_targets if name in label_names],
        dtype=torch.float32,
        device=device,
    )
    class_targets = parse_target_list(model_cfg.get("class_targets"))
    class_indices = [label_names.index(name) for name in class_targets if name in label_names]
    class_loss_weight = float(training.get("class_loss_weight", 0.0))
    class_target_set = set(class_targets)

    def criterion(
        output: torch.Tensor | dict[str, torch.Tensor],
        target: torch.Tensor,
        target_raw: torch.Tensor,
    ) -> torch.Tensor:
        pred = output["regression"] if isinstance(output, dict) else output
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        positive = target_raw > positive_threshold
        sample_weights = torch.where(
            positive,
            positive_loss_weights.unsqueeze(0),
            torch.ones_like(positive_loss_weights).unsqueeze(0),
        )
        if class_indices:
            sample_weights = sample_weights.clone()
            sample_weights[:, class_indices] = 0.0
        if sparse_indices:
            reg_mask = torch.ones_like(loss)
            sparse_positive = positive[:, sparse_indices]
            reg_mask[:, sparse_indices] = torch.where(
                sparse_positive,
                torch.full_like(loss[:, sparse_indices], sparse_regression_weight),
                torch.zeros_like(loss[:, sparse_indices]),
            )
            regression_loss = (loss * loss_weights * sample_weights * reg_mask).sum()
            regression_loss = regression_loss / torch.clamp(
                (loss_weights * sample_weights * reg_mask).sum(), min=1.0
            )
        else:
            regression_loss = (loss * loss_weights * sample_weights).mean()

        class_loss = torch.tensor(0.0, dtype=regression_loss.dtype, device=regression_loss.device)
        if isinstance(output, dict) and "class_logits" in output and class_indices:
            for name, index in zip(class_targets, class_indices):
                logits = output["class_logits"][name]
                class_target = target_raw[:, index].round().long().clamp(0, logits.shape[1] - 1)
                class_loss = class_loss + torch.nn.functional.cross_entropy(logits, class_target)
            class_loss = class_loss / max(len(class_indices), 1)

        if not isinstance(output, dict) or "sparse_logits" not in output or not sparse_indices:
            return regression_loss + class_loss_weight * class_loss

        sparse_target = positive[:, sparse_indices].float()
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["sparse_logits"],
            sparse_target,
            pos_weight=sparse_pos_weights,
        )
        total = regression_loss + sparse_class_weight * class_loss
        if isinstance(output, dict) and "class_logits" in output and class_indices:
            ce_loss = torch.tensor(0.0, dtype=regression_loss.dtype, device=regression_loss.device)
            for name, index in zip(class_targets, class_indices):
                logits = output["class_logits"][name]
                class_target = target_raw[:, index].round().long().clamp(0, logits.shape[1] - 1)
                ce_loss = ce_loss + torch.nn.functional.cross_entropy(logits, class_target)
            total = total + class_loss_weight * (ce_loss / max(len(class_indices), 1))
        return total

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    early_stop_patience = int(training.get("early_stop_patience", 0))
    early_stop_min_delta = float(training.get("early_stop_min_delta", 0.0))
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_loss < best_val - early_stop_min_delta:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_dir / "best.pt",
            )
        if early_stop_patience > 0 and epoch - best_epoch >= early_stop_patience:
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                    },
                    ensure_ascii=False,
                )
            )
            break
    (checkpoint_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
