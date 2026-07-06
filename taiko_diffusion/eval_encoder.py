from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.data.dataset import TaikoTensorDataset
from taiko_diffusion.models.encoder import (
    TaikoEncoderBranchedSparse,
    TaikoEncoderGroupedHeads,
    TaikoEncoderSparseHeads,
    TaikoEncoderV0,
)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        ranks[order[index : end + 1]] = rank
        index = end + 1
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rankdata(a), rankdata(b))


def inverse_targets(values: np.ndarray, stats: dict) -> np.ndarray:
    label_names = [str(x) for x in stats["label_names"]]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    transforms = dict(stats["transforms"])
    raw = values * std + mean
    raw = raw.copy()
    for index, name in enumerate(label_names):
        if transforms.get(name) == "log1p":
            raw[:, index] = np.expm1(raw[:, index])
        elif transforms.get(name) == "logit100":
            raw[:, index] = 101.0 / (1.0 + np.exp(-raw[:, index])) - 0.5
            raw[:, index] = np.clip(raw[:, index], 0.0, 100.0)
    return raw


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


def build_head_groups(model_config: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, value in model_config.items():
        if key.endswith("_targets"):
            labels = parse_target_list(value)
            if labels:
                groups[key[: -len("_targets")]] = labels
    return groups


def build_model(checkpoint: dict) -> torch.nn.Module:
    config = checkpoint["config"]
    model_config = config["model"]
    grid_config = config["chart_grid"]
    data_config = config["data"]
    common = {
        "input_channels": len(grid_config["channels"]),
        "conv_channels": int(model_config["conv_channels"]),
        "downsample_layers": int(model_config.get("downsample_layers", 4)),
        "transformer_layers": int(model_config.get("transformer_layers", 3)),
        "transformer_heads": int(model_config["transformer_heads"]),
        "latent_dim": int(model_config["latent_dim"]),
        "dropout": float(model_config.get("dropout", 0.1)),
    }
    if bool(model_config.get("grouped_heads", False)):
        model = TaikoEncoderGroupedHeads(
            target_columns=list(data_config["target_columns"]),
            head_groups=build_head_groups(model_config),
            **common,
        )
    elif bool(model_config.get("sparse_heads", False)):
        model = TaikoEncoderSparseHeads(
            output_dim=len(data_config["target_columns"]),
            sparse_targets=parse_target_list(model_config.get("sparse_targets")),
            **common,
        )
    elif bool(model_config.get("branched_sparse", False)):
        model = TaikoEncoderBranchedSparse(
            target_columns=list(data_config["target_columns"]),
            core_targets=parse_target_list(model_config.get("core_targets")),
            sparse_targets=parse_target_list(model_config.get("sparse_targets")),
            input_channels=len(grid_config["channels"]),
            conv_channels=int(model_config["conv_channels"]),
            shared_downsample_layers=int(model_config.get("shared_downsample_layers", 2)),
            branch_downsample_layers=int(model_config.get("branch_downsample_layers", 2)),
            core_transformer_layers=int(model_config.get("core_transformer_layers", 2)),
            event_transformer_layers=int(model_config.get("event_transformer_layers", 1)),
            transformer_heads=int(model_config["transformer_heads"]),
            latent_dim=int(model_config["latent_dim"]),
            dropout=float(model_config.get("dropout", 0.1)),
            class_targets=parse_target_list(model_config.get("class_targets")),
            class_dims=parse_int_map(model_config.get("class_dims")),
            class_detach=bool(model_config.get("class_detach", False)),
            class_branch_downsample_layers=int(model_config.get("class_branch_downsample_layers", 0)),
            class_transformer_layers=int(model_config.get("class_transformer_layers", 1)),
        )
    else:
        model = TaikoEncoderV0(
            output_dim=len(data_config["target_columns"]),
            **common,
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_split(
    model: torch.nn.Module,
    config: dict,
    split_csv: Path,
    stats_path: Path,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    label_names = [str(x) for x in stats["label_names"]]
    model_config = config.get("model", {})
    eval_config = config.get("eval", {})
    sparse_targets = parse_target_list(model_config.get("sparse_targets"))
    sparse_indices = [label_names.index(name) for name in sparse_targets if name in label_names]
    class_targets = parse_target_list(model_config.get("class_targets"))
    class_indices = [label_names.index(name) for name in class_targets if name in label_names]
    sparse_eval_mode = str(eval_config.get("sparse_eval_mode", "prob_scale"))
    sparse_threshold = float(eval_config.get("sparse_threshold", 0.5))
    dataset = TaikoTensorDataset(split_csv, stats_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    preds: list[np.ndarray] = []
    sparse_probs: list[np.ndarray] = []
    class_preds: dict[str, list[np.ndarray]] = {name: [] for name in class_targets}
    class_probs: dict[str, list[np.ndarray]] = {name: [] for name in class_targets}
    trues: list[np.ndarray] = []
    titles: list[str] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["x"])
            if isinstance(output, dict):
                pred = output["regression"].cpu().numpy()
                if "sparse_logits" in output:
                    sparse_probs.append(torch.sigmoid(output["sparse_logits"]).cpu().numpy())
                if "class_logits" in output:
                    for name in class_targets:
                        probs = torch.softmax(output["class_logits"][name], dim=1).cpu().numpy()
                        class_probs[name].append(probs)
                        class_preds[name].append(np.argmax(probs, axis=1).astype(np.float32))
            else:
                pred = output.cpu().numpy()
            preds.append(pred)
            trues.append(batch["y"].cpu().numpy())
            titles.extend(str(x) for x in batch["title"])
            sample_ids.extend(str(x) for x in batch["sample_id"])

    pred_norm = np.vstack(preds)
    true_norm = np.vstack(trues)
    pred_raw = inverse_targets(pred_norm, stats)
    true_raw = inverse_targets(true_norm, stats)
    sparse_prob_raw = np.vstack(sparse_probs) if sparse_probs else None
    if sparse_prob_raw is not None and sparse_indices:
        for sparse_pos, label_index in enumerate(sparse_indices):
            positive_score = np.maximum(pred_raw[:, label_index], 0.0)
            if sparse_eval_mode == "hard_gate":
                pred_raw[:, label_index] = np.where(
                    sparse_prob_raw[:, sparse_pos] >= sparse_threshold,
                    positive_score,
                    0.0,
                )
            else:
                pred_raw[:, label_index] = positive_score * sparse_prob_raw[:, sparse_pos]
    class_pred_raw: dict[str, np.ndarray] = {
        name: np.concatenate(parts) for name, parts in class_preds.items() if parts
    }
    class_prob_raw: dict[str, np.ndarray] = {
        name: np.vstack(parts) for name, parts in class_probs.items() if parts
    }
    for name, label_index in zip(class_targets, class_indices):
        if name in class_pred_raw:
            pred_raw[:, label_index] = class_pred_raw[name]

    metrics: list[dict] = []
    for index, name in enumerate(label_names):
        error = pred_raw[:, index] - true_raw[:, index]
        metrics.append(
            {
                "label": name,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "pearson": corr(true_raw[:, index], pred_raw[:, index]),
                "spearman": spearman(true_raw[:, index], pred_raw[:, index]),
                "true_mean": float(true_raw[:, index].mean()),
                "pred_mean": float(pred_raw[:, index].mean()),
                "true_min": float(true_raw[:, index].min()),
                "true_max": float(true_raw[:, index].max()),
            }
        )

    norm_sample_error = np.mean(np.abs(pred_norm - true_norm), axis=1)
    worst: list[dict] = []
    for sample_index in np.argsort(norm_sample_error)[-20:][::-1]:
        row = {
            "sample_id": sample_ids[sample_index],
            "title": titles[sample_index],
            "norm_mae": float(norm_sample_error[sample_index]),
        }
        for label_index, name in enumerate(label_names):
            row[f"{name}_true"] = float(true_raw[sample_index, label_index])
            row[f"{name}_pred"] = float(pred_raw[sample_index, label_index])
        if sparse_prob_raw is not None:
            for sparse_pos, name in enumerate(sparse_targets):
                row[f"{name}_prob"] = float(sparse_prob_raw[sample_index, sparse_pos])
        for name in class_targets:
            if name in class_prob_raw:
                probs = class_prob_raw[name][sample_index]
                for class_index, prob in enumerate(probs):
                    row[f"{name}_p{class_index}"] = float(prob)
        worst.append(row)

    return metrics, worst


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Taiko encoder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/encoder_v0/best.pt"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/encoder_v0"))
    parser.add_argument("--stats", type=Path, default=Path("data/splits/encoder_v0/label_stats.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/encoder_v0"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(checkpoint)

    summary = {
        "checkpoint": str(args.checkpoint),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "splits": {},
    }
    for split in ["val", "test"]:
        metrics, worst = evaluate_split(
            model,
            config,
            args.split_dir / f"{split}.csv",
            args.stats,
            args.batch_size,
        )
        write_csv(args.output_dir / f"{split}_metrics.csv", metrics)
        write_csv(args.output_dir / f"{split}_worst.csv", worst)
        summary["splits"][split] = {
            "metrics_csv": str(args.output_dir / f"{split}_metrics.csv"),
            "worst_csv": str(args.output_dir / f"{split}_worst.csv"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
