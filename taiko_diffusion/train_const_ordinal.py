from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from taiko_diffusion.models.encoder import AttentionPool, DownsampleBlock


COURSES = ["Normal", "Hard", "Oni", "Edit"]
BIN_EDGES = np.asarray([4.0, 6.0, 8.0, 10.0], dtype=np.float32)
BIN_CENTERS = torch.tensor([2.5, 5.0, 7.0, 9.0, 10.9], dtype=torch.float32)
BIN_RADII = torch.tensor([1.5, 1.0, 1.0, 1.0, 0.9], dtype=torch.float32)
LOW_DENSITY_AVG_THRESHOLD = 3.4
LOW_DENSITY_PEAK_THRESHOLD = 8.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def const_bin(value: float) -> int:
    return int(np.searchsorted(BIN_EDGES, float(value), side="right"))


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


def load_label_map(npz_path: Path) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    labels = [str(name) for name in data["label_names"]]
    return {name: float(data["y"][index]) for index, name in enumerate(labels)}


def physical_features(row: dict[str, str], data: np.lib.npyio.NpzFile, frame_ms: float) -> np.ndarray:
    x = data["x"].astype(np.float32)
    channels = [str(name) for name in data["channels"]]
    channel_index = {name: index for index, name in enumerate(channels)}

    def channel(name: str) -> np.ndarray:
        index = channel_index.get(name)
        if index is None:
            return np.zeros(x.shape[0], dtype=np.float32)
        return x[:, index]

    note = np.clip(channel("don") + channel("ka"), 0.0, 1.0)
    note_frames = note > 0
    combo = float(row.get("combo") or row.get("parsed_note_count") or note_frames.sum() or 0.0)
    duration_frames = float(row.get("duration_frames") or data["duration_frames"][0])
    duration_seconds = max(duration_frames * frame_ms / 1000.0, frame_ms / 1000.0)
    active = channel("active") > 0 if "active" in channel_index else np.ones(x.shape[0], dtype=bool)
    active_seconds = max(float(active.sum()) * frame_ms / 1000.0, frame_ms / 1000.0)
    window = max(int(round(1000.0 / frame_ms)), 1)
    peak_notes = float(
        np.convolve(note_frames.astype(np.float32), np.ones(window, dtype=np.float32), mode="same").max()
    )

    bpm_events = float(channel("bpm_change_event").sum())
    scroll_events = float(channel("scroll_change_event").sum())
    color_events = float(channel("color_change_event").sum())
    hand_density = channel("hand_change_total_density")
    half_hand_density = channel("half_hand_change_total_density")
    short_density = channel("note_density_short")
    long_density = channel("note_density_long")

    values = [
        math.log1p(combo),
        math.log1p(duration_seconds),
        combo / active_seconds,
        peak_notes,
        float(short_density.mean()),
        float(short_density.max()),
        float(long_density.mean()),
        math.log1p(bpm_events),
        math.log1p(scroll_events),
        math.log1p(color_events),
        float(hand_density.mean()),
        float(hand_density.max()),
        float(half_hand_density.mean()),
        float(half_hand_density.max()),
    ]
    return np.asarray(values, dtype=np.float32)


def physical_stats(rows: list[dict[str, str]], root: Path, frame_ms: float) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    for row in rows:
        data = np.load(root / row["npz_path"], allow_pickle=True)
        values.append(physical_features(row, data, frame_ms))
    matrix = np.vstack(values).astype(np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean, std


class ConstOrdinalDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        root: Path,
        phys_mean: np.ndarray,
        phys_std: np.ndarray,
        frame_ms: float,
    ):
        self.rows = read_rows(split_csv)
        self.root = root
        self.phys_mean = phys_mean.astype(np.float32)
        self.phys_std = phys_std.astype(np.float32)
        self.frame_ms = float(frame_ms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        data = np.load(self.root / row["npz_path"], allow_pickle=True)
        x = data["x"].astype(np.float32)
        labels = load_label_map(self.root / row["npz_path"])
        const = float(labels["const"])
        bin_index = const_bin(const)
        course = str(row.get("course") or "")
        course_index = COURSES.index(course) if course in COURSES else 0
        phys = (physical_features(row, data, self.frame_ms) - self.phys_mean) / self.phys_std
        raw_phys = physical_features(row, data, self.frame_ms)
        low_density = (
            raw_phys[2] <= LOW_DENSITY_AVG_THRESHOLD
            and raw_phys[3] <= LOW_DENSITY_PEAK_THRESHOLD
        )
        return {
            "x": torch.from_numpy(x.transpose(1, 0)),
            "phys": torch.from_numpy(phys),
            "course": torch.tensor(course_index, dtype=torch.long),
            "const": torch.tensor(const, dtype=torch.float32),
            "const_bin": torch.tensor(bin_index, dtype=torch.long),
            "complex": torch.tensor(float(labels["complex"]) / 100.0, dtype=torch.float32),
            "hs_change": torch.tensor(float(labels["hs_change"]) / 100.0, dtype=torch.float32),
            "bpm_rhythm_bin": torch.tensor(int(round(float(labels["bpm_rhythm_bin"]))), dtype=torch.long),
            "low_aux": torch.tensor(
                [const <= 5.0, const <= 6.0, low_density],
                dtype=torch.float32,
            ),
            "sample_id": row["sample_id"],
            "title": row["title"],
            "course_name": course,
        }


class ConstOrdinalModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        phys_dim: int,
        conv_channels: int = 96,
        latent_dim: int = 192,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.stem = nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3)
        self.down = nn.Sequential(*[DownsampleBlock(conv_channels, dropout) for _ in range(4)])
        layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=8,
            dim_feedforward=conv_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = AttentionPool(conv_channels)
        self.phys_mlp = nn.Sequential(
            nn.LayerNorm(phys_dim),
            nn.Linear(phys_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 48),
            nn.GELU(),
        )
        self.course_embedding = nn.Embedding(len(COURSES), 12)
        fused_dim = conv_channels + 48 + 12
        self.fused = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.bin_head = nn.Linear(latent_dim, 5)
        self.residual_head = nn.Linear(latent_dim, 5)
        self.aux_head = nn.Linear(latent_dim, 2)
        self.bpm_head = nn.Linear(latent_dim, 3)
        self.low_head = nn.Linear(latent_dim, 3)

    def forward(self, x: torch.Tensor, phys: torch.Tensor, course: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.down(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        chart = self.pool(x)
        fused = torch.cat([chart, self.phys_mlp(phys), self.course_embedding(course)], dim=1)
        latent = self.fused(fused)
        bin_logits = self.bin_head(latent)
        centers = BIN_CENTERS.to(latent.device).unsqueeze(0)
        radii = BIN_RADII.to(latent.device).unsqueeze(0)
        const_by_bin = centers + torch.tanh(self.residual_head(latent)) * radii
        probs = torch.softmax(bin_logits, dim=1)
        const_pred = torch.sum(probs * const_by_bin, dim=1)
        mode_bin = torch.argmax(bin_logits, dim=1)
        mode_const = const_by_bin.gather(1, mode_bin[:, None]).squeeze(1)
        return {
            "bin_logits": bin_logits,
            "const_by_bin": const_by_bin,
            "const_pred": const_pred,
            "mode_const": mode_const,
            "aux": torch.sigmoid(self.aux_head(latent)),
            "bpm_logits": self.bpm_head(latent),
            "low_logits": self.low_head(latent),
        }


def low_aux_pos_weights(rows: list[dict[str, str]], root: Path, frame_ms: float) -> torch.Tensor:
    positives = np.zeros(3, dtype=np.float32)
    total = 0.0
    for row in rows:
        data = np.load(root / row["npz_path"], allow_pickle=True)
        labels = load_label_map(root / row["npz_path"])
        const = float(labels["const"])
        raw_phys = physical_features(row, data, frame_ms)
        low_density = (
            raw_phys[2] <= LOW_DENSITY_AVG_THRESHOLD
            and raw_phys[3] <= LOW_DENSITY_PEAK_THRESHOLD
        )
        positives += np.asarray([const <= 5.0, const <= 6.0, low_density], dtype=np.float32)
        total += 1.0
    negatives = np.maximum(total - positives, 1.0)
    positives = np.maximum(positives, 1.0)
    return torch.tensor(np.sqrt(negatives / positives), dtype=torch.float32)


def class_weights(rows: list[dict[str, str]], root: Path) -> torch.Tensor:
    counts = np.zeros(5, dtype=np.float32)
    for row in rows:
        labels = load_label_map(root / row["npz_path"])
        counts[const_bin(labels["const"])] += 1.0
    weights = np.sqrt(np.maximum(counts.max(), 1.0) / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: ConstOrdinalModel,
    loader: DataLoader,
    device: torch.device,
    bin_weights: torch.Tensor,
    low_pos_weights: torch.Tensor,
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        phys = batch["phys"].to(device, non_blocking=True)
        course = batch["course"].to(device, non_blocking=True)
        const = batch["const"].to(device, non_blocking=True)
        const_bin_target = batch["const_bin"].to(device, non_blocking=True)
        aux_target = torch.stack(
            [
                batch["complex"].to(device, non_blocking=True),
                batch["hs_change"].to(device, non_blocking=True),
            ],
            dim=1,
        )
        bpm_target = batch["bpm_rhythm_bin"].to(device, non_blocking=True).clamp(0, 2)
        low_aux_target = batch["low_aux"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            output = model(x, phys, course)
            selected_const = output["const_by_bin"].gather(1, const_bin_target[:, None]).squeeze(1)
            const_loss = torch.nn.functional.smooth_l1_loss(output["const_pred"], const)
            selected_loss = torch.nn.functional.smooth_l1_loss(selected_const, const)
            bin_loss = torch.nn.functional.cross_entropy(
                output["bin_logits"],
                const_bin_target,
                weight=bin_weights,
            )
            aux_loss = torch.nn.functional.smooth_l1_loss(output["aux"], aux_target)
            bpm_loss = torch.nn.functional.cross_entropy(output["bpm_logits"], bpm_target)
            low_aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["low_logits"],
                low_aux_target,
                pos_weight=low_pos_weights,
            )
            loss = (
                float(loss_weights["const"]) * const_loss
                + float(loss_weights["selected"]) * selected_loss
                + float(loss_weights["bin"]) * bin_loss
                + float(loss_weights["aux"]) * aux_loss
                + float(loss_weights["bpm"]) * bpm_loss
                + float(loss_weights["low_aux"]) * low_aux_loss
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.size(0)
        total_count += x.size(0)
    return total_loss / max(total_count, 1)


def evaluate(model: ConstOrdinalModel, loader: DataLoader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            phys = batch["phys"].to(device, non_blocking=True)
            course = batch["course"].to(device, non_blocking=True)
            output = model(x, phys, course)
            probs = torch.softmax(output["bin_logits"], dim=1).cpu().numpy()
            low_probs = torch.sigmoid(output["low_logits"]).cpu().numpy()
            pred = output["const_pred"].cpu().numpy()
            mode_pred = output["mode_const"].cpu().numpy()
            pred_bin = np.argmax(probs, axis=1)
            true = batch["const"].cpu().numpy()
            true_bin = batch["const_bin"].cpu().numpy()
            for index in range(len(true)):
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "title": str(batch["title"][index]),
                        "course": str(batch["course_name"][index]),
                        "const_true": float(true[index]),
                        "const_pred": float(pred[index]),
                        "mode_const_pred": float(mode_pred[index]),
                        "error": float(pred[index] - true[index]),
                        "const_bin_true": int(true_bin[index]),
                        "const_bin_pred": int(pred_bin[index]),
                        "bin_prob": [float(value) for value in probs[index]],
                        "low_prob": [float(value) for value in low_probs[index]],
                    }
                )
    return rows


def metric_summary(rows: list[dict[str, object]], pred_key: str = "const_pred") -> dict[str, object]:
    true = np.asarray([float(row["const_true"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    error = pred - true
    result: dict[str, object] = {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "pearson": corr(true, pred),
        "spearman": spearman(true, pred),
        "bin_accuracy": float(
            np.mean(
                [
                    int(row["const_bin_true"]) == int(row["const_bin_pred"])
                    for row in rows
                ]
            )
        ),
    }
    segments: dict[str, object] = {}
    for threshold in [4.0, 5.0, 6.0, 7.0, 8.0]:
        indices = true <= threshold
        if not np.any(indices):
            continue
        seg_error = error[indices]
        segments[f"<= {threshold:g}"] = {
            "n": int(indices.sum()),
            "mae": float(np.mean(np.abs(seg_error))),
            "bias": float(np.mean(seg_error)),
            "over2": int(np.sum(seg_error >= 2.0)),
            "over3": int(np.sum(seg_error >= 3.0)),
        }
    result["segments"] = segments
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows: list[dict[str, object]] = []
    for row in rows:
        flat = dict(row)
        probs = flat.pop("bin_prob")
        for index, value in enumerate(probs):
            flat[f"bin_prob_{index}"] = value
        low_probs = flat.pop("low_prob", [])
        for index, value in enumerate(low_probs):
            flat[f"low_prob_{index}"] = value
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an experimental ordinal const encoder.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/encoder_final_main"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/const_ordinal_experiment"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/const_ordinal_experiment"))
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--const-loss-weight", type=float, default=1.0)
    parser.add_argument("--selected-loss-weight", type=float, default=0.45)
    parser.add_argument("--bin-loss-weight", type=float, default=0.35)
    parser.add_argument("--aux-loss-weight", type=float, default=0.12)
    parser.add_argument("--bpm-loss-weight", type=float, default=0.25)
    parser.add_argument("--low-aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--low-bin-weight-scale", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(".")
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    train_rows = read_rows(args.split_dir / "train.csv")
    phys_mean, phys_std = physical_stats(train_rows, root, args.frame_ms)
    train_dataset = ConstOrdinalDataset(args.split_dir / "train.csv", root, phys_mean, phys_std, args.frame_ms)
    val_dataset = ConstOrdinalDataset(args.split_dir / "val.csv", root, phys_mean, phys_std, args.frame_ms)
    test_dataset = ConstOrdinalDataset(args.split_dir / "test.csv", root, phys_mean, phys_std, args.frame_ms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    first = np.load(root / train_rows[0]["npz_path"], allow_pickle=True)
    model = ConstOrdinalModel(
        input_channels=int(first["x"].shape[1]),
        phys_dim=int(len(phys_mean)),
    ).to(device)
    bin_weights = class_weights(train_rows, root)
    if args.low_bin_weight_scale != 1.0:
        bin_weights[:2] *= float(args.low_bin_weight_scale)
        bin_weights = bin_weights / bin_weights.mean()
    bin_weights = bin_weights.to(device)
    low_pos_weights = low_aux_pos_weights(train_rows, root, args.frame_ms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.00015)
    loss_weights = {
        "const": float(args.const_loss_weight),
        "selected": float(args.selected_loss_weight),
        "bin": float(args.bin_loss_weight),
        "aux": float(args.aux_loss_weight),
        "bpm": float(args.bpm_loss_weight),
        "low_aux": float(args.low_aux_loss_weight),
    }

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, bin_weights, low_pos_weights, loss_weights, optimizer)
        val_loss = run_epoch(model, val_loader, device, bin_weights, low_pos_weights, loss_weights)
        val_rows = evaluate(model, val_loader, device)
        val_metrics = metric_summary(val_rows)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_const_mae": val_metrics["mae"],
            "val_bin_accuracy": val_metrics["bin_accuracy"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if float(val_metrics["mae"]) < best_mae - 0.001:
            best_mae = float(val_metrics["mae"])
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_const_mae": best_mae,
                    "phys_mean": phys_mean,
                    "phys_std": phys_std,
                    "bin_weights": bin_weights.detach().cpu().numpy(),
                    "low_pos_weights": low_pos_weights.detach().cpu().numpy(),
                    "args": vars(args),
                },
                args.checkpoint_dir / "best.pt",
            )
        if epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch, "best_val_const_mae": best_mae}, ensure_ascii=False))
            break

    checkpoint = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    all_metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint_dir / "best.pt"),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_const_mae": float(checkpoint["val_const_mae"]),
        "phys_mean": [float(value) for value in phys_mean],
        "phys_std": [float(value) for value in phys_std],
        "bin_weights": [float(value) for value in checkpoint["bin_weights"]],
        "low_pos_weights": [float(value) for value in checkpoint["low_pos_weights"]],
        "loss_weights": loss_weights,
        "splits": {},
    }
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        rows = evaluate(model, loader, device)
        write_csv(args.output_dir / f"{split}_predictions.csv", rows)
        worst = sorted(rows, key=lambda row: abs(float(row["error"])), reverse=True)[:20]
        write_csv(args.output_dir / f"{split}_worst.csv", worst)
        all_metrics["splits"][split] = metric_summary(rows)

    (args.checkpoint_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
