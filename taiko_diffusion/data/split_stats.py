from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from taiko_diffusion.config import load_config


LOG1P_LABELS = {"combo", "roll_time", "balloon_num"}
LOGIT100_EPS = 0.5


def logit100(value: float) -> float:
    clipped = min(max(float(value), 0.0), 100.0)
    probability = (clipped + LOGIT100_EPS) / (100.0 + 2.0 * LOGIT100_EPS)
    return float(np.log(probability / (1.0 - probability)))


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transform_y(y: np.ndarray, label_names: list[str]) -> np.ndarray:
    y = y.astype(np.float32).copy()
    for index, name in enumerate(label_names):
        if name in LOG1P_LABELS:
            y[index] = np.log1p(max(y[index], 0.0))
    return y


def load_y(row: dict[str, str], label_names: list[str]) -> np.ndarray:
    data = np.load(row["npz_path"], allow_pickle=False)
    stored_labels = [str(x) for x in data["label_names"]]
    y = data["y"].astype(np.float32)
    if stored_labels == label_names:
        return y
    values = {name: y[index] for index, name in enumerate(stored_labels)}
    return np.asarray([values[name] for name in label_names], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic splits and label normalization stats.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder_v0.yaml"))
    parser.add_argument("--index", type=Path, default=Path("data/cache/encoder_v0/index.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config is not None else {}
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    output_dir = args.output_dir or Path(training_config.get("split_dir", "data/splits/encoder_v0"))
    seed = args.seed if args.seed is not None else int(training_config.get("seed", 20260619))
    label_names = list(data_config.get("target_columns", []))
    log1p_labels = set(LOG1P_LABELS) | {str(name) for name in data_config.get("log1p_labels", [])}
    logit100_labels = {str(name) for name in data_config.get("logit100_labels", [])}

    rows = read_index(args.index)
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: row["sample_id"])
    rng.shuffle(rows)

    n = len(rows)
    train_n = int(n * args.train_ratio)
    val_n = int(n * args.val_ratio)
    train_rows = rows[:train_n]
    val_rows = rows[train_n : train_n + val_n]
    test_rows = rows[train_n + val_n :]

    write_rows(output_dir / "train.csv", train_rows)
    write_rows(output_dir / "val.csv", val_rows)
    write_rows(output_dir / "test.csv", test_rows)

    def transform_with_config(y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32).copy()
        for index, name in enumerate(label_names):
            if name in log1p_labels:
                y[index] = np.log1p(max(y[index], 0.0))
            if name in logit100_labels:
                y[index] = logit100(float(y[index]))
        return y

    train_y = np.stack([transform_with_config(load_y(row, label_names)) for row in train_rows])
    mean = train_y.mean(axis=0)
    std = train_y.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    raw_train_y = np.stack([load_y(row, label_names) for row in train_rows])

    stats = {
        "label_names": label_names,
        "transforms": {
            name: (
                "logit100"
                if name in logit100_labels
                else ("log1p" if name in log1p_labels else "identity")
            )
            for name in label_names
        },
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "raw_min": raw_train_y.min(axis=0).astype(float).tolist(),
        "raw_max": raw_train_y.max(axis=0).astype(float).tolist(),
        "split": {
            "seed": seed,
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "index": str(args.index),
        "output_dir": str(output_dir),
        "rows": n,
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "label_names": label_names,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
