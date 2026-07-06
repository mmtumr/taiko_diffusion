from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from taiko_diffusion.data.build_v5_cache import load_label_map
from taiko_diffusion.data.build_v8_cache import direct_features
from taiko_diffusion.data.dataset import TaikoTensorDataset
from taiko_diffusion.eval_encoder import build_model, inverse_targets


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def source_npz_by_sample(index_path: Path) -> dict[str, str]:
    return {row["sample_id"]: row["npz_path"] for row in read_rows(index_path)}


def predict_split(
    checkpoint_path: Path,
    split_csv: Path,
    stats_path: Path,
    batch_size: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    model.eval()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    label_names = [str(name) for name in stats["label_names"]]
    dataset = TaikoTensorDataset(split_csv, stats_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_ids: list[str] = []
    titles: list[str] = []
    pred_parts: list[np.ndarray] = []
    class_parts: dict[str, list[np.ndarray]] = {}
    class_targets = checkpoint["config"].get("model", {}).get("class_targets", "")
    if isinstance(class_targets, str):
        class_target_names = [name.strip() for name in class_targets.split(",") if name.strip()]
    else:
        class_target_names = [str(name) for name in class_targets or []]
    for name in class_target_names:
        class_parts[name] = []

    with torch.no_grad():
        for batch in loader:
            output = model(batch["x"])
            pred = output["regression"] if isinstance(output, dict) else output
            pred_parts.append(pred.cpu().numpy())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            titles.extend(str(value) for value in batch["title"])
            if isinstance(output, dict) and "class_logits" in output:
                for name in class_target_names:
                    probs = torch.softmax(output["class_logits"][name], dim=1).cpu().numpy()
                    class_parts[name].append(probs)

    pred_raw = inverse_targets(np.vstack(pred_parts), stats)
    class_probs = {name: np.vstack(parts) for name, parts in class_parts.items() if parts}
    return sample_ids, titles, label_names, pred_raw, class_probs


def physical_stats(sample_id: str, source_index: dict[str, str], frame_ms: float) -> dict[str, float]:
    source_path = source_index[sample_id]
    source = np.load(source_path, allow_pickle=False)
    x = source["x"].astype(np.float32)
    channels = [str(name) for name in source["channels"]]
    labels = load_label_map(source_path)
    return direct_features(x, channels, labels, frame_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final Taiko condition JSONL from cached splits.")
    parser.add_argument("--main-checkpoint", type=Path, default=Path("checkpoints/encoder_final_main/best.pt"))
    parser.add_argument("--main-split", type=Path, default=Path("data/splits/encoder_final_main/test.csv"))
    parser.add_argument("--main-stats", type=Path, default=Path("data/splits/encoder_final_main/label_stats.json"))
    parser.add_argument(
        "--note-checkpoint",
        type=Path,
        default=Path("checkpoints/encoder_v8_note_type_log1p_handtiming_solo/best.pt"),
    )
    parser.add_argument(
        "--note-split",
        type=Path,
        default=Path("data/splits/encoder_v8_note_type_log1p_handtiming_solo/test.csv"),
    )
    parser.add_argument(
        "--note-stats",
        type=Path,
        default=Path("data/splits/encoder_v8_note_type_log1p_handtiming_solo/label_stats.json"),
    )
    parser.add_argument("--source-index", type=Path, default=Path("data/cache/encoder_v1/index.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/final_conditions_test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--note-high-threshold", type=float, default=25.0)
    args = parser.parse_args()

    main_ids, main_titles, main_labels, main_pred, main_class_probs = predict_split(
        args.main_checkpoint,
        args.main_split,
        args.main_stats,
        args.batch_size,
    )
    note_ids, _, note_labels, note_pred, _ = predict_split(
        args.note_checkpoint,
        args.note_split,
        args.note_stats,
        args.batch_size,
    )
    note_by_id = {
        sample_id: float(note_pred[index, note_labels.index("note_type")])
        for index, sample_id in enumerate(note_ids)
    }
    source_index = source_npz_by_sample(args.source_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for index, sample_id in enumerate(main_ids):
            row = {
                "sample_id": sample_id,
                "title": main_titles[index],
                "physical": physical_stats(sample_id, source_index, args.frame_ms),
                "main_style": {},
                "note_style": {},
            }
            for label_index, label in enumerate(main_labels):
                if label == "bpm_rhythm_bin":
                    continue
                row["main_style"][label] = float(main_pred[index, label_index])
            if "bpm_rhythm_bin" in main_class_probs:
                probs = main_class_probs["bpm_rhythm_bin"][index]
                row["main_style"]["bpm_rhythm_bin"] = int(np.argmax(probs))
                row["main_style"]["bpm_rhythm_probs"] = [float(value) for value in probs]
            note_type = note_by_id[sample_id]
            row["note_style"]["note_type"] = note_type
            row["note_style"]["note_type_high"] = bool(note_type >= args.note_high_threshold)
            row["note_style"]["note_type_high_threshold"] = float(args.note_high_threshold)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(args.output), "rows": len(main_ids)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
