from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/calibration_physical"
AXES = ("burst", "complex", "rhythm")


def robust_z(values: pd.Series) -> pd.Series:
    center = float(values.median())
    scale = float((values - center).abs().median()) * 1.4826
    return (values - center) / max(scale, 1e-6)


def main() -> None:
    data = pd.read_csv(ROOT / "data/custom_abilities/targets.csv").set_index("rating_index")
    pairs = []
    summary: dict[str, object] = {"charts": len(data), "axes": {}}
    for axis in AXES:
        coefficients = np.polyfit(data.custom_main, data[f"custom_{axis}"], 2)
        data[f"{axis}_residual_z"] = robust_z(data[f"custom_{axis}"] - np.polyval(coefficients, data.custom_main))
    # A 0.4-wide main band is narrow enough to represent "same overall
    # difficulty", while leaving enough charts for reliable pair selection.
    data["main_bin"] = (data.custom_main / 0.4).round().astype(int)
    for axis in AXES:
        split_counts = {split: 0 for split in ("train", "val", "test")}
        for (split, _), group in data.groupby(["split", "main_bin"]):
            if len(group) < 4:
                continue
            ordered = group.sort_values(f"{axis}_residual_z")
            lows, highs = ordered.head(2), ordered.tail(2)
            for (_, lower), (_, higher) in zip(lows.iterrows(), highs.iloc[::-1].iterrows()):
                gap = float(higher[f"{axis}_residual_z"] - lower[f"{axis}_residual_z"])
                main_gap = abs(float(higher.custom_main - lower.custom_main))
                if gap < 1.0 or main_gap > 0.45:
                    continue
                pairs.append({
                    "axis": axis,
                    "split": split,
                    "higher_rating_index": int(higher.name),
                    "higher_title": higher.title,
                    "lower_rating_index": int(lower.name),
                    "lower_title": lower.title,
                    "higher_main": float(higher.custom_main),
                    "lower_main": float(lower.custom_main),
                    "confidence": min(gap, 3.0),
                })
                split_counts[str(split)] += 1
        summary["axes"][axis] = split_counts
    frame = pd.DataFrame(pairs).sort_values(["axis", "confidence"], ascending=[True, False])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "consensus_pairs.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        frame[frame.split == split].to_csv(OUTPUT / f"consensus_pairs_{split}.csv", index=False, encoding="utf-8-sig")
    summary["pairs"] = len(frame)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
