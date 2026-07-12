from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AXES = ["stamina", "handspeed", "burst", "complex", "rhythm"]


def residual(values: pd.Series, main: pd.Series) -> pd.Series:
    coefficients = np.polyfit(main.to_numpy(), values.to_numpy(), 2)
    return values - np.polyval(coefficients, main)


def robust_z(values: pd.Series) -> pd.Series:
    center = float(values.median())
    scale = float((values - center).abs().median()) * 1.4826
    return (values - center) / max(scale, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=Path("data/custom_abilities/targets.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions).set_index("rating_index")
    targets = pd.read_csv(args.targets).set_index("rating_index")
    common = predictions.index.intersection(targets.index)
    predictions, targets = predictions.loc[common], targets.loc[common]
    report: dict[str, object] = {"charts": len(common), "axes": {}}
    examples = []
    for axis in AXES:
        target_z = robust_z(residual(targets[f"custom_{axis}"], targets.custom_main))
        prediction_z = robust_z(residual(predictions[axis], predictions.main))
        low_cut, high_cut = target_z.quantile([0.10, 0.90])
        middle_distance = (target_z - target_z.median()).abs()
        groups = {
            "low": target_z <= low_cut,
            "middle": middle_distance <= middle_distance.quantile(0.10),
            "high": target_z >= high_cut,
        }
        medians = {name: float(prediction_z[mask].median()) for name, mask in groups.items()}
        report["axes"][axis] = {
            "residual_spearman": float(target_z.corr(prediction_z, method="spearman")),
            "prediction_residual_z_median": medians,
            "ordered": medians["low"] < medians["middle"] < medians["high"],
            "high_low_gap": medians["high"] - medians["low"],
        }
        selections = {
            "low": target_z.nsmallest(5).index,
            "middle": middle_distance.nsmallest(5).index,
            "high": target_z.nlargest(5).index,
        }
        for group, indices in selections.items():
            for rating_index in indices:
                examples.append({
                    "axis": axis,
                    "group": group,
                    "rating_index": int(rating_index),
                    "title": str(targets.loc[rating_index, "title"]),
                    "main": float(predictions.loc[rating_index, "main"]),
                    "target_residual_z": float(target_z.loc[rating_index]),
                    "prediction_residual_z": float(prediction_z.loc[rating_index]),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(examples).to_csv(args.output / "examples.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
