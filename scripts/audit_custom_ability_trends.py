from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/encoder_custom_abilities_v1/trend_audit"
AXES = {
    "stamina": ("v2_stamina", "old_stamina"),
    "handspeed": ("v2_handspeed", "old_handspeed"),
    "burst": ("v2_burst", "old_burst"),
    "complex": ("v2_complex", "old_complex"),
    "rhythm": ("v2_rhythm", "old_rhythm"),
}


def residual(values: pd.Series, main: pd.Series) -> pd.Series:
    coefficients = np.polyfit(main.to_numpy(), values.to_numpy(), 2)
    return pd.Series(values.to_numpy() - np.polyval(coefficients, main), index=values.index)


def robust_z(values: pd.Series) -> pd.Series:
    center = values.median(); scale = (values - center).abs().median() * 1.4826
    return (values - center) / max(float(scale), 1e-6)


def grouped_trend(reference: pd.Series, custom: pd.Series) -> dict[str, object]:
    low_cut, high_cut = reference.quantile(0.10), reference.quantile(0.90)
    middle_distance = (reference - reference.median()).abs()
    groups = {
        "low": reference <= low_cut,
        "middle": middle_distance <= middle_distance.quantile(0.10),
        "high": reference >= high_cut,
    }
    medians = {name: float(custom[mask].median()) for name, mask in groups.items()}
    return {"custom_residual_z_median": medians, "ordered": medians["low"] < medians["middle"] < medians["high"],
            "high_low_gap": medians["high"] - medians["low"], "counts": {name: int(mask.sum()) for name, mask in groups.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "eval/encoder_custom_abilities_v1/predictions.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions).set_index("rating_index")
    manifest = pd.read_csv(ROOT / "data/manifests/v2_regression.csv").set_index("rating_index")
    workbook = next(path for path in ROOT.parent.glob("*.xlsx") if "11.25" in path.name)
    old = pd.read_excel(workbook, sheet_name=1, header=1); old.index = np.arange(3, 3 + len(old))
    old = old.rename(columns={old.columns[3]: "old_complex", old.columns[4]: "old_stamina", old.columns[5]: "old_handspeed", old.columns[12]: "old_rhythm"})
    old["old_burst"] = pd.to_numeric(old.old_handspeed, errors="coerce") - 0.5 * pd.to_numeric(old.old_stamina, errors="coerce")
    data = predictions[["title", "stamina", "handspeed", "burst", "complex", "rhythm"]].join(manifest[["v2_main", "v2_stamina", "v2_handspeed", "v2_burst", "v2_complex", "v2_rhythm"]]).join(old[["old_stamina", "old_handspeed", "old_burst", "old_complex", "old_rhythm"]])
    numeric = [column for column in data.columns if column != "title"]; data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce"); data = data.dropna()
    report: dict[str, object] = {"charts": len(data), "axes": {}}
    examples = []
    for axis, (v2_column, old_column) in AXES.items():
        custom_z = robust_z(residual(data[axis], data.v2_main))
        v2_z = robust_z(residual(data[v2_column], data.v2_main))
        old_z = robust_z(residual(data[old_column], data.v2_main))
        report["axes"][axis] = {
            "residual_spearman_v2": float(custom_z.corr(v2_z, method="spearman")),
            "residual_spearman_old": float(custom_z.corr(old_z, method="spearman")),
            "v2_groups": grouped_trend(v2_z, custom_z),
            "old_groups": grouped_trend(old_z, custom_z),
        }
        consensus = (v2_z + old_z) / 2
        for group_name, selected in {
            "low": consensus.nsmallest(5).index,
            "middle": (consensus - consensus.median()).abs().nsmallest(5).index,
            "high": consensus.nlargest(5).index,
        }.items():
            for index in selected:
                examples.append({"axis": axis, "group": group_name, "rating_index": int(index), "title": data.loc[index, "title"],
                                 "main": float(data.loc[index, "v2_main"]), "custom_residual_z": float(custom_z.loc[index]),
                                 "v2_residual_z": float(v2_z.loc[index]), "old_residual_z": float(old_z.loc[index])})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(examples).to_csv(args.output / "examples.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
