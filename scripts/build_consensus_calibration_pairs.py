from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "data/calibration"
AXES = {
    "stamina": ("v2_stamina", "old_stamina"),
    "handspeed": ("v2_handspeed", "old_handspeed"),
    "burst": ("v2_burst", "old_burst"),
    "complex": ("v2_complex", "old_complex"),
    "rhythm": ("v2_rhythm", "old_rhythm"),
}


def robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    scale = (values - median).abs().median() * 1.4826
    return (values - median) / max(float(scale), 1e-6)


def residual(values: pd.Series, main: pd.Series) -> pd.Series:
    coefficients = np.polyfit(main.to_numpy(), values.to_numpy(), 2)
    return values - np.polyval(coefficients, main)


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/manifests/v2_regression.csv").set_index("rating_index")
    workbook = next(path for path in WORKSPACE.glob("*.xlsx") if "11.25" in path.name)
    old = pd.read_excel(workbook, sheet_name=1, header=1)
    old.index = np.arange(3, 3 + len(old))
    old = old.rename(
        columns={
            old.columns[3]: "old_complex",
            old.columns[4]: "old_stamina",
            old.columns[5]: "old_handspeed",
            old.columns[12]: "old_rhythm",
        }
    )
    old["old_burst"] = pd.to_numeric(old["old_handspeed"], errors="coerce") - 0.5 * pd.to_numeric(old["old_stamina"], errors="coerce")
    data = manifest.join(old[["old_complex", "old_stamina", "old_handspeed", "old_burst", "old_rhythm"]])
    split_by_index: dict[int, str] = {}
    split_root = ROOT / "data/cache/encoder_v2_multimodal_grouped"
    for split in ("train", "val", "test"):
        rows = pd.read_csv(split_root / f"{split}.csv")
        for value in rows.sample_id.str.extract(r"_r(\d+)")[0].astype(int):
            split_by_index[int(value)] = split
    data["split"] = [split_by_index.get(int(index), "") for index in data.index]
    numeric = ["v2_main", *[name for pair in AXES.values() for name in pair]]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric).copy()
    data["main_bin"] = (data.v2_main / 0.25).round().astype(int)

    pairs: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    summary: dict[str, object] = {"charts": len(data), "axes": {}}
    for axis, (v2_name, old_name) in AXES.items():
        v2_res = robust_z(residual(data[v2_name], data.v2_main))
        old_res = robust_z(residual(data[old_name], data.v2_main))
        data[f"{axis}_v2_z"] = v2_res
        data[f"{axis}_old_z"] = old_res
        data[f"{axis}_consensus"] = (v2_res + old_res) / 2
        axis_pairs = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        for (split, _), group in data.groupby(["split", "main_bin"]):
            if len(group) < 4:
                continue
            high = group.sort_values(f"{axis}_consensus", ascending=False).head(2)
            low = group.sort_values(f"{axis}_consensus").head(2)
            for (_, upper), (_, lower) in zip(high.iterrows(), low.iterrows()):
                v2_gap = float(upper[f"{axis}_v2_z"] - lower[f"{axis}_v2_z"])
                old_gap = float(upper[f"{axis}_old_z"] - lower[f"{axis}_old_z"])
                record = {
                    "axis": axis,
                    "split": split,
                    "higher_rating_index": int(upper.name),
                    "higher_title": upper.title,
                    "lower_rating_index": int(lower.name),
                    "lower_title": lower.title,
                    "higher_main": float(upper.v2_main),
                    "lower_main": float(lower.v2_main),
                    "v2_gap_z": v2_gap,
                    "old_gap_z": old_gap,
                    "confidence": float(min(v2_gap, old_gap)),
                }
                if abs(float(upper.v2_main - lower.v2_main)) <= 0.30 and v2_gap >= 1.25 and old_gap >= 1.25:
                    pairs.append(record); axis_pairs += 1; split_counts[split] += 1
        disagreement = np.sign(v2_res) != np.sign(old_res)
        severe = disagreement & ((v2_res - old_res).abs() >= 2.5)
        conflict_rows = data[severe].copy()
        conflict_rows["disagreement"] = (v2_res - old_res).abs()[severe]
        for index, row in conflict_rows.nlargest(30, "disagreement").iterrows():
            conflicts.append({"axis": axis, "rating_index": int(index), "title": row.title, "v2_main": float(row.v2_main),
                              "v2_residual_z": float(v2_res.loc[index]), "old_residual_z": float(old_res.loc[index])})
        summary["axes"][axis] = {"pairs": axis_pairs, "split_pairs": split_counts, "residual_spearman": float(v2_res.corr(old_res, method="spearman")), "severe_conflicts": int(severe.sum())}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pair_frame = pd.DataFrame(pairs).sort_values(["axis", "confidence"], ascending=[True, False])
    pair_frame.to_csv(OUTPUT / "consensus_pairs.csv", index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        pair_frame[pair_frame.split == split].to_csv(OUTPUT / f"consensus_pairs_{split}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conflicts).to_csv(OUTPUT / "conflicts.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
