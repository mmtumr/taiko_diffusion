from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from taiko_diffusion.data.build_v2_multimodal_cache import pool_time
from taiko_diffusion.data.hand_techniques import TECHNIQUES, technique_tracks_from_grid


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/cache/encoder_v2_multimodal"
OUTPUT = ROOT / "data/cache/encoder_v2_multimodal_hands_v2"
CHART_SPLITS = ROOT / "data/splits/encoder_v2_regression"


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    chart_paths = {}
    for split in ("train", "val", "test"):
        chart_paths.update({row["sample_id"]: row["npz_path"] for row in read(CHART_SPLITS / f"{split}.csv")})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    costs = []
    for split in ("train", "val", "test"):
        rows = read(SOURCE / f"{split}.csv"); output_rows = []
        for row in rows:
            old = np.load(row["npz_path"]); chart = np.load(chart_paths[row["sample_id"]], allow_pickle=False)
            x = chart["x"].astype(np.float32); duration = int(chart["duration_frames"][0]); channels = [str(value) for value in chart["channels"]]
            tracks, technique_costs = technique_tracks_from_grid(x[:duration], channels)
            hand = pool_time(tracks, old["chart"].shape[1])
            path = OUTPUT / f"{row['sample_id']}.npz"
            np.savez_compressed(path, chart=old["chart"], hand=hand, audio=old["audio"], y=old["y"],
                                technique_costs=np.asarray([technique_costs[name] for name in TECHNIQUES], np.float32))
            output_rows.append({**row, "npz_path": str(path)})
            costs.append(technique_costs)
        with (OUTPUT / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer=csv.DictWriter(file,fieldnames=["sample_id","title","npz_path"]);writer.writeheader();writer.writerows(output_rows)
    stats=json.loads((SOURCE/"stats.json").read_text(encoding="utf-8"));stats["hand_techniques"]=list(TECHNIQUES);stats["hand_channels"]=48
    stats["mean_technique_cost"]={name:float(np.mean([row[name] for row in costs])) for name in TECHNIQUES}
    (OUTPUT/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"samples":len(costs),"mean_cost":stats["mean_technique_cost"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
