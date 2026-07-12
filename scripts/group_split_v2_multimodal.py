from __future__ import annotations

import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/encoder_v2_multimodal"


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,default=CACHE);parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    source=args.source if args.source.is_absolute() else ROOT/args.source
    manifest = pd.read_csv(ROOT / "data/manifests/v2_regression.csv").set_index("rating_index")
    rows = pd.concat([pd.read_csv(source / f"{split}.csv") for split in ("train", "val", "test")], ignore_index=True)
    rows["rating_index"] = rows.sample_id.str.extract(r"_r(\d+)").astype(int)
    rows["group"] = [str(Path(manifest.loc[index, "ese_path"]).parent).casefold() for index in rows.rating_index]
    groups = sorted(rows.group.unique()); random.Random(20260619).shuffle(groups)
    train_end, val_end = int(len(groups) * 0.8), int(len(groups) * 0.9)
    assignment = {group: ("train" if i < train_end else "val" if i < val_end else "test") for i, group in enumerate(groups)}
    output = args.output if args.output else source.parent / f"{source.name}_grouped"
    if not output.is_absolute(): output=ROOT/output
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        selected = rows[rows.group.map(assignment) == split][["sample_id", "title", "npz_path"]]
        selected.to_csv(output / f"{split}.csv", index=False, encoding="utf-8-sig")
    train = pd.read_csv(output / "train.csv")
    y = np.stack([np.load(path)["y"] for path in train.npz_path])
    stats = {"mean": y.mean(0).tolist(), "std": np.maximum(y.std(0), 1e-6).tolist(),
             "groups": len(groups), "counts": {s: len(pd.read_csv(output / f"{s}.csv")) for s in ("train", "val", "test")}}
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
