from __future__ import annotations

import json
import re
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import torch

from taiko_diffusion.models.v2_multimodal import V2TechniqueHeadEncoder


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/encoder_v2_multimodal_hands_v2_grouped"
CHECKPOINT = ROOT / "checkpoints/encoder_custom_abilities_v1/best.pt"
OUTPUT = ROOT / "eval/encoder_custom_abilities_v1"
LABELS = ["main", "stamina", "handspeed", "burst", "complex", "rhythm"]


def id_map() -> dict[int, tuple[int, int]]:
    workbook = next(path for path in ROOT.parent.glob("*.xlsx") if "11.25" in path.name)
    book = openpyxl.load_workbook(workbook, data_only=True, read_only=True)
    sheet = book["歌曲数据表"]
    result = {}
    for index, values in enumerate(sheet.iter_rows(min_row=3, max_col=15, values_only=True), start=3):
        match = re.search(r"/song/(\d+)-([45])/", str(values[14] or ""))
        if match: result[index] = (int(match.group(1)), int(match.group(2)))
    return result


def predictions(checkpoint_path: Path = CHECKPOINT) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels = checkpoint["channels"]
    model = V2TechniqueHeadEncoder(channels["chart"], channels["hand"], channels["audio"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
    rows = []
    with torch.no_grad():
        for split in ("train", "val", "test"):
            for entry in pd.read_csv(CACHE / f"{split}.csv").itertuples():
                with np.load(entry.npz_path) as data:
                    pred = model(torch.from_numpy(data["chart"])[None].to(device), torch.from_numpy(data["hand"])[None].to(device), torch.from_numpy(data["audio"])[None].to(device)).cpu().numpy()[0]
                raw = np.clip(pred * std + mean, 0.0, 15.5); rating_index = int(entry.sample_id.rsplit("_r", 1)[1])
                rows.append({"rating_index": rating_index, "title": entry.title, **dict(zip(LABELS, raw))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pred = predictions(args.checkpoint); ids = id_map(); pred[["song_id", "level"]] = pd.DataFrame([ids.get(index, (0, 0)) for index in pred.rating_index], index=pred.index)
    url = "https://viewer.sakura-bot.cn/api/taiko/data/grade_dojo_nijiiro_history_simple"
    with urllib.request.urlopen(url, timeout=60) as response: history = json.loads(response.read().decode("utf-8"))
    lookup = {(int(row.song_id), int(row.level)): row for row in pred.itertuples()}
    rows = []
    for year, version in history["versions"].items():
        for rank, grade in enumerate(version["grades"]):
            if rank < 10: continue
            values = []
            for song in grade["songs"]:
                level = 4 if song.get("difficulty") == "level4" else 5 if song.get("difficulty") == "level5" else 0
                row = lookup.get((int(song.get("id") or 0), level))
                if row: values.append(row)
            if values:
                record = {"year": year, "grade": grade["grade"], "rank": rank, "matched": len(values)}
                for label in LABELS: record[label] = float(np.mean([getattr(value, label) for value in values]))
                rows.append(record)
    edition = pd.DataFrame(rows)
    medians = edition.groupby(["grade", "rank"])[LABELS].median().reset_index().sort_values("rank")
    correlations = {label: float(edition["rank"].corr(edition[label], method="spearman")) for label in LABELS}
    # Remove the common main-rating trend and check for unexpected systematic Dan drift.
    residual_correlations = {}
    for label in LABELS[1:]:
        coefficients = np.polyfit(pred.main, pred[label], 2)
        lookup_residual = {int(row.rating_index): float(row[label] - np.polyval(coefficients, row.main)) for _, row in pred.iterrows()}
        edition_residual = []
        for _, item in edition.iterrows():
            # Edition rows do not retain song IDs; compare median axis residual
            # against the median predicted main curve instead.
            edition_residual.append(float(item[label] - np.polyval(coefficients, item["main"])))
        residual_correlations[label] = float(edition["rank"].corr(pd.Series(edition_residual, index=edition.index), method="spearman"))
    report = {"matched_editions": len(edition), "raw_spearman": correlations, "main_adjusted_spearman": residual_correlations,
              "grade_medians": medians.to_dict("records")}
    args.output.mkdir(parents=True, exist_ok=True);pred.to_csv(args.output/"predictions.csv",index=False,encoding="utf-8-sig");edition.to_csv(args.output/"dan_editions.csv",index=False,encoding="utf-8-sig");medians.to_csv(args.output/"dan_medians.csv",index=False,encoding="utf-8-sig");(args.output/"dan_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
