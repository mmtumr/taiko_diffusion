from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
V2_PATH = WORKSPACE / "taiko_rating" / "data" / "v2_constants.json"
OUTPUT = ROOT / "data" / "manifests" / "v2_regression.csv"


def workbook_path() -> Path:
    return next(path for path in WORKSPACE.glob("*.xlsx") if "11.25" in path.name)


def rating_ids() -> dict[int, tuple[int, int]]:
    book = openpyxl.load_workbook(workbook_path(), data_only=True, read_only=True)
    sheet = book["歌曲数据表"]
    result: dict[int, tuple[int, int]] = {}
    for row_index, values in enumerate(sheet.iter_rows(min_row=3, max_col=15, values_only=True), start=3):
        match = re.search(r"/song/(\d+)-([45])/", str(values[14] or ""))
        if match:
            result[row_index] = (int(match.group(1)), int(match.group(2)))
    return result


def main() -> None:
    split_dir = ROOT / "data" / "splits" / "encoder_final_main"
    source = pd.concat([pd.read_csv(path) for path in split_dir.glob("*.csv")], ignore_index=True)
    source["rating_index"] = source["sample_id"].str.extract(r"_r(\d+)").astype(int)
    ids = rating_ids()
    v2 = {(int(row["id"]), int(row["level"])): row for row in json.loads(V2_PATH.read_text(encoding="utf-8"))}
    rows: list[dict[str, object]] = []
    for record in source.to_dict("records"):
        key = ids.get(int(record["rating_index"]))
        constants = v2.get(key) if key else None
        if not constants:
            continue
        rows.append(
            {
                "rating_index": int(record["rating_index"]),
                "title": record["title"],
                "ese_path": record["ese_path"],
                "ese_course": record["course"],
                "ese_note_count": int(record["parsed_note_count"]),
                "combo": int(record["combo"]),
                "v2_main": constants["main"],
                "v2_stamina": constants["stamina"],
                "v2_handspeed": constants["handspeed"],
                "v2_burst": constants["burst"],
                "v2_complex": constants["complex"],
                "v2_rhythm": constants["rhythm"],
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("rating_index").to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(json.dumps({"source": len(source), "matched": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
