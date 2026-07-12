from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from taiko_diffusion.data.ability_proxies import pooled_rhythm_proxy, proxy_values
from taiko_diffusion.data.tja import parse_tja_course


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/encoder_v2_multimodal_hands_v2_grouped"
OUTPUT = ROOT / "data/custom_abilities"
LABELS = ["custom_main", "custom_stamina", "custom_handspeed", "custom_burst", "custom_complex", "custom_rhythm"]


def conditional_quantile_targets(frame: pd.DataFrame, axis: str, teacher: str, neighbours: int = 80) -> np.ndarray:
    main = frame.v2_main.to_numpy(); proxy = frame[f"proxy_{axis}"].to_numpy(); values = frame[teacher].to_numpy(); output = np.zeros(len(frame))
    for index in range(len(frame)):
        nearest = np.argsort(np.abs(main - main[index]))[: min(neighbours, len(frame))]
        local_proxy = proxy[nearest]
        quantile = (np.sum(local_proxy < proxy[index]) + 0.5 * np.sum(local_proxy == proxy[index])) / len(nearest)
        output[index] = np.quantile(values[nearest], np.clip(quantile, 0.01, 0.99))
    return np.clip(output, 0, 15.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-rhythm", action="store_true", help="Reuse existing physical proxies and only refresh rhythm.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = pd.read_csv(ROOT / "data/manifests/v2_regression.csv").set_index("rating_index")
    teacher_names = {"stamina": "v2_stamina", "handspeed": "v2_handspeed", "burst": "v2_burst", "complex": "v2_complex", "rhythm": "v2_rhythm"}
    previous = None
    if args.refresh_rhythm:
        previous = pd.read_csv(output / "targets.csv").set_index("sample_id")
    rows = []
    processed = 0
    for split in ("train", "val", "test"):
        entries = pd.read_csv(CACHE / f"{split}.csv")
        for entry in entries.itertuples():
            rating_index = int(entry.sample_id.rsplit("_r", 1)[1]); teacher = manifest.loc[rating_index]
            with np.load(entry.npz_path) as data:
                parsed = parse_tja_course(Path(teacher.ese_path), teacher.ese_course)
                if previous is None:
                    proxies, diagnostics = proxy_values(parsed, data["chart"], data["audio"])
                else:
                    old = previous.loc[entry.sample_id]
                    rhythm, syncopation, family_mixing = pooled_rhythm_proxy(parsed, data["chart"], data["audio"])
                    proxies = {name: float(old[f"proxy_{name}"]) for name in teacher_names}
                    proxies["rhythm"] = rhythm
                    diagnostics = {name.removeprefix("diagnostic_"): float(old[name]) for name in previous.columns if name.startswith("diagnostic_")}
                    if "burst_peak_seconds" in diagnostics:
                        diagnostics["burst_peak_nps"] = diagnostics.pop("burst_peak_seconds")
                    diagnostics["rhythm_syncopation"] = syncopation
                    diagnostics["rhythm_family_mixing"] = family_mixing
            row = {"sample_id": entry.sample_id, "rating_index": rating_index, "title": teacher.title, "split": split,
                   "v2_main": float(teacher.v2_main), **{name: float(teacher[column]) for name, column in teacher_names.items()}}
            row.update({f"proxy_{name}": value for name, value in proxies.items()})
            diagnostic_values = diagnostics if isinstance(diagnostics, dict) else diagnostics.__dict__
            row.update({f"diagnostic_{name}": value for name, value in diagnostic_values.items()}); rows.append(row)
            processed += 1
            if processed == 1 or processed % 50 == 0:
                print(json.dumps({"processed": processed}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame["custom_main"] = frame.v2_main
    for axis, teacher in teacher_names.items():
        frame[f"custom_{axis}"] = conditional_quantile_targets(frame, axis, axis)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "targets.csv", index=False, encoding="utf-8-sig")
    summary = {"rows": len(frame), "labels": LABELS, "proxy_spearman": {axis: float(frame[f"proxy_{axis}"].corr(frame[axis], method="spearman")) for axis in teacher_names},
               "target_main_spearman": {axis: float(frame[f"custom_{axis}"].corr(frame.custom_main, method="spearman")) for axis in teacher_names}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
