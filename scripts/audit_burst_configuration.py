from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from taiko_diffusion.data.ability_proxies import playable_notes
from taiko_diffusion.data.tja import parse_tja_course


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/encoder_custom_abilities_v3_physical/burst_configuration_audit"


def binary_entropy(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return float(-(value * np.log2(value) + (1.0 - value) * np.log2(1.0 - value)))


def conditional_entropy(colors: np.ndarray, order: int = 2) -> float:
    if len(colors) <= order:
        return 0.0
    contexts: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(order, len(colors)):
        contexts[tuple(int(value) for value in colors[index - order:index])].append(int(colors[index]))
    total = sum(len(values) for values in contexts.values())
    return float(sum(len(values) / total * binary_entropy(np.mean(values)) for values in contexts.values()))


def configuration_features(times: np.ndarray, colors: np.ndarray, bpm: np.ndarray) -> dict[str, float]:
    if len(times) < 2:
        return {name: 0.0 for name in ("peak_nps", "color_balance", "change_rate", "change_entropy", "context_entropy", "configuration", "string_notes", "string_seconds", "string_nps", "string_load", "pure_share")}
    half_window = 375.0
    counts = np.searchsorted(times, times + half_window, side="right") - np.searchsorted(times, times - half_window, side="left")
    peak = int(np.argmax(counts))
    peak_nps = float(counts[peak] / 0.75)
    left = right = peak
    while left > 0:
        gap_beats = (times[left] - times[left - 1]) * bpm[left] / 60_000.0
        if gap_beats > 0.55:
            break
        left -= 1
    while right + 1 < len(times):
        gap_beats = (times[right + 1] - times[right]) * bpm[right + 1] / 60_000.0
        if gap_beats > 0.55:
            break
        right += 1
    phrase_colors = colors[left:right + 1]
    ka_share = float(np.mean(phrase_colors))
    balance = 2.0 * min(ka_share, 1.0 - ka_share)
    change_rate = float(np.mean(phrase_colors[1:] != phrase_colors[:-1])) if len(phrase_colors) > 1 else 0.0
    change_ent = binary_entropy(change_rate)
    context_ent = conditional_entropy(phrase_colors)
    # Pure streams and deterministic alternation are low; balanced sequences
    # whose next colour is hard to predict are high.
    configuration = balance * (0.20 + 0.30 * change_ent + 0.50 * context_ent)
    seconds = max(float(times[right] - times[left]) / 1000.0, 0.05)
    string_nps = (right - left) / seconds if right > left else 0.0
    length_factor = min(np.log1p(len(phrase_colors)) / np.log(65.0), 1.25)
    string_load = configuration * length_factor * min(string_nps / 10.0, 1.5)
    return {
        "peak_nps": peak_nps,
        "color_balance": balance,
        "change_rate": change_rate,
        "change_entropy": change_ent,
        "context_entropy": context_ent,
        "configuration": configuration,
        "string_notes": float(len(phrase_colors)),
        "string_seconds": seconds,
        "string_nps": string_nps,
        "string_load": string_load,
        "pure_share": max(ka_share, 1.0 - ka_share),
    }


def residual(values: pd.Series, main: pd.Series) -> pd.Series:
    return values - np.polyval(np.polyfit(main, values, 2), main)


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/manifests/v2_regression.csv").set_index("rating_index")
    targets = pd.read_csv(ROOT / "data/custom_abilities_v3_physical/targets.csv").set_index("rating_index")
    rows = []
    for rating_index, item in manifest.iterrows():
        chart = parse_tja_course(Path(item.ese_path), item.ese_course)
        times, colors, bpm, _ = playable_notes(chart)
        rows.append({"rating_index": int(rating_index), "title": item.title, **configuration_features(times, colors, bpm)})
    features = pd.DataFrame(rows).set_index("rating_index")
    data = targets.join(features.drop(columns="title"), how="inner")
    target_residual = residual(data.burst, data.v2_main)
    base_residual = residual(data.proxy_burst, data.v2_main)
    split_train, split_test = data.split == "train", data.split == "test"

    grid = []
    for reward in np.linspace(0.0, 8.0, 33):
        for pure_discount in np.linspace(0.0, 0.5, 21):
            candidate = data.proxy_burst * (1.0 - pure_discount * (data.pure_share - 0.5).clip(lower=0.0) * 2.0) + reward * data.string_load
            candidate_residual = residual(candidate, data.v2_main)
            grid.append({
                "reward": float(reward),
                "pure_discount": float(pure_discount),
                "train_spearman": float(spearmanr(candidate_residual[split_train], target_residual[split_train]).statistic),
                "test_spearman": float(spearmanr(candidate_residual[split_test], target_residual[split_test]).statistic),
            })
    grid_frame = pd.DataFrame(grid)
    selected = grid_frame.sort_values("train_spearman", ascending=False).iloc[0]
    candidate = data.proxy_burst * (1.0 - selected.pure_discount * (data.pure_share - 0.5).clip(lower=0.0) * 2.0) + selected.reward * data.string_load
    candidate_residual = residual(candidate, data.v2_main)
    disagreement = target_residual - base_residual.rank(pct=True) * target_residual.std()
    report = {
        "charts": len(data),
        "base_residual_spearman": {
            "train": float(spearmanr(base_residual[split_train], target_residual[split_train]).statistic),
            "test": float(spearmanr(base_residual[split_test], target_residual[split_test]).statistic),
        },
        "feature_correlation_with_v2_residual": {
            name: float(spearmanr(data[name], target_residual).statistic)
            for name in ("color_balance", "change_rate", "context_entropy", "configuration", "string_notes", "string_nps", "string_load", "pure_share")
        },
        "selected_formula": selected.to_dict(),
        "selected_test_spearman": float(spearmanr(candidate_residual[split_test], target_residual[split_test]).statistic),
    }
    output = data[["title", "v2_main", "burst", "proxy_burst"]].copy()
    output["v2_burst_residual"] = target_residual
    output["base_residual"] = base_residual
    output["candidate"] = candidate
    output["candidate_residual"] = candidate_residual
    output = output.join(features.drop(columns="title"))
    output["residual_gap"] = target_residual - base_residual
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.sort_values("residual_gap", ascending=False).to_csv(OUTPUT / "charts.csv", encoding="utf-8-sig")
    grid_frame.to_csv(OUTPUT / "grid.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
