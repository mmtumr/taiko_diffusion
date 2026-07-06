from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def frame_to_char(values: np.ndarray, names: list[str]) -> str:
    channel = {name: index for index, name in enumerate(names)}
    if "note_event" in channel:
        if values[channel["note_event"]] <= 0.5:
            return "0"
        ka_value = values[channel["ka_probability"]] if "ka_probability" in channel else 0.0
        return "2" if ka_value > 0.5 else "1"
    if values[channel["roll_end"]] > 0.5 or values[channel["balloon_end"]] > 0.5:
        return "8"
    if values[channel["balloon_start"]] > 0.5:
        return "7"
    if values[channel["roll_start"]] > 0.5:
        return "5"
    if values[channel["ka"]] > 0.5 and values[channel["ka"]] >= values[channel["don"]]:
        return "2"
    if values[channel["don"]] > 0.5:
        return "1"
    return "0"


def density_topk_binary(
    probability: np.ndarray,
    names: list[str],
    data: np.lib.npyio.NpzFile,
    frame_ms: float,
    ka_ratio: float | None,
    onset_mix: float,
) -> np.ndarray:
    condition_names = [str(name) for name in data["condition_names"]]
    raw_condition = data["raw_condition"].astype(np.float32)
    condition = {name: float(raw_condition[index]) for index, name in enumerate(condition_names)}
    avg_density = max(condition.get("avg_density", 0.0), 0.0)
    window_seconds = probability.shape[0] * frame_ms / 1000.0
    note_count = int(round(avg_density * window_seconds))
    channel = {name: index for index, name in enumerate(names)}
    if ka_ratio is None and "ka_ratio" in condition:
        ka_ratio = condition["ka_ratio"]

    if "note_event" in channel:
        note_score = probability[:, channel["note_event"]]
    else:
        note_score = np.maximum(probability[:, channel["don"]], probability[:, channel["ka"]])
    if onset_mix > 0.0 and "audio" in data.files and data["audio"].size:
        audio = data["audio"].astype(np.float32)
        onset = audio[-2] if audio.shape[0] < audio.shape[1] else audio[:, -2]
        if onset.shape[0] != note_score.shape[0]:
            x_old = np.linspace(0.0, 1.0, num=onset.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, num=note_score.shape[0], dtype=np.float32)
            onset = np.interp(x_new, x_old, onset).astype(np.float32)
        onset = onset - float(onset.min())
        onset = onset / max(float(onset.max()), 1e-6)
        note_score = note_score + float(onset_mix) * onset
    note_count = max(0, min(note_count, probability.shape[0]))
    selected = np.zeros(probability.shape[0], dtype=bool)
    if note_count > 0:
        selected[np.argpartition(note_score, -note_count)[-note_count:]] = True
    binary = np.zeros_like(probability, dtype=np.float32)
    if "note_event" in channel:
        binary[selected, channel["note_event"]] = 1.0
        if "ka_probability" in channel:
            if ka_ratio is None:
                binary[selected, channel["ka_probability"]] = (
                    probability[selected, channel["ka_probability"]] > 0.5
                ).astype(np.float32)
            else:
                selected_indices = np.where(selected)[0]
                ka_count = int(round(len(selected_indices) * min(max(ka_ratio, 0.0), 1.0)))
                if ka_count > 0:
                    scores = probability[selected_indices, channel["ka_probability"]]
                    ka_indices = selected_indices[np.argpartition(scores, -ka_count)[-ka_count:]]
                    binary[ka_indices, channel["ka_probability"]] = 1.0
    else:
        don_is_stronger = probability[:, channel["don"]] >= probability[:, channel["ka"]]
        binary[selected & don_is_stronger, channel["don"]] = 1.0
        binary[selected & ~don_is_stronger, channel["ka"]] = 1.0
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a sampled 8-track chart window to a rough TJA file.")
    parser.add_argument("--sample", type=Path, default=Path("eval/diffusion_v0_sample.npz"))
    parser.add_argument("--output", type=Path, default=Path("eval/diffusion_v0_sample.tja"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--frame-ms", type=float, default=46.4399)
    parser.add_argument("--density-topk", action="store_true")
    parser.add_argument("--ka-ratio", type=float, default=None)
    parser.add_argument("--onset-mix", type=float, default=0.0)
    args = parser.parse_args()

    data = np.load(args.sample, allow_pickle=False)
    probability = data["probability"].astype(np.float32)
    names = [str(name) for name in data["target_channels"]]
    if args.density_topk:
        binary = density_topk_binary(
            probability,
            names,
            data,
            float(args.frame_ms),
            args.ka_ratio,
            float(args.onset_mix),
        )
    else:
        binary = (probability >= float(args.threshold)).astype(np.float32)
    frames_per_measure = max(1, int(round((4.0 * 60000.0 / float(args.bpm)) / float(args.frame_ms))))
    chars = [frame_to_char(binary[index], names) for index in range(binary.shape[0])]
    measures = [
        "".join(chars[start : start + frames_per_measure]) + ","
        for start in range(0, len(chars), frames_per_measure)
    ]
    title = str(data["source_title"][0]) if "source_title" in data.files else "diffusion_v0_sample"
    text = "\n".join(
        [
            f"TITLE:{title} diffusion_sample",
            f"BPM:{float(args.bpm):.6g}",
            "COURSE:Oni",
            "LEVEL:10",
            "#START",
            *measures,
            "#END",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print({"output": str(args.output), "measures": len(measures), "frames_per_measure": frames_per_measure})


if __name__ == "__main__":
    main()
