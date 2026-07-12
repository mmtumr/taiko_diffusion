from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def frame_to_char(values: np.ndarray, names: list[str]) -> str:
    channel = {name: index for index, name in enumerate(names)}
    if "hold_end" in channel and values[channel["hold_end"]] > 0.5:
        return "8"
    if "hold_start" in channel and values[channel["hold_start"]] > 0.5:
        return "5"
    if "note_event" in channel:
        if values[channel["note_event"]] <= 0.5:
            return "0"
        ka_value = values[channel["ka_probability"]] if "ka_probability" in channel else 0.0
        return "2" if ka_value > 0.5 else "1"
    if "roll_end" not in channel and "balloon_end" not in channel:
        if values[channel["ka"]] > 0.5 and values[channel["ka"]] >= values[channel["don"]]:
            return "2"
        return "1" if values[channel["don"]] > 0.5 else "0"
    if values[channel["roll_end"]] > 0.5 or values[channel["balloon_end"]] > 0.5:
        return "8"
    if values[channel["balloon_start"]] > 0.5:
        return "7"
    if values[channel["roll_start"]] > 0.5:
        return "5"
    if "big_ka" in channel and values[channel["big_ka"]] > 0.5:
        return "4"
    if "big_don" in channel and values[channel["big_don"]] > 0.5:
        return "3"
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
    if "avg_density" not in condition and "decode_avg_density" in data.files:
        condition["avg_density"] = float(data["decode_avg_density"][0])
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
    legal_mask = data["legal_mask"].astype(np.float32) if "legal_mask" in data.files and data["legal_mask"].size else None
    if legal_mask is not None:
        note_score = np.where(legal_mask > 0.5, note_score, -np.inf)
        note_count = min(note_count, int((legal_mask > 0.5).sum()))
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
        if "big_don" in channel and "big_ka" in channel:
            selected_indices = np.flatnonzero(selected)
            big_count = int(round(len(selected_indices) * min(max(condition.get("big_note_ratio", 0.0), 0.0), 1.0)))
            if big_count > 0 and "measure_indices" in data.files and "slot_indices" in data.files:
                measure_indices = data["measure_indices"].astype(np.int32)
                slot_indices = data["slot_indices"].astype(np.int32)
                positioned = selected_indices[
                    (measure_indices[selected_indices] >= 0) & (slot_indices[selected_indices] >= 0)
                ]
                positioned = sorted(
                    positioned,
                    key=lambda frame: int(measure_indices[frame]) * 96 + int(slot_indices[frame]),
                )
                absolute_slots = np.asarray(
                    [int(measure_indices[frame]) * 96 + int(slot_indices[frame]) for frame in positioned],
                    dtype=np.int32,
                )
                sparse_indices = []
                for index, frame in enumerate(positioned):
                    gap_before = absolute_slots[index] - absolute_slots[index - 1] if index > 0 else 12
                    gap_after = absolute_slots[index + 1] - absolute_slots[index] if index + 1 < len(positioned) else 12
                    if gap_before >= 12 and gap_after >= 12:
                        sparse_indices.append(int(frame))
                big_score = np.maximum(
                    probability[:, channel["big_don"]],
                    probability[:, channel["big_ka"]],
                )
                groups: list[list[int]] = []
                for frame in sparse_indices:
                    absolute_slot = int(measure_indices[frame]) * 96 + int(slot_indices[frame])
                    if groups:
                        previous = groups[-1][-1]
                        previous_slot = int(measure_indices[previous]) * 96 + int(slot_indices[previous])
                    else:
                        previous_slot = -100
                    if absolute_slot == previous_slot + 12:
                        groups[-1].append(int(frame))
                    else:
                        groups.append([int(frame)])
                groups.sort(key=lambda group: float(np.mean(big_score[group])), reverse=True)
                big_indices: list[int] = []
                for group in groups:
                    if big_indices and abs(big_count - len(big_indices)) < abs(big_count - len(big_indices) - len(group)):
                        continue
                    big_indices.extend(group)
                    if len(big_indices) >= big_count:
                        break
                for frame in big_indices:
                    if binary[frame, channel["ka"]] > 0.5:
                        binary[frame, channel["ka"]] = 0.0
                        binary[frame, channel["big_ka"]] = 1.0
                    else:
                        binary[frame, channel["don"]] = 0.0
                        binary[frame, channel["big_don"]] = 1.0
        if all(name in channel for name in ["roll_start", "roll_body", "roll_end", "balloon_start", "balloon_body", "balloon_end"]):
            span_ratio = min(max(condition.get("balloon_roll_ratio", 0.0), 0.0), 1.0)
            span_count = int(round(span_ratio * max(len(np.flatnonzero(selected)), 1) / 8.0))
            available = legal_mask > 0.5 if legal_mask is not None else np.ones(probability.shape[0], dtype=bool)
            occupied = np.zeros(probability.shape[0], dtype=bool)
            start_score = np.maximum(probability[:, channel["roll_start"]], probability[:, channel["balloon_start"]])
            start_score = np.where(available, start_score, -np.inf)
            for _ in range(span_count):
                start = int(np.argmax(np.where(occupied, -np.inf, start_score)))
                if not np.isfinite(start_score[start]):
                    break
                candidates = np.flatnonzero(available & (np.arange(probability.shape[0]) >= start + 2) & (np.arange(probability.shape[0]) <= start + 32) & ~occupied)
                if candidates.size == 0:
                    break
                end_score = np.maximum(probability[candidates, channel["roll_end"]], probability[candidates, channel["balloon_end"]])
                end = int(candidates[np.argmax(end_score)])
                is_balloon = probability[start, channel["balloon_start"]] > probability[start, channel["roll_start"]]
                binary[start : end + 1] = 0.0
                prefix = "balloon" if is_balloon else "roll"
                binary[start, channel[f"{prefix}_start"]] = 1.0
                binary[start + 1 : end, channel[f"{prefix}_body"]] = 1.0
                binary[end, channel[f"{prefix}_end"]] = 1.0
                occupied[max(0, start - 1) : min(probability.shape[0], end + 2)] = True
        elif all(name in channel for name in ["hold_start", "hold_body", "hold_end"]):
            starts = probability[:, channel["hold_start"]] > 0.5
            holding = probability[:, channel["hold_body"]] > 0.5
            for start in np.flatnonzero(starts):
                end = int(start)
                cursor = int(start) + 1
                while cursor < probability.shape[0] and holding[cursor] and not starts[cursor]:
                    end = cursor
                    cursor += 1
                if end == start:
                    continue
                binary[start : end + 1] = 0.0
                binary[start, channel["hold_start"]] = 1.0
                binary[start + 1 : end + 1, channel["hold_body"]] = 1.0
                binary[end, channel["hold_body"]] = 0.0
                binary[end, channel["hold_end"]] = 1.0
    return binary


def hold_spans(binary: np.ndarray, names: list[str]) -> list[tuple[int, int]]:
    channel = {name: index for index, name in enumerate(names)}
    if "hold_start" not in channel or "hold_end" not in channel:
        return []
    starts = np.flatnonzero(binary[:, channel["hold_start"]] > 0.5)
    ends = np.flatnonzero(binary[:, channel["hold_end"]] > 0.5)
    spans = []
    for start in starts:
        later = ends[ends > start]
        if later.size:
            spans.append((int(start), int(later[0])))
    return spans


def balloon_span_indices(span_count: int, ratio: float) -> set[int]:
    balloon_count = min(span_count, max(0, int(round(span_count * min(max(ratio, 0.0), 1.0)))))
    if balloon_count == 0:
        return set()
    return set(np.linspace(0, span_count - 1, num=balloon_count, dtype=np.int32).tolist())


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
    strict_slots = (
        "measure_indices" in data.files
        and "slot_indices" in data.files
        and np.any(data["measure_indices"] >= 0)
        and np.any(data["slot_indices"] >= 0)
    )
    bpm = float(data["bpm_track"][0]) if "bpm_track" in data.files and data["bpm_track"].size else float(args.bpm)
    condition_names = [str(name) for name in data["condition_names"]]
    raw_condition = data["raw_condition"].astype(np.float32)
    condition = {name: float(raw_condition[index]) for index, name in enumerate(condition_names)}
    spans = hold_spans(binary, names)
    balloon_indices = balloon_span_indices(len(spans), condition.get("balloon_roll_ratio", 0.0))
    balloon_starts = {spans[index][0] for index in balloon_indices}
    balloon_hits = [
        max(1, int(np.floor((spans[index][1] - spans[index][0]) * float(args.frame_ms) / 1000.0 * 15.0)))
        for index in sorted(balloon_indices)
    ]
    if strict_slots:
        measure_indices = data["measure_indices"].astype(np.int32)
        slot_indices = data["slot_indices"].astype(np.int32)
        subdivision_bin = int(round(condition.get("subdivision_bin", condition.get("complex_bin", 2.0))))
        slots_per_measure = 16 if subdivision_bin == 0 else 96
        measures = []
        for measure_index in sorted(set(int(value) for value in measure_indices if value >= 0)):
            chars = ["0"] * slots_per_measure
            frames = np.flatnonzero((measure_indices == measure_index) & (slot_indices >= 0))
            for frame in frames:
                common_slot = int(slot_indices[frame])
                output_slot = common_slot // 6 if slots_per_measure == 16 else common_slot
                if output_slot < slots_per_measure:
                    char = frame_to_char(binary[frame], names)
                    chars[output_slot] = "7" if frame in balloon_starts and char == "5" else char
            commands = []
            channel = {name: index for index, name in enumerate(names)}
            if condition.get("bpm_rhythm_bin", 0.0) > 0.0 and "bpm_change_event" in channel and "bpm_value" in channel:
                event_frame = int(frames[np.argmax(probability[frames, channel["bpm_change_event"]])]) if frames.size else -1
                if event_frame >= 0 and probability[event_frame, channel["bpm_change_event"]] > 0.5:
                    next_bpm = float(probability[event_frame, channel["bpm_value"]] * 300.0)
                    if next_bpm > 30.0:
                        commands.append(f"#BPMCHANGE {next_bpm:.6g}")
            if condition.get("hs_change_bin", 0.0) > 0.0 and "scroll_change_event" in channel and "scroll_value" in channel:
                event_frame = int(frames[np.argmax(probability[frames, channel["scroll_change_event"]])]) if frames.size else -1
                if event_frame >= 0 and probability[event_frame, channel["scroll_change_event"]] > 0.5:
                    next_scroll = max(0.05, float(probability[event_frame, channel["scroll_value"]] * 4.0))
                    commands.append(f"#SCROLL {next_scroll:.6g}")
            measures.extend([*commands, "".join(chars) + ","])
        frames_per_measure = slots_per_measure
    else:
        frames_per_measure = max(1, int(round((4.0 * 60000.0 / bpm) / float(args.frame_ms))))
        chars = []
        for index in range(binary.shape[0]):
            char = frame_to_char(binary[index], names)
            chars.append("7" if index in balloon_starts and char == "5" else char)
        measures = [
            "".join(chars[start : start + frames_per_measure]) + ","
            for start in range(0, len(chars), frames_per_measure)
        ]
    title = str(data["source_title"][0]) if "source_title" in data.files else "diffusion_v0_sample"
    text = "\n".join(
        [
            f"TITLE:{title} diffusion_sample",
            f"BPM:{bpm:.6g}",
            *(["BALLOON:" + ",".join(str(value) for value in balloon_hits)] if balloon_hits else []),
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
