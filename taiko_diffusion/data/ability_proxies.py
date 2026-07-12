from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from taiko_diffusion.data.hand_techniques import TECHNIQUES, phrase_ids, simulate_techniques
from taiko_diffusion.data.tja import ParsedCourse, TimedValue


PLAYABLE_KINDS = {"don", "ka", "big_don", "big_ka"}
KA_KINDS = {"ka", "big_ka"}


@dataclass(frozen=True)
class ProxyDiagnostics:
    burst_peak_nps: float
    burst_sustained_seconds: float
    technique_switch_rate: float
    hard_force_share: float
    single_hand_resolvable_share: float
    rhythm_syncopation: float
    rhythm_family_mixing: float
    rhythm_accent_color_irregularity: float
    rhythm_hs_motion: float


def _value_at(changes: list[TimedValue], times_ms: np.ndarray) -> np.ndarray:
    if not len(times_ms):
        return np.zeros(0, dtype=np.float64)
    change_times = np.asarray([value.time_ms for value in changes], dtype=np.float64)
    change_values = np.asarray([value.value for value in changes], dtype=np.float64)
    indices = np.searchsorted(change_times, times_ms, side="right") - 1
    return change_values[np.maximum(indices, 0)]


def playable_notes(chart: ParsedCourse) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    notes = [note for note in chart.notes if note.kind in PLAYABLE_KINDS]
    times = np.asarray([note.time_ms for note in notes], dtype=np.float64)
    colors = np.asarray([note.kind in KA_KINDS for note in notes], dtype=np.int8)
    bpm = _value_at(chart.bpm_changes, times)
    measure = _value_at(chart.measure_changes, times)
    return times, colors, bpm, measure


def density_track(times_ms: np.ndarray, duration_ms: float, window_seconds: float, step_seconds: float = 0.1) -> np.ndarray:
    """Centred sliding note density in notes/second."""
    if duration_ms <= 0 or not len(times_ms):
        return np.zeros(1, dtype=np.float64)
    centres = np.arange(0.0, duration_ms + step_seconds * 1000.0, step_seconds * 1000.0)
    half = window_seconds * 500.0
    left = np.searchsorted(times_ms, centres - half, side="left")
    right = np.searchsorted(times_ms, centres + half, side="right")
    return (right - left).astype(np.float64) / window_seconds


def _longest_run(mask: np.ndarray, step_seconds: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * step_seconds


def _run_around(mask: np.ndarray, index: int, step_seconds: float) -> float:
    left = right = index
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return (right - left + 1) * step_seconds


def burst_proxy(times_ms: np.ndarray, duration_ms: float) -> tuple[float, float, float]:
    """Reward a 0.25--2 s density peak and discount a peak that stays high."""
    if len(times_ms) < 2:
        return 0.0, 0.0, 0.0
    step = 0.1
    short = density_track(times_ms, duration_ms, 0.75, step)
    medium = density_track(times_ms, duration_ms, 2.0, step)
    long = density_track(times_ms, duration_ms, 6.0, step)
    peak_index = int(np.argmax(short))
    peak = float(short[peak_index])
    prominence_curve = np.maximum(short - long, 0.0)
    prominence = float(np.quantile(prominence_curve, 0.98))
    contrast = prominence / max(float(np.quantile(short, 0.98)), 1e-6)
    peak_width = _run_around(short >= max(peak * 0.75, 1e-6), peak_index, step)
    plateau_width = _longest_run(medium >= max(float(np.max(medium)) * 0.82, 1e-6), step)
    sustained = max(peak_width, plateau_width)
    rise = np.maximum(short[5:] - short[:-5], 0.0) if len(short) > 5 else np.zeros(1)
    attack = float(np.quantile(rise, 0.98) / max(peak, 1e-6))
    compactness = float(np.exp(-max(sustained - 2.0, 0.0) / 2.5))
    # The curve shape, not the peak alone, decides burst: a prominent, sharp,
    # narrow hill scores high; a long plateau migrates toward stamina.
    shape = 0.12 + 0.58 * np.clip(contrast, 0.0, 1.0) + 0.30 * np.clip(attack * 2.0, 0.0, 1.0)
    score = peak * shape * compactness
    return float(score), peak, sustained


def single_hand_demand(times_ms: np.ndarray, colors: np.ndarray | None = None) -> float:
    """How strongly a passage exceeds a duration-aware single-hand limit."""
    times_ms = np.asarray(times_ms, dtype=np.float64)
    if len(times_ms) < 2:
        return 0.0
    duration = max(float(times_ms[-1] - times_ms[0]) / 1000.0, 0.05)
    nps = (len(times_ms) - 1) / duration
    # This is a practical chart-handling limit, not a laboratory maximum. A
    # short single-colour push can approach ~7 NPS, but sustained handling falls
    # toward ~5.2 NPS. BPM150 16ths (10 NPS) are therefore severely over limit.
    capacity = 5.2 + 1.8 * np.exp(-duration / 0.85)
    if colors is not None and len(colors) == len(times_ms) and len(colors) > 1:
        color_change_rate = float(np.mean(np.asarray(colors)[1:] != np.asarray(colors)[:-1]))
        # Moving one hand between the drumhead and rim makes 硬抗 stop being a
        # credible simplification much earlier than a single-colour stream.
        capacity -= 1.25 * np.sqrt(color_change_rate)
    x = float(np.clip((nps - (capacity - 0.35)) / 2.0, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def optimal_technique_path(
    local_costs: np.ndarray,
    note_counts: np.ndarray,
    *,
    switch_penalty: float = 0.035,
    hard_force_index: int = 4,
) -> tuple[float, float, float, np.ndarray]:
    """Choose a technique per phrase and retain switching/hard-force costs."""
    local_costs = np.asarray(local_costs, dtype=np.float64)
    note_counts = np.asarray(note_counts, dtype=np.float64)
    if local_costs.ndim != 2 or local_costs.shape[0] == 0:
        return 0.0, 0.0, 0.0, np.zeros(0, dtype=np.int64)
    phrases, methods = local_costs.shape
    cumulative = local_costs[0] * max(note_counts[0], 1.0)
    back = np.zeros((phrases, methods), dtype=np.int64)
    for phrase in range(1, phrases):
        transition = cumulative[:, None] + switch_penalty * max(note_counts[phrase], 1.0) * (
            np.arange(methods)[:, None] != np.arange(methods)[None, :]
        )
        previous = np.argmin(transition, axis=0)
        cumulative = transition[previous, np.arange(methods)] + local_costs[phrase] * max(note_counts[phrase], 1.0)
        back[phrase] = previous
    path = np.zeros(phrases, dtype=np.int64)
    path[-1] = int(np.argmin(cumulative))
    for phrase in range(phrases - 1, 0, -1):
        path[phrase - 1] = back[phrase, path[phrase]]
    weights = np.maximum(note_counts, 1.0)
    base = float(np.average(local_costs[np.arange(phrases), path], weights=weights))
    switch_rate = float(np.mean(path[1:] != path[:-1])) if phrases > 1 else 0.0
    hard_share = float(np.sum(weights[path == hard_force_index]) / np.sum(weights))
    score = base + 0.55 * switch_rate + 0.70 * hard_share
    return float(score), switch_rate, hard_share, path


def technique_complexity_proxy(times_ms: np.ndarray, colors: np.ndarray, bpm: np.ndarray) -> tuple[float, float, float, float]:
    if len(times_ms) < 2:
        return 0.0, 0.0, 0.0, 1.0
    # Ten milliseconds preserves fast doubles without a full millisecond track.
    frame_ms = 10.0
    note_frames = np.rint(times_ms / frame_ms).astype(np.int32)
    for index in range(1, len(note_frames)):
        note_frames[index] = max(note_frames[index], note_frames[index - 1] + 1)
    frames = int(note_frames[-1] + 2)
    phrases = phrase_ids(note_frames, bpm, frame_ms)
    results = simulate_techniques(note_frames, colors, bpm, frames, frame_ms)
    # Long connected phrases may change their most natural handling midway.
    # Split them into compact chunks so that this demand is visible instead of
    # assigning one technique to the entire song or entire long stream.
    segments: list[np.ndarray] = []
    segment_groups: list[int] = []
    for phrase in np.unique(phrases):
        positions = np.flatnonzero(phrases == phrase)
        chunks = [np.asarray(chunk) for chunk in np.array_split(positions, max(1, int(np.ceil(len(positions) / 6.0))))]
        segments.extend(chunks)
        segment_groups.extend([int(phrase)] * len(chunks))
    local_costs = np.zeros((len(segments), len(TECHNIQUES)), dtype=np.float64)
    note_counts = np.zeros(len(segments), dtype=np.float64)
    hand_demands = np.zeros(len(segments), dtype=np.float64)
    for phrase_row, positions in enumerate(segments):
        frames_here = note_frames[positions]
        colors_here = colors[positions]
        note_counts[phrase_row] = len(positions)
        hand_demands[phrase_row] = single_hand_demand(times_ms[positions], colors_here)
        for method_index, method in enumerate(TECHNIQUES):
            result = results[method]
            hands = result.hands[frames_here].astype(np.int8)
            mapping = hands ^ colors_here
            mapping_changes = float(np.mean(mapping[1:] != mapping[:-1])) if len(mapping) > 1 else 0.0
            hand_switch = (hands[1:] != hands[:-1]).astype(np.float64)
            hand_irregularity = float(np.mean(hand_switch[1:] != hand_switch[:-1])) if len(hand_switch) > 1 else 0.0
            repeat = result.repeat_stress[frames_here]
            repeat_load = float(np.mean(repeat) + 0.65 * np.quantile(repeat, 0.90))
            # Balance cognitive mapping simplicity against the physical price
            # of 硬抗/半分工.
            raw_cost = 0.70 * mapping_changes + 0.28 * hand_irregularity + 1.10 * repeat_load
            local_costs[phrase_row, method_index] = raw_cost * (0.05 + 0.95 * hand_demands[phrase_row])
    # Rest gaps reset the choice. Switching after a scatter-note gap is free;
    # only changes within one connected phrase count as technique switching.
    score_sum = hard_sum = 0.0
    path_switches = path_transitions = 0
    for group in np.unique(segment_groups):
        selected = np.flatnonzero(np.asarray(segment_groups) == group)
        _, group_switch_rate, group_hard_share, group_path = optimal_technique_path(local_costs[selected], note_counts[selected])
        group_weight = float(np.sum(note_counts[selected]))
        group_base = float(np.average(local_costs[selected, group_path], weights=note_counts[selected]))
        transition_demand = 0.0
        if len(selected) > 1:
            changed = group_path[1:] != group_path[:-1]
            transition_demand = float(np.mean(changed * np.minimum(hand_demands[selected][1:], hand_demands[selected][:-1])))
        hard_demand = float(
            np.sum(note_counts[selected] * hand_demands[selected] * (group_path == 4)) / max(group_weight, 1.0)
        )
        group_score = group_base + 0.55 * transition_demand + 0.70 * hard_demand
        score_sum += group_score * group_weight
        hard_sum += group_hard_share * group_weight
        transitions = max(len(selected) - 1, 0)
        path_switches += group_switch_rate * transitions
        path_transitions += transitions
    score = score_sum / max(float(np.sum(note_counts)), 1.0)
    path_switch_rate = float(path_switches / max(path_transitions, 1))
    path_hard_share = float(hard_sum / max(float(np.sum(note_counts)), 1.0))
    raw_path = np.argmin(local_costs, axis=1)
    same_phrase = np.asarray(segment_groups[1:]) == np.asarray(segment_groups[:-1])
    raw_changes = raw_path[1:] != raw_path[:-1]
    raw_switch_rate = float(np.mean(raw_changes[same_phrase])) if same_phrase.any() else 0.0
    raw_switch_demand = float(
        np.mean(raw_changes[same_phrase] * np.minimum(hand_demands[1:][same_phrase], hand_demands[:-1][same_phrase]))
    ) if same_phrase.any() else 0.0
    raw_hard_share = float(np.sum(note_counts[raw_path == 4]) / max(np.sum(note_counts), 1.0))
    raw_hard_demand = float(
        np.sum(note_counts * hand_demands * (raw_path == 4)) / max(np.sum(note_counts), 1.0)
    )
    # Even when a player elects to stay on one robust technique, frequent
    # changes in the locally easiest option and sections favouring 硬抗 remain
    # real complexity demands.
    switch_rate = max(path_switch_rate, raw_switch_rate)
    hard_share = max(path_hard_share, raw_hard_share)
    score += 0.38 * raw_switch_demand + 0.70 * raw_hard_demand
    weighted_demand = float(np.average(hand_demands, weights=note_counts))
    color_changes = float(np.mean(colors[1:] != colors[:-1]))
    resolvable_share = float(np.sum(note_counts[hand_demands < 0.20]) / max(np.sum(note_counts), 1.0))
    return float(score + 0.22 * color_changes * weighted_demand), switch_rate, hard_share, resolvable_share


def _normalised_entropy(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    counts = np.asarray(list(Counter(values.tolist()).values()), dtype=np.float64)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def hs_motion(chart: ParsedCourse) -> float:
    values = np.asarray([value.value for value in chart.scroll_changes], dtype=np.float64)
    if len(values) < 2:
        return 0.0
    magnitude = np.maximum(np.abs(values), 1e-3)
    relative = np.abs(np.diff(np.log(magnitude)))
    significant = np.maximum(relative - np.log(1.05), 0.0)
    sign_flips = int(np.sum(np.sign(values[1:]) != np.sign(values[:-1])))
    minutes = max(chart.duration_ms / 60_000.0, 1.0)
    event_rate = float(np.sum(significant > 0.0) / minutes)
    return float(np.clip(np.sum(significant) / 0.8 + 0.20 * sign_flips + 0.025 * event_rate, 0.0, 1.0))


def accent_color_irregularity(pooled_chart: np.ndarray, pooled_audio: np.ndarray) -> float:
    """How inconsistently red/blue notes carry the music's acoustic accents."""
    don, ka = np.maximum(pooled_chart[0], 0.0), np.maximum(pooled_chart[1], 0.0)
    note = don + ka
    selected = note > 0
    if np.sum(selected) < 4:
        return 0.0
    weights = note[selected]
    color = (ka[selected] - don[selected]) / np.maximum(weights, 1e-6)
    onset = pooled_audio[64][selected]
    weight_sum = max(float(np.sum(weights)), 1e-6)
    color_mean = float(np.sum(weights * color) / weight_sum)
    onset_mean = float(np.sum(weights * onset) / weight_sum)
    color_centered, onset_centered = color - color_mean, onset - onset_mean
    color_std = float(np.sqrt(np.sum(weights * color_centered**2) / weight_sum))
    onset_std = float(np.sqrt(np.sum(weights * onset_centered**2) / weight_sum))
    if color_std < 0.05 or onset_std < 1e-5:
        return 0.0
    correlation = float(
        np.sum(weights * color_centered * onset_centered) / (weight_sum * color_std * onset_std + 1e-6)
    )
    accent = onset >= np.median(onset)
    color_is_ka = color > 0
    mapping = accent ^ color_is_ka
    mapping_change = float(np.mean(mapping[1:] != mapping[:-1])) if len(mapping) > 1 else 0.0
    color_balance = float(np.clip(1.0 - abs(color_mean), 0.0, 1.0))
    accent_contrast = float(np.clip(onset_std / 0.10, 0.0, 1.0))
    # A stable red-accent/blue-fill mapping is easiest. Inconsistent mapping is
    # difficult; a stable inversion (blue carrying strong accents) gets only a
    # smaller penalty because it is unusual but still learnable.
    score = accent_contrast * color_balance * (
        0.45 * (1.0 - abs(correlation)) + 0.35 * mapping_change + 0.20 * max(correlation, 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def rhythm_proxy(
    chart: ParsedCourse,
    audio_onset_mismatch: float,
    onset_variability: float = 0.0,
    weak_onset_note_share: float = 0.0,
    accent_color: float = 0.0,
) -> tuple[float, float, float]:
    """Score rhythm processing while recognising a stable triplet grid as regular."""
    times, _, bpm, measure = playable_notes(chart)
    if len(times) < 3:
        return 0.0, 0.0, 0.0
    intervals = np.diff(times) * bpm[1:] / 60_000.0
    intervals = np.round(intervals * 24.0) / 24.0
    binary_error = np.abs(intervals * 4.0 - np.round(intervals * 4.0))
    triplet_error = np.abs(intervals * 3.0 - np.round(intervals * 3.0))
    binary_only = (binary_error < 0.035) & (triplet_error >= 0.035)
    triplet_only = (triplet_error < 0.035) & (binary_error >= 0.035)
    irregular = (binary_error >= 0.035) & (triplet_error >= 0.035)
    classified = int(binary_only.sum() + triplet_only.sum())
    family_mixing = float(min(binary_only.sum(), triplet_only.sum()) / max(classified, 1))
    irregular_share = float(np.mean(irregular))

    dominant_triplet = triplet_only.sum() > binary_only.sum()
    barlines = np.asarray(chart.barlines, dtype=np.float64)
    off_grid = []
    for time, current_bpm in zip(times, bpm):
        bar = np.searchsorted(barlines, time, side="right") - 1
        if bar < 0:
            continue
        beat_phase = ((time - barlines[bar]) * current_bpm / 60_000.0) % 1.0
        subdivisions = 3.0 if dominant_triplet else 2.0
        off_grid.append(abs(beat_phase * subdivisions - round(beat_phase * subdivisions)) > 0.08)
    syncopation = float(np.mean(off_grid)) if off_grid else 0.0

    common_intervals = intervals[intervals <= 2.0]
    counts = Counter(common_intervals.tolist())
    top_three_share = sum(value for _, value in counts.most_common(3)) / max(len(common_intervals), 1)
    motif_variety = float(1.0 - top_three_share)
    interval_entropy = _normalised_entropy(common_intervals)

    bpm_values = np.asarray([value.value for value in chart.bpm_changes], dtype=np.float64)
    relative_tempo_changes = np.abs(np.diff(np.log(np.maximum(bpm_values, 1.0))))
    # Sub-2% changes in classical TJAs generally encode expressive timing, not
    # a BPM change the player must consciously process.
    tempo_motion = float(np.sum(np.maximum(relative_tempo_changes - 0.02, 0.0)))
    significant_tempo_events = int(np.sum(relative_tempo_changes >= 0.02))
    tempo_event_rate = significant_tempo_events / max(chart.duration_ms / 60_000.0, 1.0)
    tempo_motion = min(tempo_motion / 0.25 + 0.025 * tempo_event_rate, 1.0)
    scroll_motion = hs_motion(chart)
    nonstandard_meter = float(np.mean(np.abs(measure - 1.0) > 1e-4))
    meter_changes = max(len(chart.measure_changes) - 1, 0) / max(chart.duration_ms / 60_000.0, 1.0)
    meter_motion = min(nonstandard_meter + 0.20 * meter_changes, 1.0)
    melodic_following = (
        audio_onset_mismatch * (0.15 + syncopation + 0.7 * family_mixing)
        + 1.5 * onset_variability
        + 8.0 * max(weak_onset_note_share - 0.03, 0.0)
    )
    score = (
        0.95 * syncopation
        + 1.35 * family_mixing
        + 0.85 * irregular_share
        + 0.55 * motif_variety
        + 0.18 * interval_entropy
        + 0.42 * tempo_motion
        + 0.42 * scroll_motion
        + 0.55 * meter_motion
        + 0.48 * melodic_following
        + 0.55 * accent_color
    )
    return float(score), syncopation, family_mixing


def pooled_rhythm_proxy(chart: ParsedCourse, pooled_chart: np.ndarray, pooled_audio: np.ndarray) -> tuple[float, float, float]:
    note_weight = np.maximum(pooled_chart[0] + pooled_chart[1], 0.0)
    onset = pooled_audio[64]
    onset_match = float(np.sum(note_weight * onset) / max(np.sum(note_weight), 1e-6))
    onset_mismatch = float(np.clip(1.0 - onset_match, 0.0, 1.0))
    note_tokens = note_weight > 0
    token_weights = note_weight[note_tokens]
    token_onsets = onset[note_tokens]
    onset_variability = float(
        np.sqrt(np.sum(token_weights * (token_onsets - onset_match) ** 2) / max(np.sum(token_weights), 1e-6))
    )
    weak_onset_note_share = float(
        np.sum(token_weights[token_onsets < 0.20]) / max(np.sum(token_weights), 1e-6)
    )
    accent_color = accent_color_irregularity(pooled_chart, pooled_audio)
    return rhythm_proxy(chart, onset_mismatch, onset_variability, weak_onset_note_share, accent_color)


def proxy_values(chart: ParsedCourse, pooled_chart: np.ndarray, pooled_audio: np.ndarray) -> tuple[dict[str, float], ProxyDiagnostics]:
    times, colors, bpm, _ = playable_notes(chart)
    duration_ms = max(chart.duration_ms, float(times[-1]) if len(times) else 0.0)
    note_rate = len(times) / max(duration_ms / 1000.0, 1e-6)
    burst, burst_peak, sustained = burst_proxy(times, duration_ms)
    two_second_peak = float(np.max(density_track(times, duration_ms, 2.0))) if len(times) else 0.0
    complex_value, technique_switch_rate, hard_force_share, resolvable_share = technique_complexity_proxy(times, colors, bpm)
    rhythm, syncopation, family_mixing = pooled_rhythm_proxy(chart, pooled_chart, pooled_audio)
    accent_color = accent_color_irregularity(pooled_chart, pooled_audio)
    scroll_motion = hs_motion(chart)
    values = {
        "stamina": float(note_rate),
        "handspeed": float(two_second_peak + 0.35 * burst_peak),
        "burst": burst,
        "complex": complex_value,
        "rhythm": rhythm,
    }
    diagnostics = ProxyDiagnostics(
        burst_peak_nps=burst_peak,
        burst_sustained_seconds=sustained,
        technique_switch_rate=technique_switch_rate,
        hard_force_share=hard_force_share,
        single_hand_resolvable_share=resolvable_share,
        rhythm_syncopation=syncopation,
        rhythm_family_mixing=family_mixing,
        rhythm_accent_color_irregularity=accent_color,
        rhythm_hs_motion=scroll_motion,
    )
    return values, diagnostics
