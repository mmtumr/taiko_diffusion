from __future__ import annotations

import numpy as np

from taiko_diffusion.data.ability_proxies import (
    accent_color_irregularity,
    burst_proxy,
    hs_motion,
    optimal_technique_path,
    rhythm_proxy,
    single_hand_demand,
    technique_complexity_proxy,
)
from taiko_diffusion.data.tja import ParsedCourse, TimedNote, TimedValue


def test_short_peak_scores_more_burst_than_long_stream():
    short = np.arange(5_000.0, 6_000.0, 80.0)
    sustained = np.arange(1_000.0, 11_000.0, 80.0)
    short_score, _, _ = burst_proxy(short, 12_000.0)
    sustained_score, _, sustained_seconds = burst_proxy(sustained, 12_000.0)
    assert sustained_seconds > 2.0
    assert short_score > sustained_score


def test_path_reports_switching_and_hard_force_dependency():
    stable = np.full((4, 6), 1.0)
    stable[:, 0] = 0.1
    switched = np.full((4, 6), 1.0)
    switched[0, 0] = switched[2, 0] = 0.0
    switched[1, 4] = switched[3, 4] = 0.0
    stable_score, stable_switches, stable_hard, _ = optimal_technique_path(stable, np.ones(4), switch_penalty=0.01)
    switched_score, switch_rate, hard_share, _ = optimal_technique_path(switched, np.ones(4), switch_penalty=0.01)
    assert stable_switches == stable_hard == 0.0
    assert switch_rate > 0.0 and hard_share > 0.0
    assert switched_score > stable_score


def test_single_hand_limit_suppresses_low_density_complexity():
    low_times = np.arange(12, dtype=np.float64) * 200.0   # 5 NPS
    high_times = np.arange(12, dtype=np.float64) * 100.0  # 10 NPS, BPM150 16ths
    colors = np.asarray([0, 0, 1, 0, 1, 1] * 2, dtype=np.int8)
    single_color = np.zeros(12, dtype=np.int8)
    bpm = np.full(12, 150.0)
    low, *_ = technique_complexity_proxy(low_times, colors, bpm)
    high, *_ = technique_complexity_proxy(high_times, colors, bpm)
    assert single_hand_demand(low_times, single_color) < 0.1
    assert single_hand_demand(low_times, colors) > single_hand_demand(low_times, single_color) + 0.25
    assert single_hand_demand(high_times, single_color) > 0.95
    assert single_hand_demand(high_times, colors) > 0.99
    assert high > low * 2.0


def test_inconsistent_red_blue_accents_are_harder():
    chart_consistent = np.zeros((102, 8), dtype=np.float32)
    chart_irregular = np.zeros((102, 8), dtype=np.float32)
    audio = np.zeros((132, 8), dtype=np.float32)
    audio[64] = np.asarray([0.9, 0.1] * 4)
    chart_consistent[0, ::2] = 1.0
    chart_consistent[1, 1::2] = 1.0
    chart_irregular[0, [0, 1, 4, 5]] = 1.0
    chart_irregular[1, [2, 3, 6, 7]] = 1.0
    assert accent_color_irregularity(chart_consistent, audio) < 0.05
    assert accent_color_irregularity(chart_irregular, audio) > 0.25


def _chart(intervals: list[float], bpm_changes: list[TimedValue] | None = None) -> ParsedCourse:
    times = np.cumsum([0.0, *intervals]) * 500.0  # 120 BPM: values are beats.
    duration = float(times[-1] + 2_000.0)
    return ParsedCourse(
        path="synthetic.tja",
        title="synthetic",
        course="Oni",
        level="1",
        initial_bpm=120.0,
        notes=[TimedNote(float(time), "don") for time in times],
        barlines=list(np.arange(0.0, duration + 1.0, 2_000.0)),
        bpm_changes=bpm_changes or [TimedValue(0.0, 120.0)],
        measure_changes=[TimedValue(0.0, 1.0)],
        duration_ms=duration,
    )


def test_regular_triplets_are_not_treated_as_binary_offbeats():
    triplet = _chart([1 / 3, 1 / 3, 1 / 3] * 20)
    mixed = _chart(
        [1 / 3, 1 / 4, 2 / 3, 1 / 2] * 15,
        [TimedValue(0.0, 120.0), TimedValue(5_000.0, 150.0)],
    )
    triplet_score, _, triplet_mix = rhythm_proxy(triplet, 0.5)
    mixed_score, _, mixed_mix = rhythm_proxy(mixed, 0.5)
    assert triplet_mix < mixed_mix
    assert triplet_score < mixed_score


def test_large_hs_changes_add_rhythm_processing_load():
    stable = _chart([0.5] * 40)
    changing = _chart([0.5] * 40)
    stable.scroll_changes = [TimedValue(0.0, 1.0)]
    changing.scroll_changes = [TimedValue(0.0, 1.0), TimedValue(4_000.0, 2.0), TimedValue(8_000.0, 0.5)]
    assert hs_motion(stable) == 0.0
    assert hs_motion(changing) > 0.5
    assert rhythm_proxy(changing, 0.2)[0] > rhythm_proxy(stable, 0.2)[0]
