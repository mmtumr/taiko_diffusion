from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TECHNIQUES = ("balanced", "reverse_balanced", "half_switch", "full_alternate", "brute_force", "half_specialized")


@dataclass(frozen=True)
class TechniqueResult:
    name: str
    hands: np.ndarray
    repeat_stress: np.ndarray
    switch_events: np.ndarray
    total_cost: float

    def tracks(self, frames: int) -> np.ndarray:
        output = np.zeros((frames, 4), dtype=np.float32)
        note_frames = np.flatnonzero(self.hands >= 0)
        output[note_frames, self.hands[note_frames]] = 1.0
        output[:, 2] = self.repeat_stress
        output[:, 3] = self.switch_events
        return output


def phrase_ids(note_frames: np.ndarray, bpm_at_notes: np.ndarray, frame_ms: float) -> np.ndarray:
    """Split 连音/散音 using a BPM-relative gap instead of a fixed second value."""
    result = np.zeros(len(note_frames), dtype=np.int32)
    phrase = 0
    for index in range(1, len(note_frames)):
        beat_frames = 60_000.0 / max(float(bpm_at_notes[index]), 1.0) / frame_ms
        # A gap longer than roughly one beat starts a new 散音/连音 group.
        if note_frames[index] - note_frames[index - 1] > max(beat_frames * 0.9, 8.0):
            phrase += 1
        result[index] = phrase
    return result


def _stress(gap: int, beat_frames: float) -> float:
    # Reusing one hand inside an eighth-note interval rapidly increases load.
    target = max(beat_frames / 2.0, 2.0)
    return max(target / max(float(gap), 1.0) - 1.0, 0.0)


def _strict_alternate(note_frames: np.ndarray, starts: np.ndarray, phrases: np.ndarray, frames: int, name: str) -> TechniqueResult:
    hands = np.full(frames, -1, dtype=np.int8)
    local = 0
    previous_phrase = -1
    for index, frame in enumerate(note_frames):
        if phrases[index] != previous_phrase:
            local = 0
            previous_phrase = int(phrases[index])
        hands[frame] = int(starts[index] ^ (local & 1))
        local += 1
    repeat = np.zeros(frames, dtype=np.float32)
    switch = np.zeros(frames, dtype=np.float32)
    previous_hand = -1
    for frame in note_frames:
        if previous_hand >= 0 and hands[frame] != previous_hand:
            switch[frame] = 1
        previous_hand = int(hands[frame])
    return TechniqueResult(name, hands, repeat, switch, float(switch.sum()) * 0.01)


def _beam_assign(
    note_frames: np.ndarray,
    colors: np.ndarray,
    bpm: np.ndarray,
    phrases: np.ndarray,
    frames: int,
    *,
    name: str,
    repeat_weight: float,
    alternate_weight: float,
    color_weight: float,
    dominant_phrase_weight: float,
    balance_weight: float,
    beam_size: int = 16,
) -> TechniqueResult:
    # state: cost, assignments, last hand, last frames per hand, counts
    beam = [(0.0, (), -1, (-100_000, -100_000), (0, 0))]
    for index, frame in enumerate(note_frames):
        candidates = []
        beat_frames = 60_000.0 / max(float(bpm[index]), 1.0) / 46.4399
        phrase_start = index == 0 or phrases[index] != phrases[index - 1]
        for cost, assignment, last, last_frames, counts in beam:
            for hand in (0, 1):
                repeat = _stress(int(frame - last_frames[hand]), beat_frames)
                extra = repeat_weight * repeat
                if hand == last:
                    extra += alternate_weight
                if color_weight and hand != int(colors[index]):
                    extra += color_weight
                if phrase_start and hand != 0:
                    extra += dominant_phrase_weight
                new_counts = (counts[0] + int(hand == 0), counts[1] + int(hand == 1))
                extra += balance_weight * abs(new_counts[0] - new_counts[1]) / max(index + 1, 1)
                new_last = list(last_frames); new_last[hand] = int(frame)
                candidates.append((cost + extra, assignment + (hand,), hand, tuple(new_last), new_counts))
        candidates.sort(key=lambda item: item[0])
        beam = candidates[:beam_size]
    cost, assignment, _, _, _ = beam[0]
    hands = np.full(frames, -1, dtype=np.int8)
    repeat_track = np.zeros(frames, dtype=np.float32)
    switch_track = np.zeros(frames, dtype=np.float32)
    last_frames = [-100_000, -100_000]
    last_hand = -1
    for index, (frame, hand) in enumerate(zip(note_frames, assignment)):
        beat_frames = 60_000.0 / max(float(bpm[index]), 1.0) / 46.4399
        hands[frame] = hand
        repeat_track[frame] = _stress(int(frame - last_frames[hand]), beat_frames)
        switch_track[frame] = float(last_hand >= 0 and last_hand != hand)
        last_frames[hand] = int(frame); last_hand = hand
    return TechniqueResult(name, hands, repeat_track, switch_track, float(cost))


def simulate_techniques(note_frames: np.ndarray, colors: np.ndarray, bpm_at_notes: np.ndarray, frames: int, frame_ms: float = 46.4399) -> dict[str, TechniqueResult]:
    note_frames = np.asarray(note_frames, dtype=np.int32)
    colors = np.asarray(colors, dtype=np.int8)
    bpm_at_notes = np.asarray(bpm_at_notes, dtype=np.float32)
    if not len(note_frames):
        empty = np.full(frames, -1, dtype=np.int8); zero = np.zeros(frames, np.float32)
        return {name: TechniqueResult(name, empty.copy(), zero.copy(), zero.copy(), 0.0) for name in TECHNIQUES}
    phrases = phrase_ids(note_frames, bpm_at_notes, frame_ms)
    balanced_starts = np.zeros(len(note_frames), dtype=np.int8)
    reverse_starts = np.ones(len(note_frames), dtype=np.int8)
    # 半换 chooses the phrase start that keeps cumulative hand use closest.
    half_starts = np.zeros(len(note_frames), dtype=np.int8)
    phrase_values = np.unique(phrases); left = right = 0
    for phrase in phrase_values:
        positions = np.flatnonzero(phrases == phrase); length = len(positions)
        start = int(left > right); half_starts[positions] = start
        left += (length + int(start == 0)) // 2; right += (length + int(start == 1)) // 2
    global_starts = np.zeros(len(note_frames), dtype=np.int8)
    # For 全换, each note is treated as one uninterrupted phrase.
    global_phrases = np.zeros(len(note_frames), dtype=np.int32)
    results = {
        "balanced": _strict_alternate(note_frames, balanced_starts, phrases, frames, "balanced"),
        "reverse_balanced": _strict_alternate(note_frames, reverse_starts, phrases, frames, "reverse_balanced"),
        "half_switch": _strict_alternate(note_frames, half_starts, phrases, frames, "half_switch"),
        "full_alternate": _strict_alternate(note_frames, global_starts, global_phrases, frames, "full_alternate"),
        "brute_force": _beam_assign(note_frames, colors, bpm_at_notes, phrases, frames, name="brute_force", repeat_weight=0.45, alternate_weight=0.02, color_weight=0.0, dominant_phrase_weight=0.10, balance_weight=0.02),
        "half_specialized": _beam_assign(note_frames, colors, bpm_at_notes, phrases, frames, name="half_specialized", repeat_weight=0.70, alternate_weight=0.03, color_weight=0.35, dominant_phrase_weight=0.05, balance_weight=0.04),
    }
    return results


def technique_tracks_from_grid(x: np.ndarray, channels: list[str], frame_ms: float = 46.4399) -> tuple[np.ndarray, dict[str, float]]:
    don = x[:, channels.index("don")] > 0.5
    ka = x[:, channels.index("ka")] > 0.5
    note_frames = np.flatnonzero(don | ka)
    colors = ka[note_frames].astype(np.int8)  # 正手=0 for 咚, 反手=1 for 咔 in 半分工.
    bpm_track = x[:, channels.index("bpm")]
    # The chart grid stores BPM divided by 300.
    bpm = np.maximum(bpm_track[note_frames] * 300.0, 1.0)
    results = simulate_techniques(note_frames, colors, bpm, x.shape[0], frame_ms)
    tracks = np.concatenate([results[name].tracks(x.shape[0]) for name in TECHNIQUES], axis=1)
    return tracks, {name: results[name].total_cost for name in TECHNIQUES}
