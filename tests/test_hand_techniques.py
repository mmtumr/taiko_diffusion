import numpy as np

from taiko_diffusion.data.hand_techniques import phrase_ids, simulate_techniques


def test_full_alternate_never_repeats_hand():
    frames = np.asarray([0, 2, 4, 30, 32], dtype=np.int32)
    result = simulate_techniques(frames, np.zeros(5), np.full(5, 120), 40)["full_alternate"]
    assert result.hands[frames].tolist() == [0, 1, 0, 1, 0]


def test_balanced_returns_to_dominant_hand_after_scatter_gap():
    frames = np.asarray([0, 2, 30, 32], dtype=np.int32)
    result = simulate_techniques(frames, np.zeros(4), np.full(4, 120), 40)["balanced"]
    assert result.hands[frames].tolist() == [0, 1, 0, 1]


def test_half_specialized_prefers_don_and_ka_separation():
    frames = np.arange(0, 24, 2, dtype=np.int32)
    colors = np.asarray([0, 1] * 6, dtype=np.int8)
    result = simulate_techniques(frames, colors, np.full(len(frames), 120), 30)["half_specialized"]
    assert np.mean(result.hands[frames] == colors) >= 0.75


def test_phrase_split_scales_with_bpm():
    frames = np.asarray([0, 8, 20], dtype=np.int32)
    slow = phrase_ids(frames, np.full(3, 60), 46.4399)
    fast = phrase_ids(frames, np.full(3, 240), 46.4399)
    assert slow[-1] < fast[-1]
