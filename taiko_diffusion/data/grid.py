from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from taiko_diffusion.data.tja import ParsedCourse, TimedValue


CHANNELS = [
    "don",
    "ka",
    "big_don",
    "big_ka",
    "roll_start",
    "roll_body",
    "roll_end",
    "balloon_start",
    "balloon_body",
    "balloon_end",
    "gogo",
    "barline",
    "bpm",
    "bpm_change_event",
    "bpm_delta",
    "scroll",
    "scroll_change_event",
    "scroll_delta",
    "measure",
    "active",
]


@dataclass
class GridResult:
    x: np.ndarray
    channels: list[str]
    duration_frames: int
    clipped: bool


def frame_index(time_ms: float, frame_ms: float, max_frames: int) -> int:
    return max(0, min(max_frames - 1, int(round(time_ms / frame_ms))))


def fill_timed_values(
    x: np.ndarray,
    channel: int,
    changes: list[TimedValue],
    duration_frames: int,
    frame_ms: float,
    max_frames: int,
    scale: float = 1.0,
) -> None:
    if not changes or duration_frames <= 0:
        return
    ordered = sorted(changes, key=lambda item: item.time_ms)
    for index, change in enumerate(ordered):
        start = frame_index(change.time_ms, frame_ms, max_frames)
        if start >= duration_frames:
            continue
        if index + 1 < len(ordered):
            end = frame_index(ordered[index + 1].time_ms, frame_ms, max_frames)
        else:
            end = duration_frames
        end = max(start + 1, min(end, duration_frames))
        x[start:end, channel] = change.value / scale


def mark_change_events(
    x: np.ndarray,
    event_channel: int | None,
    delta_channel: int | None,
    changes: list[TimedValue],
    frame_ms: float,
    max_frames: int,
    scale: float = 1.0,
) -> None:
    if len(changes) <= 1:
        return
    previous = changes[0].value
    for change in sorted(changes, key=lambda item: item.time_ms)[1:]:
        idx = frame_index(change.time_ms, frame_ms, max_frames)
        if event_channel is not None:
            x[idx, event_channel] = 1.0
        if delta_channel is not None:
            x[idx, delta_channel] = (change.value - previous) / scale
        previous = change.value


def chart_to_grid(
    chart: ParsedCourse,
    frame_ms: float,
    max_frames: int,
    channels: list[str] | None = None,
) -> GridResult:
    channels = channels or CHANNELS
    channel_index = {name: index for index, name in enumerate(channels)}
    x = np.zeros((max_frames, len(channels)), dtype=np.float32)
    duration_frames = min(max_frames, int(np.ceil(chart.duration_ms / frame_ms)) + 1)
    clipped = chart.duration_ms / frame_ms >= max_frames

    if duration_frames > 0 and "active" in channel_index:
        x[:duration_frames, channel_index["active"]] = 1.0

    for note in chart.notes:
        if note.kind not in channel_index:
            continue
        idx = frame_index(note.time_ms, frame_ms, max_frames)
        x[idx, channel_index[note.kind]] = 1.0

    for span in chart.rolls:
        start = frame_index(span.start_ms, frame_ms, max_frames)
        end = frame_index(span.end_ms, frame_ms, max_frames)
        if span.kind == "balloon":
            start_name, body_name, end_name = "balloon_start", "balloon_body", "balloon_end"
        else:
            start_name, body_name, end_name = "roll_start", "roll_body", "roll_end"
        if start_name in channel_index:
            x[start, channel_index[start_name]] = 1.0
        if end_name in channel_index:
            x[end, channel_index[end_name]] = 1.0
        if body_name in channel_index:
            body_start = min(start + 1, max_frames)
            body_end = max(body_start, min(end, max_frames))
            x[body_start:body_end, channel_index[body_name]] = 1.0

    if "gogo" in channel_index:
        for span in chart.gogo:
            start = frame_index(span.start_ms, frame_ms, max_frames)
            end = frame_index(span.end_ms, frame_ms, max_frames)
            x[start : max(start + 1, min(end, max_frames)), channel_index["gogo"]] = 1.0

    if "barline" in channel_index:
        for time_ms in chart.barlines:
            idx = frame_index(time_ms, frame_ms, max_frames)
            x[idx, channel_index["barline"]] = 1.0

    if "bpm" in channel_index:
        fill_timed_values(
            x, channel_index["bpm"], chart.bpm_changes, duration_frames, frame_ms, max_frames, scale=300.0
        )
    mark_change_events(
        x,
        channel_index.get("bpm_change_event"),
        channel_index.get("bpm_delta"),
        chart.bpm_changes,
        frame_ms,
        max_frames,
        scale=300.0,
    )
    if "scroll" in channel_index:
        fill_timed_values(x, channel_index["scroll"], chart.scroll_changes, duration_frames, frame_ms, max_frames)
    mark_change_events(
        x,
        channel_index.get("scroll_change_event"),
        channel_index.get("scroll_delta"),
        chart.scroll_changes,
        frame_ms,
        max_frames,
        scale=4.0,
    )
    if "measure" in channel_index:
        fill_timed_values(x, channel_index["measure"], chart.measure_changes, duration_frames, frame_ms, max_frames)

    return GridResult(x=x, channels=channels, duration_frames=duration_frames, clipped=clipped)
