from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


BRANCH_RE = re.compile(
    r"^\s*#(BRANCHSTART|BRANCHEND|SECTION|LEVELHOLD|N\b|E\b|M\b)", re.IGNORECASE
)
HEADER_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$")


class TjaParseError(ValueError):
    """Raised when a TJA chart cannot be parsed for the current pipeline."""


class BranchUnsupportedError(TjaParseError):
    """Raised when a selected course contains branch commands."""


@dataclass
class TimedNote:
    time_ms: float
    kind: str


@dataclass
class TimedSpan:
    start_ms: float
    end_ms: float
    kind: str


@dataclass
class TimedValue:
    time_ms: float
    value: float


@dataclass
class ParsedCourse:
    path: str
    title: str
    course: str
    level: str
    initial_bpm: float
    notes: list[TimedNote] = field(default_factory=list)
    rolls: list[TimedSpan] = field(default_factory=list)
    gogo: list[TimedSpan] = field(default_factory=list)
    barlines: list[float] = field(default_factory=list)
    bpm_changes: list[TimedValue] = field(default_factory=list)
    scroll_changes: list[TimedValue] = field(default_factory=list)
    measure_changes: list[TimedValue] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def playable_note_count(self) -> int:
        return sum(1 for note in self.notes if note.kind in {"don", "ka", "big_don", "big_ka"})


@dataclass
class RawCourse:
    meta: dict[str, str]
    body: list[str]


def parse_float(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return default


def strip_inline_comment(line: str) -> str:
    if line.lstrip().startswith("//"):
        return ""
    return line.split("//", 1)[0].strip()


def read_raw_courses(path: Path) -> list[RawCourse]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    global_meta: dict[str, str] = {}
    courses: list[RawCourse] = []
    current_meta: dict[str, str] | None = None
    current_body: list[str] = []
    in_body = False

    def finish_course() -> None:
        nonlocal current_meta, current_body, in_body
        if current_meta is not None:
            courses.append(RawCourse(meta={**global_meta, **current_meta}, body=current_body))
        current_meta = None
        current_body = []
        in_body = False

    for raw_line in lines:
        line = strip_inline_comment(raw_line)
        if not line:
            continue
        header = HEADER_RE.match(line)
        if header and not in_body:
            key = header.group(1).upper()
            value = header.group(2).strip()
            if key == "COURSE":
                finish_course()
                current_meta = {"COURSE": value}
            elif current_meta is None:
                global_meta[key] = value
            else:
                current_meta[key] = value
            continue

        upper = line.upper()
        if upper.startswith("#START"):
            if current_meta is None:
                current_meta = {}
            in_body = True
            continue
        if upper.startswith("#END"):
            finish_course()
            continue
        if in_body and current_meta is not None:
            current_body.append(line)

    finish_course()
    return [course for course in courses if course.meta.get("COURSE")]


def select_course(courses: list[RawCourse], course_name: str) -> RawCourse:
    requested = course_name.casefold()
    matches = [course for course in courses if course.meta.get("COURSE", "").casefold() == requested]
    if not matches:
        available = ", ".join(course.meta.get("COURSE", "") for course in courses)
        raise TjaParseError(f"Course {course_name!r} not found. Available: {available}")
    if len(matches) > 1:
        raise TjaParseError(f"Course {course_name!r} is ambiguous in TJA file.")
    return matches[0]


def measure_beats(measure_ratio: float) -> float:
    return 4.0 * measure_ratio


def finalize_measure(
    chars: list[str],
    parsed: ParsedCourse,
    time_ms: float,
    bpm: float,
    measure_ratio: float,
    barline_on: bool,
    open_span: tuple[str, float] | None,
) -> tuple[float, tuple[str, float] | None]:
    duration_ms = measure_beats(measure_ratio) * 60000.0 / max(bpm, 1e-6)
    if barline_on:
        parsed.barlines.append(time_ms)
    if chars:
        step_ms = duration_ms / len(chars)
        for index, char in enumerate(chars):
            note_time = time_ms + index * step_ms
            if char == "1":
                parsed.notes.append(TimedNote(note_time, "don"))
            elif char == "2":
                parsed.notes.append(TimedNote(note_time, "ka"))
            elif char == "3":
                parsed.notes.append(TimedNote(note_time, "big_don"))
            elif char == "4":
                parsed.notes.append(TimedNote(note_time, "big_ka"))
            elif char in {"5", "6"}:
                if open_span is not None:
                    parsed.rolls.append(TimedSpan(open_span[1], note_time, open_span[0]))
                open_span = ("roll", note_time)
            elif char in {"7", "9"}:
                if open_span is not None:
                    parsed.rolls.append(TimedSpan(open_span[1], note_time, open_span[0]))
                open_span = ("balloon", note_time)
            elif char == "8":
                if open_span is not None and note_time >= open_span[1]:
                    parsed.rolls.append(TimedSpan(open_span[1], note_time, open_span[0]))
                open_span = None
    return time_ms + duration_ms, open_span


def parse_measure_arg(arg: str, current_ratio: float) -> float:
    if "/" not in arg:
        return current_ratio
    left, right = arg.split("/", 1)
    numerator = parse_float(left, 4.0)
    denominator = parse_float(right, 4.0)
    if denominator == 0:
        return current_ratio
    return numerator / denominator


def parse_tja_course(path: Path, course_name: str, discard_branch: bool = True) -> ParsedCourse:
    raw_course = select_course(read_raw_courses(path), course_name)
    if discard_branch and any(BRANCH_RE.match(line) for line in raw_course.body):
        raise BranchUnsupportedError(f"Branch commands are not supported: {path} [{course_name}]")

    meta = raw_course.meta
    initial_bpm = parse_float(meta.get("BPM", "120"), 120.0)
    parsed = ParsedCourse(
        path=str(path),
        title=meta.get("TITLE", ""),
        course=meta.get("COURSE", ""),
        level=meta.get("LEVEL", ""),
        initial_bpm=initial_bpm,
        bpm_changes=[TimedValue(0.0, initial_bpm)],
        scroll_changes=[TimedValue(0.0, 1.0)],
        measure_changes=[TimedValue(0.0, 1.0)],
    )

    time_ms = 0.0
    bpm = initial_bpm
    scroll = 1.0
    measure_ratio = 1.0
    barline_on = True
    gogo_start: float | None = None
    open_span: tuple[str, float] | None = None
    measure_chars: list[str] = []

    def flush_measure() -> None:
        nonlocal time_ms, open_span, measure_chars
        time_ms, open_span = finalize_measure(
            measure_chars,
            parsed,
            time_ms,
            bpm,
            measure_ratio,
            barline_on,
            open_span,
        )
        measure_chars = []

    for raw_line in raw_course.body:
        line = strip_inline_comment(raw_line)
        if not line:
            continue
        if BRANCH_RE.match(line):
            if discard_branch:
                raise BranchUnsupportedError(f"Branch commands are not supported: {path} [{course_name}]")
            continue

        if line.startswith("#"):
            parts = line.split(maxsplit=1)
            command = parts[0].upper()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if command == "#BPMCHANGE":
                bpm = parse_float(arg, bpm)
                parsed.bpm_changes.append(TimedValue(time_ms, bpm))
            elif command == "#SCROLL":
                scroll = parse_float(arg, scroll)
                parsed.scroll_changes.append(TimedValue(time_ms, scroll))
            elif command == "#MEASURE":
                measure_ratio = parse_measure_arg(arg, measure_ratio)
                parsed.measure_changes.append(TimedValue(time_ms, measure_ratio))
            elif command == "#GOGOSTART":
                gogo_start = time_ms
            elif command == "#GOGOEND":
                if gogo_start is not None and time_ms >= gogo_start:
                    parsed.gogo.append(TimedSpan(gogo_start, time_ms, "gogo"))
                gogo_start = None
            elif command == "#BARLINEOFF":
                barline_on = False
            elif command == "#BARLINEON":
                barline_on = True
            elif command == "#DELAY":
                time_ms += parse_float(arg, 0.0) * 1000.0
            continue

        for char in line:
            if char == ",":
                flush_measure()
            elif char.isdigit():
                measure_chars.append(char)

    if measure_chars:
        flush_measure()
    if gogo_start is not None and time_ms >= gogo_start:
        parsed.gogo.append(TimedSpan(gogo_start, time_ms, "gogo"))
    if open_span is not None and time_ms >= open_span[1]:
        parsed.rolls.append(TimedSpan(open_span[1], time_ms, open_span[0]))
    parsed.duration_ms = time_ms
    return parsed

