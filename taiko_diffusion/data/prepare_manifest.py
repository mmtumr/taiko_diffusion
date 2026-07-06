from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import openpyxl


BRANCH_RE = re.compile(
    r"^\s*#(BRANCHSTART|BRANCHEND|SECTION|LEVELHOLD|N\b|E\b|M\b)", re.IGNORECASE
)
HEADER_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$")
NOTE_CHARS = set("1234")
ROLL_START_CHARS = set("567")


@dataclass
class RatingRow:
    rating_index: int
    title: str
    is_ura: bool
    const: float
    combo: int
    complex: float
    avg_density: float
    peak_density: float
    note_type: float
    bpm_change: float
    hs_change: float
    rhythm: float
    roll_time: str
    balloon_num: int


@dataclass
class EseCourse:
    path: str
    category: str
    title: str
    title_ja: str
    title_zh: str
    title_ko: str
    course: str
    level: str
    is_ura: bool
    has_branch: bool
    note_count: int
    roll_start_count: int
    balloon_declared: str
    normalized_keys: str


def clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def strip_ura_marker(title: str) -> tuple[str, bool]:
    markers = ["(裏)", "（裏）", "【裏】", "[裏]", "(Ura)", "(ura)", " Ura", " ura"]
    is_ura = False
    result = title
    for marker in markers:
        if marker in result:
            result = result.replace(marker, "")
            is_ura = True
    return result.strip(), is_ura


def normalize_title(title: str) -> str:
    title, _ = strip_ura_marker(title)
    title = unicodedata.normalize("NFKC", title).casefold()
    replacements = {
        "♢": "diamond",
        "♦": "diamond",
        "†": "",
        "♡": "",
        "❤": "",
        "♥": "",
        "～": "",
        "~": "",
        "・": "",
        "･": "",
        "×": "x",
        "＋": "+",
        "！": "!",
        "？": "?",
        "ー": "",
        "-": "",
        "_": "",
        " ": "",
    }
    for src, dst in replacements.items():
        title = title.replace(src, dst)
    title = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", "", title)
    return title


def rating_title_variants(title: str) -> set[str]:
    base, _ = strip_ura_marker(title)
    variants = {base, title}
    variants.add(base.replace("〜", "～"))
    variants.add(base.replace("～", ""))
    variants.add(base.replace(" ", ""))
    return {normalize_title(x) for x in variants if x}


def tja_title_keys(*titles: str) -> set[str]:
    keys: set[str] = set()
    for title in titles:
        if title:
            keys.add(normalize_title(title))
    return {x for x in keys if x}


def load_rating_rows(path: Path) -> list[RatingRow]:
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook["歌曲数据表"]
    rows: list[RatingRow] = []
    for row_index, values in enumerate(
        worksheet.iter_rows(min_row=3, max_col=18, values_only=True), start=3
    ):
        title = clean_cell(values[0])
        if not title:
            continue
        _, is_ura = strip_ura_marker(title)
        rows.append(
            RatingRow(
                rating_index=row_index,
                title=title,
                is_ura=is_ura,
                const=as_float(values[1]),
                combo=as_int(values[2]),
                complex=as_float(values[3]),
                avg_density=as_float(values[4]),
                peak_density=as_float(values[5]),
                note_type=as_float(values[6]),
                bpm_change=as_float(values[7]),
                hs_change=as_float(values[8]),
                rhythm=as_float(values[12]),
                roll_time=clean_cell(values[16]),
                balloon_num=as_int(values[17]),
            )
        )
    return rows


def strip_inline_comment(line: str) -> str:
    if line.lstrip().startswith("//"):
        return ""
    return line.split("//", 1)[0].strip()


def chart_digits(body: list[str]) -> str:
    digits: list[str] = []
    for raw_line in body:
        line = strip_inline_comment(raw_line)
        if not line or line.startswith("#"):
            continue
        digits.extend(char for char in line if char.isdigit())
    return "".join(digits)


def parse_tja_file(path: Path, root: Path) -> list[EseCourse]:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return []

    global_meta: dict[str, str] = {}
    courses: list[EseCourse] = []
    current_meta: dict[str, str] | None = None
    current_body: list[str] = []
    in_started_body = False

    def finish_course() -> None:
        nonlocal current_meta, current_body, in_started_body
        if current_meta is None:
            return
        meta = {**global_meta, **current_meta}
        course = clean_cell(meta.get("COURSE", ""))
        if not course:
            current_meta = None
            current_body = []
            in_started_body = False
            return

        body_text = chart_digits(current_body)
        title = clean_cell(meta.get("TITLE", ""))
        title_ja = clean_cell(meta.get("TITLEJA", ""))
        title_zh = clean_cell(meta.get("TITLEZH", ""))
        title_ko = clean_cell(meta.get("TITLEKO", ""))
        title_keys = tja_title_keys(title, title_ja, title_zh, title_ko)
        rel = path.relative_to(root)
        course_name = course.casefold()
        courses.append(
            EseCourse(
                path=str(path),
                category=rel.parts[0] if rel.parts else "",
                title=title,
                title_ja=title_ja,
                title_zh=title_zh,
                title_ko=title_ko,
                course=course,
                level=clean_cell(meta.get("LEVEL", "")),
                is_ura=course_name == "edit"
                or any(marker in path.name.casefold() for marker in ["ura", "edit"])
                or "裏" in title
                or "裏" in title_ja
                or "裏" in title_zh,
                has_branch=any(BRANCH_RE.match(line) for line in current_body),
                note_count=sum(1 for ch in body_text if ch in NOTE_CHARS),
                roll_start_count=sum(1 for ch in body_text if ch in ROLL_START_CHARS),
                balloon_declared=clean_cell(meta.get("BALLOON", "")),
                normalized_keys="|".join(sorted(title_keys)),
            )
        )
        current_meta = None
        current_body = []
        in_started_body = False

    for raw_line in lines:
        line = strip_inline_comment(raw_line)
        if not line:
            continue
        header = HEADER_RE.match(line)
        if header and not in_started_body:
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
        if line.upper().startswith("#START"):
            if current_meta is not None:
                in_started_body = True
            continue
        if line.upper().startswith("#END"):
            finish_course()
            continue
        if in_started_body and current_meta is not None:
            current_body.append(line)

    finish_course()
    return courses


def scan_ese_courses(root: Path) -> list[EseCourse]:
    courses: list[EseCourse] = []
    for path in root.rglob("*.tja"):
        courses.extend(parse_tja_file(path, root))
    return courses


def course_rank(course: EseCourse, rating: RatingRow) -> tuple[int, int, int]:
    course_name = course.course.casefold()
    if rating.is_ura:
        ura_penalty = 0 if course.is_ura or course_name == "edit" else 5
    else:
        ura_penalty = 5 if course.is_ura or course_name == "edit" else 0
    preferred_course = 0 if course_name in {"oni", "edit"} else 2
    combo_delta = abs(course.note_count - rating.combo) if rating.combo > 0 else 99999
    return (ura_penalty, preferred_course, combo_delta)


def build_matches(
    rating_rows: list[RatingRow], courses: list[EseCourse]
) -> tuple[list[dict], list[dict], list[dict]]:
    key_to_courses: dict[str, list[EseCourse]] = {}
    for course in courses:
        if course.has_branch:
            continue
        if course.course.casefold() not in {"oni", "edit"}:
            continue
        for key in course.normalized_keys.split("|"):
            if key:
                key_to_courses.setdefault(key, []).append(course)

    matched: list[dict] = []
    ambiguous: list[dict] = []
    unmatched: list[dict] = []
    for rating in rating_rows:
        candidates: list[EseCourse] = []
        seen: set[tuple[str, str]] = set()
        for key in rating_title_variants(rating.title):
            for course in key_to_courses.get(key, []):
                candidate_key = (course.path, course.course)
                if candidate_key not in seen:
                    candidates.append(course)
                    seen.add(candidate_key)

        if not candidates:
            unmatched.append(asdict(rating))
            continue

        candidates = sorted(candidates, key=lambda c: course_rank(c, rating))
        best = candidates[0]
        best_rank = course_rank(best, rating)
        tied = [c for c in candidates if course_rank(c, rating) == best_rank]
        if len(tied) > 1:
            ambiguous.append(
                {
                    **asdict(rating),
                    "candidate_count": len(candidates),
                    "candidates": json.dumps(
                        [asdict(c) for c in candidates[:10]], ensure_ascii=False
                    ),
                }
            )
            continue

        matched.append(
            {
                **asdict(rating),
                "ese_path": best.path,
                "ese_category": best.category,
                "ese_title": best.title,
                "ese_title_ja": best.title_ja,
                "ese_title_zh": best.title_zh,
                "ese_course": best.course,
                "ese_level": best.level,
                "ese_is_ura": best.is_ura,
                "ese_note_count": best.note_count,
                "combo_delta": abs(best.note_count - rating.combo),
                "ese_roll_start_count": best.roll_start_count,
                "ese_balloon_declared": best.balloon_declared,
            }
        )

    return matched, ambiguous, unmatched


def split_strict_matches(
    matched: list[dict], combo_ratio: float, combo_abs_min: int
) -> tuple[list[dict], list[dict]]:
    strict: list[dict] = []
    loose: list[dict] = []
    for row in matched:
        combo = as_int(row.get("combo"))
        delta = as_int(row.get("combo_delta"))
        threshold = max(combo_abs_min, round(combo * combo_ratio))
        row = {**row, "combo_delta_threshold": threshold}
        if delta <= threshold:
            strict.append(row)
        else:
            loose.append(row)
    return strict, loose


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Taiko encoder manifests.")
    parser.add_argument("--ese-root", type=Path, required=True)
    parser.add_argument("--rating-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--strict-combo-ratio", type=float, default=0.05)
    parser.add_argument("--strict-combo-abs-min", type=int, default=20)
    args = parser.parse_args()

    rating_rows = load_rating_rows(args.rating_xlsx)
    courses = scan_ese_courses(args.ese_root)
    matched, ambiguous, unmatched = build_matches(rating_rows, courses)
    strict_matched, loose_matched = split_strict_matches(
        matched, args.strict_combo_ratio, args.strict_combo_abs_min
    )

    write_csv(args.output_dir / "rating_rows.csv", [asdict(row) for row in rating_rows])
    write_csv(args.output_dir / "ese_courses.csv", [asdict(row) for row in courses])
    write_csv(args.output_dir / "matched_dataset.csv", matched)
    write_csv(args.output_dir / "strict_matched_dataset.csv", strict_matched)
    write_csv(args.output_dir / "loose_matched_dataset.csv", loose_matched)
    write_csv(args.output_dir / "ambiguous_rating.csv", ambiguous)
    write_csv(args.output_dir / "unmatched_rating.csv", unmatched)

    skipped_branch = sum(1 for row in courses if row.has_branch)
    summary = {
        "rating_rows": len(rating_rows),
        "ese_courses": len(courses),
        "ese_courses_with_branch_skipped_for_matching": skipped_branch,
        "matched": len(matched),
        "strict_matched": len(strict_matched),
        "loose_matched_needs_review": len(loose_matched),
        "strict_combo_rule": (
            f"combo_delta <= max({args.strict_combo_abs_min}, "
            f"round(combo * {args.strict_combo_ratio}))"
        ),
        "ambiguous": len(ambiguous),
        "unmatched": len(unmatched),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
