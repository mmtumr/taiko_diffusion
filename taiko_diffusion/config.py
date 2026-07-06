from __future__ import annotations

from pathlib import Path


def parse_scalar(text: str):
    text = text.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"').strip("'")


def load_simple_yaml(path: Path) -> dict:
    result: dict = {}
    current_section: dict | None = None
    current_list_key: str | None = None
    current_map_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"Unsupported config line: {raw_line}")
            key = stripped[:-1]
            result[key] = {}
            current_section = result[key]
            current_list_key = None
            current_map_key = None
            continue

        if current_section is None:
            raise ValueError(f"Config value without section: {raw_line}")

        if indent == 2:
            if ":" not in stripped:
                raise ValueError(f"Unsupported config line: {raw_line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_section[key] = []
                current_list_key = key
                current_map_key = key
            else:
                current_section[key] = parse_scalar(value)
                current_list_key = None
                current_map_key = None
            continue

        if indent == 4 and stripped.startswith("- ") and current_list_key is not None:
            current_section[current_list_key].append(parse_scalar(stripped[2:]))
            continue

        if indent == 4 and current_map_key is not None and ":" in stripped:
            if isinstance(current_section[current_map_key], list):
                current_section[current_map_key] = {}
            key, value = stripped.split(":", 1)
            current_section[current_map_key][key.strip()] = parse_scalar(value)
            continue

        raise ValueError(f"Unsupported config line: {raw_line}")

    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    try:
        import yaml

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except ModuleNotFoundError:
        return load_simple_yaml(path)
