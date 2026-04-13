import json
from importlib.resources import files
from typing import Any, Dict, List

EXPECTED_PROMPT_ALL_ROWS = 337


def download_dataset() -> None:
    raise NotImplementedError("download_dataset is not implemented. Prepare datasets externally.")


def check_integrity() -> None:
    raise NotImplementedError("check_integrity is not implemented. Validate dataset integrity externally.")


def load_dataset() -> None:
    raise NotImplementedError("load_dataset is not implemented. Use load_prompt_all or custom loaders.")


def load_jsonl_resource(file_name: str) -> List[Dict[str, Any]]:
    """Load a JSONL resource from phyavbench/data in strict mode.

    Raises:
        FileNotFoundError: The resource file does not exist.
        ValueError: A non-empty line cannot be parsed as JSON object.
    """
    resource = files("phyavbench").joinpath("data", file_name)
    if not resource.is_file():
        raise FileNotFoundError(f"Resource not found: phyavbench/data/{file_name}")

    rows: List[Dict[str, Any]] = []
    content = resource.read_text(encoding="utf-8")
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in phyavbench/data/{file_name} at line {line_no}: {exc.msg}") from exc

        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSONL in phyavbench/data/{file_name} at line {line_no}: expected JSON object")

        rows.append(value)

    return rows


def load_prompt_all() -> List[Dict[str, Any]]:
    """Load the packaged prompt_all.jsonl file with strict row-count validation."""
    rows = load_jsonl_resource("prompt_all.jsonl")
    if len(rows) != EXPECTED_PROMPT_ALL_ROWS:
        raise ValueError(
            f"prompt_all.jsonl must contain exactly {EXPECTED_PROMPT_ALL_ROWS} JSON objects, got {len(rows)}"
        )

    return rows
