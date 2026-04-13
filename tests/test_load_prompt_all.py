from pathlib import Path
import sys


# Allow importing from src-layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.utils.data import EXPECTED_PROMPT_ALL_ROWS, load_prompt_all


def test_load_prompt_all_returns_expected_records() -> None:
    rows = load_prompt_all()

    assert isinstance(rows, list)
    assert len(rows) == EXPECTED_PROMPT_ALL_ROWS

    first = rows[0]
    assert isinstance(first, dict)
    assert "id" in first
    assert "prompt_a" in first
    assert "prompt_b" in first
