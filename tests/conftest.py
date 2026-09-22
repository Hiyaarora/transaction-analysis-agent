"""Shared fixtures.

Tests build small synthetic CSVs in `tmp_path` so every assertion is against
values the test itself wrote. Nothing about the supplied dataset's contents is
baked into the application; the one test that touches the real file only
checks structure.
"""

from pathlib import Path

import pytest

HEADER = "id,date,region,product,units,unit_price,discount,question"


@pytest.fixture
def write_csv(tmp_path: Path):
    """Return a helper that writes `lines` (list of str) to a CSV and returns its path."""

    def _write(lines: list[str], header: str = HEADER, name: str = "data.csv") -> Path:
        path = tmp_path / name
        path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return path

    return _write


@pytest.fixture
def mixed_csv(write_csv) -> Path:
    """Three transactions followed by two question rows — the supplied file's shape."""
    return write_csv(
        [
            "T1,2026-01-03,UK,Alpha,10,100,0.10,",
            "T2,2026-02-05,DE,Beta,5,200,0.00,",
            "T3,2026-03-11,UK,Beta,8,200,0.05,",
            'Q1,,,,,,,"What is the total revenue?"',
            'Q2,,,,,,,"How many Beta transactions?"',
        ]
    )
