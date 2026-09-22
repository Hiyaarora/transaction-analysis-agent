"""Describe the loaded transactions: the runtime side of the contract.

`build_profile` looks at the actual DataFrame and records what is there —
which values each categorical column holds, the span of the dates, the range
of each numeric column, how many cells are missing. Later layers consult this
object to answer "does this column exist? is 'UK' a real region? is this date
inside the data?" without ever reading rows themselves.

Nothing here is remembered between runs. Load a different file, get a
different profile.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app import contract


@dataclass(frozen=True)
class DataProfile:
    #: File the profile describes. A bare name, never a path: the planner is
    #: not shown it, and nothing resolves it back to the filesystem.
    source_name: str
    row_count: int
    columns: tuple[str, ...]
    categorical_values: dict[str, tuple[str, ...]]
    numeric_ranges: dict[str, tuple[float, float] | None]
    date_range: tuple[date, date] | None
    null_counts: dict[str, int]
    extra_columns: tuple[str, ...]
    supported_metrics: tuple[str, ...]
    # Values that were present in the file but failed to parse. Kept apart
    # from `null_counts` (which also includes genuinely empty cells) so a
    # dirty file can be reported honestly.
    parse_error_counts: dict[str, int] = field(default_factory=dict)


def build_profile(
    tx: pd.DataFrame, parse_errors: dict[str, int] | None = None, source_name: str = ""
) -> DataProfile:
    categorical_values = {
        col: tuple(sorted(tx[col].dropna().unique())) for col in contract.CATEGORICAL_COLUMNS
    }
    numeric_ranges = {col: _range(tx[col]) for col in contract.NUMERIC_COLUMNS}

    dates = tx[contract.DATE_COLUMN].dropna()
    date_range = (dates.min().date(), dates.max().date()) if not dates.empty else None

    # The loader normalises every kind of empty cell to NA, so `isna()` is the
    # single definition of "missing" for text, numeric and date columns alike.
    null_counts = {col: int(tx[col].isna().sum()) for col in tx.columns}

    extra_columns = tuple(c for c in tx.columns if c not in contract.REQUIRED_COLUMNS)
    supported_metrics = (*contract.NUMERIC_COLUMNS, *contract.DERIVED_METRICS)

    return DataProfile(
        source_name=source_name,
        row_count=len(tx),
        columns=tuple(tx.columns),
        categorical_values=categorical_values,
        numeric_ranges=numeric_ranges,
        date_range=date_range,
        null_counts=null_counts,
        extra_columns=extra_columns,
        supported_metrics=supported_metrics,
        parse_error_counts=dict(parse_errors or {}),
    )


def _range(series: pd.Series) -> tuple[float, float] | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.min()), float(values.max())
