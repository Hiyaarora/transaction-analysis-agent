"""The restricted tool interface: every deterministic operation the agent can run.

This module is the whole of what the executor may call. Six functions, each a
pure pandas operation with a closed set of options:

    filter_rows     which rows?              (eq, neq, gt, gte, lt, lte, between, in, is_null, not_null)
    compute_metric  add a derived column     (metrics from a whitelist; formulas live here in Python)
    aggregate       one number from a column (sum, mean, median, min, max, count)
    group_by        one number per category
    filter_groups   which groups satisfy a condition on that number?   (eq, neq, gt, gte, lt, lte, between)
    select_extreme  which group is highest / lowest?

There is deliberately no function that takes an expression, a formula string,
a path, or code. A caller who wants something outside these options gets a
`ToolError`, not an interpretation.

Missing-value policy (see DATA_CONTRACT.md):
  * a derived metric is missing for any row with a missing input;
  * aggregations skip missing values and report `rows_used` next to
    `rows_total`, so the answer can say "computed from 9 of 10 rows";
  * `count` counts rows, not values — "how many rows have a missing discount"
    is `filter is_null` followed by `count`;
  * a result with nothing to compute from has `value=None`.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from app import contract


class ToolError(Exception):
    """A tool was asked for something outside its contract."""


# Closed option sets. The validator (Phase 4) and the plan schema (Phase 3)
# reuse these so there is exactly one list of what is allowed.
FILTER_OPS: tuple[str, ...] = ("eq", "neq", "gt", "gte", "lt", "lte", "between", "in", "is_null", "not_null")
AGG_FUNCS: tuple[str, ...] = ("sum", "mean", "median", "min", "max", "count")
# Post-aggregation filtering compares one number per group, so the null and
# membership operators from FILTER_OPS make no sense here and are excluded.
GROUP_FILTER_OPS: tuple[str, ...] = ("eq", "neq", "gt", "gte", "lt", "lte", "between")
EXTREME_MODES: tuple[str, ...] = ("highest", "lowest")

_ORDERING_OPS = ("gt", "gte", "lt", "lte", "between")
_FLOAT_TOLERANCE = 1e-9


# --- results -------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateResult:
    value: float | int | None
    rows_total: int
    rows_used: int


@dataclass(frozen=True)
class GroupResult:
    by: str
    column: str | None  # None when func == "count": rows are counted, no value column involved
    func: str
    # (group value, aggregate or None, rows_total, rows_used) per group, sorted by group
    rows: list[tuple[str, float | int | None, int, int]]
    rows_without_group: int = 0
    # Set by filter_groups: groups skipped because they had nothing to compare.
    groups_without_value: int = 0


@dataclass(frozen=True)
class ExtremeResult:
    mode: str
    groups: tuple[str, ...]  # more than one when tied; empty when nothing to compare
    value: float | int | None


# --- filter --------------------------------------------------------------------


def filter_rows(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.DataFrame:
    _require_column(df, column)
    if op not in FILTER_OPS:
        raise ToolError(f"Unsupported filter operator '{op}'. Supported: {', '.join(FILTER_OPS)}")

    series = df[column]
    if op == "is_null":
        return df[series.isna()]
    if op == "not_null":
        return df[series.notna()]

    kind = _column_kind(series)
    if op in _ORDERING_OPS and kind == "text":
        raise ToolError(f"Operator '{op}' needs a numeric or date column; '{column}' is text.")

    if op == "in":
        if not isinstance(value, (list, tuple)) or not value:
            raise ToolError("Operator 'in' needs a non-empty list of values.")
        return df[series.isin([_coerce_value(v, kind, column) for v in value])]

    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ToolError("Operator 'between' needs exactly two values [low, high].")
        low, high = (_coerce_value(v, kind, column) for v in value)
        return df[(series >= low) & (series <= high)]  # inclusive on both ends

    target = _coerce_value(value, kind, column)
    if kind == "numeric" and op in ("eq", "neq"):
        # Float equality with a tolerance: 0.1 + 0.1 must still match 0.20.
        close = np.isclose(series.to_numpy(dtype="float64"), target, rtol=0, atol=_FLOAT_TOLERANCE)
        mask = pd.Series(close, index=series.index) & series.notna()
        return df[mask if op == "eq" else (~mask & series.notna())]

    if op == "eq":
        return df[series == target]
    if op == "neq":
        return df[(series != target) & series.notna()]
    comparison = {"gt": series.gt, "gte": series.ge, "lt": series.lt, "lte": series.le}[op]
    return df[comparison(target)]  # NaN/NaT compare False, so missing rows never match


# --- compute -------------------------------------------------------------------


def compute_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a copy of `df` with the derived column added.

    The formula is Python, chosen by name. A missing input on a row makes the
    metric missing on that row; nothing is filled in.
    """
    if metric not in contract.DERIVED_METRICS:
        raise ToolError(f"Unsupported metric '{metric}'. Supported: {', '.join(contract.DERIVED_METRICS)}")
    for needed in contract.DERIVED_METRICS[metric]:
        _require_column(df, needed)

    out = df.copy()
    if metric == "revenue":
        out["revenue"] = out["units"] * out["unit_price"] * (1 - out["discount"])
    return out


# --- aggregate -----------------------------------------------------------------


def aggregate(df: pd.DataFrame, column: str, func: str) -> AggregateResult:
    _require_column(df, column)
    if func not in AGG_FUNCS:
        raise ToolError(f"Unsupported aggregation '{func}'. Supported: {', '.join(AGG_FUNCS)}")

    if func == "count":
        return AggregateResult(value=len(df), rows_total=len(df), rows_used=len(df))

    if _column_kind(df[column]) != "numeric":
        raise ToolError(f"Aggregation '{func}' needs a numeric column; '{column}' is not.")

    values = df[column].dropna()
    if values.empty:
        return AggregateResult(value=None, rows_total=len(df), rows_used=0)
    result = float(getattr(values, func)())
    return AggregateResult(value=result, rows_total=len(df), rows_used=len(values))


# --- group_by ------------------------------------------------------------------


def group_by(df: pd.DataFrame, by: str, column: str | None = None, func: str = "count") -> GroupResult:
    _require_column(df, by)
    if by not in contract.CATEGORICAL_COLUMNS:
        raise ToolError(f"Cannot group by '{by}'. Group columns: {', '.join(contract.CATEGORICAL_COLUMNS)}")
    if func not in AGG_FUNCS:
        raise ToolError(f"Unsupported aggregation '{func}'. Supported: {', '.join(AGG_FUNCS)}")
    if func == "count":
        column = None  # counting rows needs no value column
    elif column is None:
        raise ToolError(f"Aggregation '{func}' needs a column to aggregate.")
    else:
        _require_column(df, column)

    # A row whose group key is missing belongs to no group; it is excluded and
    # counted rather than silently dropped or lumped into a fake group.
    keyed = df[df[by].notna()]
    rows = [
        (str(key), *_agg_tuple(group, column, func))
        for key, group in sorted(keyed.groupby(by, sort=True), key=lambda kv: str(kv[0]))
    ]
    return GroupResult(by=by, column=column, func=func, rows=rows, rows_without_group=len(df) - len(keyed))


def _agg_tuple(group: pd.DataFrame, column: str | None, func: str) -> tuple[float | int | None, int, int]:
    if func == "count":
        return len(group), len(group), len(group)
    result = aggregate(group, column, func)
    return result.value, result.rows_total, result.rows_used


# --- filter_groups -------------------------------------------------------------


def filter_groups(grouped: GroupResult, op: str, value: Any) -> GroupResult:
    """Keep the groups whose aggregate satisfies `op value` (SQL's HAVING).

    Operates on a GroupResult, never on rows, so it composes after group_by
    and before select_extreme. Groups with no aggregate (value None) cannot be
    compared; they are left out and counted in `groups_without_value`.
    """
    if op not in GROUP_FILTER_OPS:
        raise ToolError(f"Unsupported group filter operator '{op}'. Supported: {', '.join(GROUP_FILTER_OPS)}")

    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ToolError("Operator 'between' needs exactly two values [low, high].")
        low, high = (_coerce_value(v, "numeric", grouped.column or "count") for v in value)
        keep = lambda v: low <= v <= high  # noqa: E731 — inclusive on both ends
    else:
        target = _coerce_value(value, "numeric", grouped.column or "count")
        keep = {
            "eq": lambda v: _close(v, target),
            "neq": lambda v: not _close(v, target),
            "gt": lambda v: v > target,
            "gte": lambda v: v >= target,
            "lt": lambda v: v < target,
            "lte": lambda v: v <= target,
        }[op]

    comparable = [r for r in grouped.rows if r[1] is not None]
    kept = [r for r in comparable if keep(float(r[1]))]
    return GroupResult(
        by=grouped.by,
        column=grouped.column,
        func=grouped.func,
        rows=kept,
        rows_without_group=grouped.rows_without_group,
        groups_without_value=len(grouped.rows) - len(comparable),
    )


# --- select_extreme ------------------------------------------------------------


def select_extreme(grouped: GroupResult, mode: str) -> ExtremeResult:
    if mode not in EXTREME_MODES:
        raise ToolError(f"Unsupported mode '{mode}'. Supported: {', '.join(EXTREME_MODES)}")

    candidates = [(g, v) for g, v, *_ in grouped.rows if v is not None]
    if not candidates:
        return ExtremeResult(mode=mode, groups=(), value=None)

    pick = max if mode == "highest" else min
    best = pick(v for _, v in candidates)
    # Every group at the extreme value is reported; choosing one would be a guess.
    tied = tuple(g for g, v in candidates if v == best)
    return ExtremeResult(mode=mode, groups=tied, value=best)


# --- helpers -------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _FLOAT_TOLERANCE


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ToolError(f"Unknown column '{column}'. Available: {', '.join(df.columns)}")


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


def _coerce_value(value: Any, kind: str, column: str) -> Any:
    """Turn a plan value into something comparable with the column, or refuse."""
    if kind == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolError(f"Column '{column}' is numeric; got {value!r}.")
        return float(value)
    if kind == "date":
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        if isinstance(value, date):
            return pd.Timestamp(value)
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce", format="ISO8601")
            if pd.isna(parsed):
                raise ToolError(f"Column '{column}' is a date; {value!r} is not an ISO date (YYYY-MM-DD).")
            return parsed
        raise ToolError(f"Column '{column}' is a date; got {value!r}.")
    if not isinstance(value, str):
        raise ToolError(f"Column '{column}' is text; got {value!r}.")
    return value
