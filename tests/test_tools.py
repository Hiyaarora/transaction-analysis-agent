"""Deterministic tool layer: pure pandas, exact numbers, no LLM.

Frames are built inline so every expected value is visible next to the
assertion. Missing values are represented exactly as the loader produces
them: NA/NaN/NaT.
"""

from datetime import date

import pandas as pd
import pytest

from app.tools import (
    AggregateResult,
    ExtremeResult,
    GroupResult,
    ToolError,
    aggregate,
    compute_metric,
    filter_groups,
    filter_rows,
    group_by,
    select_extreme,
)


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="ISO8601")
    for c in ("units", "unit_price", "discount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("id", "region", "product"):
        df[c] = df[c].astype("str").replace("", pd.NA)
    return df


def row(**o) -> dict:
    base = dict(id="T1", date="2026-01-03", region="UK", product="Alpha", units=10, unit_price=100, discount=0.1)
    base.update(o)
    return base


@pytest.fixture
def tx() -> pd.DataFrame:
    return frame(
        [
            row(id="T1", date="2026-01-03", region="UK", product="Alpha", units=10, unit_price=100, discount=0.10),  # rev 900
            row(id="T2", date="2026-01-20", region="DE", product="Beta", units=5, unit_price=200, discount=0.00),   # rev 1000
            row(id="T3", date="2026-02-05", region="UK", product="Beta", units=8, unit_price=200, discount=0.20),   # rev 1280
            row(id="T4", date="2026-03-01", region="FR", product="Gamma", units=2, unit_price=500, discount=0.20),  # rev 800
        ]
    )


# --- filter --------------------------------------------------------------------


def test_filter_eq_categorical(tx):
    out = filter_rows(tx, column="region", op="eq", value="UK")
    assert out["id"].tolist() == ["T1", "T3"]


def test_filter_neq_and_in(tx):
    assert filter_rows(tx, "region", "neq", "UK")["id"].tolist() == ["T2", "T4"]
    assert filter_rows(tx, "product", "in", ["Beta", "Gamma"])["id"].tolist() == ["T2", "T3", "T4"]


def test_filter_numeric_comparisons(tx):
    assert filter_rows(tx, "units", "gt", 5)["id"].tolist() == ["T1", "T3"]
    assert filter_rows(tx, "units", "gte", 5)["id"].tolist() == ["T1", "T2", "T3"]
    assert filter_rows(tx, "units", "lt", 5)["id"].tolist() == ["T4"]
    assert filter_rows(tx, "units", "lte", 5)["id"].tolist() == ["T2", "T4"]


def test_filter_numeric_eq_uses_tolerance(tx):
    # 0.3 - 0.1 == 0.19999999999999998 in floating point; a 20% discount must still match 0.20
    assert (0.3 - 0.1) != 0.2  # the premise of the test
    assert filter_rows(tx, "discount", "eq", 0.3 - 0.1)["id"].tolist() == ["T3", "T4"]
    assert filter_rows(tx, "discount", "neq", 0.3 - 0.1)["id"].tolist() == ["T1", "T2"]


def test_filter_between_is_inclusive_for_numbers(tx):
    assert filter_rows(tx, "units", "between", [5, 8])["id"].tolist() == ["T2", "T3"]


def test_filter_between_is_inclusive_for_dates(tx):
    out = filter_rows(tx, "date", "between", [date(2026, 1, 20), date(2026, 2, 5)])
    assert out["id"].tolist() == ["T2", "T3"]


def test_filter_date_comparison_accepts_iso_string(tx):
    assert filter_rows(tx, "date", "gte", "2026-02-01")["id"].tolist() == ["T3", "T4"]


def test_filter_is_null_and_not_null():
    df = frame([row(id="A", discount=None), row(id="B"), row(id="C", region=None)])
    assert filter_rows(df, "discount", "is_null", None)["id"].tolist() == ["A"]
    assert filter_rows(df, "discount", "not_null", None)["id"].tolist() == ["B", "C"]
    assert filter_rows(df, "region", "is_null", None)["id"].tolist() == ["C"]


def test_filter_missing_values_never_match_comparisons():
    df = frame([row(id="A", units=None), row(id="B", units=3)])
    assert filter_rows(df, "units", "gt", 0)["id"].tolist() == ["B"]
    assert filter_rows(df, "units", "neq", 3)["id"].tolist() == []


def test_filter_does_not_mutate_input(tx):
    before = tx.copy()
    filter_rows(tx, "region", "eq", "UK")
    pd.testing.assert_frame_equal(tx, before)


def test_filter_rejects_unknown_column(tx):
    with pytest.raises(ToolError, match="profit"):
        filter_rows(tx, "profit", "gt", 0)


def test_filter_rejects_unknown_operator(tx):
    with pytest.raises(ToolError, match="like"):
        filter_rows(tx, "region", "like", "U%")


def test_filter_rejects_ordering_on_text_column(tx):
    with pytest.raises(ToolError, match="gt"):
        filter_rows(tx, "region", "gt", "A")


def test_filter_rejects_wrong_value_shape(tx):
    with pytest.raises(ToolError):
        filter_rows(tx, "units", "between", [1])
    with pytest.raises(ToolError):
        filter_rows(tx, "units", "gt", "many")
    with pytest.raises(ToolError):
        filter_rows(tx, "date", "gt", "Mars")


# --- compute -------------------------------------------------------------------


def test_compute_revenue_formula(tx):
    out = compute_metric(tx, "revenue")
    assert out["revenue"].tolist() == [900.0, 1000.0, 1280.0, 800.0]
    assert "revenue" not in tx.columns  # input untouched


def test_compute_revenue_is_missing_when_any_input_missing():
    df = frame([row(id="A", discount=None), row(id="B", units=None), row(id="C")])
    out = compute_metric(df, "revenue")
    assert out["revenue"].isna().tolist() == [True, True, False]
    assert out["revenue"].iloc[2] == 900.0


def test_compute_rejects_unknown_metric(tx):
    with pytest.raises(ToolError, match="profit"):
        compute_metric(tx, "profit")


def test_compute_rejects_missing_input_column(tx):
    with pytest.raises(ToolError, match="discount"):
        compute_metric(tx.drop(columns=["discount"]), "revenue")


# --- aggregate -----------------------------------------------------------------


def test_aggregate_functions(tx):
    assert aggregate(tx, "units", "sum") == AggregateResult(value=25.0, rows_total=4, rows_used=4)
    assert aggregate(tx, "units", "mean").value == 6.25
    assert aggregate(tx, "units", "median").value == 6.5
    assert aggregate(tx, "units", "min").value == 2.0
    assert aggregate(tx, "units", "max").value == 10.0


def test_aggregate_count_counts_rows_not_values():
    df = frame([row(id="A", discount=None), row(id="B")])
    assert aggregate(df, "discount", "count") == AggregateResult(value=2, rows_total=2, rows_used=2)


def test_aggregate_skips_missing_and_reports_rows_used():
    df = frame([row(id="A", units=None), row(id="B", units=4), row(id="C", units=6)])
    assert aggregate(df, "units", "mean") == AggregateResult(value=5.0, rows_total=3, rows_used=2)


def test_aggregate_all_missing_gives_none_value():
    df = frame([row(id="A", units=None)])
    assert aggregate(df, "units", "sum") == AggregateResult(value=None, rows_total=1, rows_used=0)


def test_aggregate_empty_frame_gives_none_value(tx):
    empty = filter_rows(tx, "region", "eq", "ZZ")
    assert aggregate(empty, "units", "sum") == AggregateResult(value=None, rows_total=0, rows_used=0)
    assert aggregate(empty, "units", "count") == AggregateResult(value=0, rows_total=0, rows_used=0)


def test_aggregate_rejects_unknown_function_and_column(tx):
    with pytest.raises(ToolError, match="variance"):
        aggregate(tx, "units", "variance")
    with pytest.raises(ToolError, match="profit"):
        aggregate(tx, "profit", "sum")


def test_aggregate_rejects_numeric_function_on_text_column(tx):
    with pytest.raises(ToolError, match="region"):
        aggregate(tx, "region", "sum")


# --- group_by ------------------------------------------------------------------


def test_group_by_sum(tx):
    with_rev = compute_metric(tx, "revenue")
    result = group_by(with_rev, by="region", column="revenue", func="sum")
    assert isinstance(result, GroupResult)
    assert result.by == "region"
    assert result.rows == [("DE", 1000.0, 1, 1), ("FR", 800.0, 1, 1), ("UK", 2180.0, 2, 2)]


def test_group_by_median(tx):
    result = group_by(tx, "product", "units", "median")
    assert [(g, v) for g, v, *_ in result.rows] == [("Alpha", 10.0), ("Beta", 6.5), ("Gamma", 2.0)]


def test_group_by_reports_rows_used_per_group():
    df = frame([row(region="UK", units=None), row(region="UK", units=4), row(region="DE", units=None)])
    result = group_by(df, "region", "units", "mean")
    assert result.rows == [("DE", None, 1, 0), ("UK", 4.0, 2, 1)]


def test_group_by_excludes_rows_with_missing_group_key():
    df = frame([row(region=None, units=1), row(region="UK", units=4)])
    result = group_by(df, "region", "units", "sum")
    assert result.rows == [("UK", 4.0, 1, 1)]
    assert result.rows_without_group == 1


def test_group_by_rejects_non_categorical_key(tx):
    with pytest.raises(ToolError, match="units"):
        group_by(tx, "units", "unit_price", "sum")


# --- select_extreme ------------------------------------------------------------


def test_select_extreme_highest_and_lowest(tx):
    grouped = group_by(compute_metric(tx, "revenue"), "region", "revenue", "sum")
    assert select_extreme(grouped, "highest") == ExtremeResult(mode="highest", groups=("UK",), value=2180.0)
    assert select_extreme(grouped, "lowest") == ExtremeResult(mode="lowest", groups=("FR",), value=800.0)


def test_select_extreme_reports_all_ties():
    df = frame([row(region="UK", units=5), row(region="DE", units=5), row(region="FR", units=1)])
    grouped = group_by(df, "region", "units", "sum")
    assert select_extreme(grouped, "highest").groups == ("DE", "UK")


def test_select_extreme_ignores_groups_without_a_value():
    df = frame([row(region="UK", units=None), row(region="DE", units=3)])
    grouped = group_by(df, "region", "units", "sum")
    assert select_extreme(grouped, "lowest").groups == ("DE",)


def test_select_extreme_on_empty_result_gives_none():
    df = frame([row(region="UK", units=None)])
    grouped = group_by(df, "region", "units", "sum")
    assert select_extreme(grouped, "highest") == ExtremeResult(mode="highest", groups=(), value=None)


def test_select_extreme_rejects_unknown_mode(tx):
    grouped = group_by(tx, "region", "units", "sum")
    with pytest.raises(ToolError, match="biggest"):
        select_extreme(grouped, "biggest")


# --- consistency on the supplied file (no literal values) --------------------


def test_group_totals_add_up_on_supplied_dataset():
    from app.data_loader import load_dataset

    tx = compute_metric(load_dataset("data/project_4.csv").transactions, "revenue")
    total = aggregate(tx, "revenue", "sum").value
    by_region = group_by(tx, "region", "revenue", "sum")
    assert sum(v for _, v, *_ in by_region.rows) == pytest.approx(total)
    assert sum(n for _, _, n, _ in by_region.rows) == len(tx)


# --- group_by count without a column (amendment) -------------------------------


def test_group_by_count_needs_no_column(tx):
    result = group_by(tx, by="region", func="count")
    assert result.column is None
    assert result.rows == [("DE", 1, 1, 1), ("FR", 1, 1, 1), ("UK", 2, 2, 2)]


def test_group_by_non_count_still_requires_column(tx):
    with pytest.raises(ToolError, match="column"):
        group_by(tx, by="region", func="sum")


# --- filter_groups (amendment) --------------------------------------------------


@pytest.fixture
def by_region(tx) -> GroupResult:
    # DE 1000, FR 800, UK 2180
    return group_by(compute_metric(tx, "revenue"), "region", "revenue", "sum")


def _groups(result: GroupResult) -> list[str]:
    return [g for g, *_ in result.rows]


def test_filter_groups_gt_and_lt(by_region):
    assert _groups(filter_groups(by_region, "gt", 900)) == ["DE", "UK"]
    assert _groups(filter_groups(by_region, "lt", 900)) == ["FR"]


def test_filter_groups_gte_lte_are_inclusive(by_region):
    assert _groups(filter_groups(by_region, "gte", 1000)) == ["DE", "UK"]
    assert _groups(filter_groups(by_region, "lte", 1000)) == ["DE", "FR"]


def test_filter_groups_eq_neq_use_tolerance(by_region):
    almost = 1000 + 1e-12
    assert _groups(filter_groups(by_region, "eq", almost)) == ["DE"]
    assert _groups(filter_groups(by_region, "neq", almost)) == ["FR", "UK"]


def test_filter_groups_between_is_inclusive(by_region):
    assert _groups(filter_groups(by_region, "between", [800, 1000])) == ["DE", "FR"]


def test_filter_groups_preserves_group_metadata_and_row_counts(by_region):
    out = filter_groups(by_region, "gt", 900)
    assert isinstance(out, GroupResult)
    assert (out.by, out.column, out.func) == ("region", "revenue", "sum")
    assert out.rows == [("DE", 1000.0, 1, 1), ("UK", 2180.0, 2, 2)]


def test_filter_groups_empty_result_is_valid_not_error(by_region):
    out = filter_groups(by_region, "gt", 1_000_000)
    assert out.rows == []


def test_filter_groups_excludes_and_counts_groups_without_value():
    df = frame([row(region="UK", units=None), row(region="DE", units=3), row(region="FR", units=9)])
    grouped = group_by(df, "region", "units", "sum")  # UK has value None
    out = filter_groups(grouped, "gte", 0)
    assert _groups(out) == ["DE", "FR"]
    assert out.groups_without_value == 1


def test_filter_groups_on_counts(tx):
    counts = group_by(tx, by="region", func="count")
    assert _groups(filter_groups(counts, "gt", 1)) == ["UK"]


def test_filter_groups_chains_into_select_extreme(by_region):
    kept = filter_groups(by_region, "lt", 2000)  # DE, FR
    assert select_extreme(kept, "highest").groups == ("DE",)


def test_filter_groups_rejects_row_only_operators(by_region):
    for op in ("in", "is_null", "not_null", "like"):
        with pytest.raises(ToolError, match=op):
            filter_groups(by_region, op, [1])


def test_filter_groups_rejects_non_numeric_value(by_region):
    with pytest.raises(ToolError):
        filter_groups(by_region, "gt", "many")
    with pytest.raises(ToolError):
        filter_groups(by_region, "gt", True)


def test_filter_groups_rejects_bad_between_shape(by_region):
    with pytest.raises(ToolError, match="between"):
        filter_groups(by_region, "between", [1])
