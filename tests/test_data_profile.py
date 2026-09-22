"""Profile behaviour: transactions DataFrame -> DataProfile.

The profile is what the validator and planner will consult. It must describe
the loaded data — never a remembered list of values.
"""

from datetime import date

import numpy as np
import pandas as pd

from app.data_profile import DataProfile, build_profile


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="ISO8601")
    for c in ("units", "unit_price", "discount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _row(**overrides) -> dict:
    base = dict(id="T1", date="2026-01-03", region="UK", product="Alpha", units=10, unit_price=100, discount=0.1)
    base.update(overrides)
    return base


def test_row_count():
    p = build_profile(_frame([_row(), _row(id="T2")]))
    assert isinstance(p, DataProfile)
    assert p.row_count == 2


def test_categorical_values_are_discovered_and_sorted():
    df = _frame([_row(region="ZZ", product="Omega"), _row(region="AA", product="Alpha"), _row(region="ZZ")])
    p = build_profile(df)
    assert p.categorical_values == {"region": ("AA", "ZZ"), "product": ("Alpha", "Omega")}


def test_categorical_values_exclude_missing():
    df = _frame([_row(region="UK"), _row(region=None)])
    p = build_profile(df)
    assert p.categorical_values["region"] == ("UK",)
    assert p.null_counts["region"] == 1


def test_date_range():
    df = _frame([_row(date="2026-03-01"), _row(date="2025-12-25"), _row(date=None)])
    p = build_profile(df)
    assert p.date_range == (date(2025, 12, 25), date(2026, 3, 1))


def test_date_range_is_none_when_all_dates_missing():
    p = build_profile(_frame([_row(date=None)]))
    assert p.date_range is None


def test_numeric_ranges():
    df = _frame([_row(units=2, discount=0.0), _row(units=20, discount=0.2), _row(units=None)])
    p = build_profile(df)
    assert p.numeric_ranges["units"] == (2.0, 20.0)
    assert p.numeric_ranges["discount"] == (0.0, 0.2)


def test_numeric_range_is_none_when_all_missing():
    p = build_profile(_frame([_row(discount=None)]))
    assert p.numeric_ranges["discount"] is None


def test_null_counts_cover_every_column():
    df = _frame([_row(units=None, region=None), _row()])
    p = build_profile(df)
    assert p.null_counts == {"id": 0, "date": 0, "region": 1, "product": 0, "units": 1, "unit_price": 0, "discount": 0}


def test_extra_columns_are_listed_but_not_given_a_role():
    df = _frame([_row(zone="north")])
    p = build_profile(df)
    assert p.extra_columns == ("zone",)
    assert "zone" not in p.categorical_values
    assert "zone" not in p.numeric_ranges
    assert "zone" in p.columns


def test_supported_metrics_include_derived_revenue():
    p = build_profile(_frame([_row()]))
    assert "revenue" in p.supported_metrics


def test_parse_error_counts_are_carried_through():
    p = build_profile(_frame([_row()]), parse_errors={"units": 3})
    assert p.parse_error_counts == {"units": 3}
    assert build_profile(_frame([_row()])).parse_error_counts == {}
