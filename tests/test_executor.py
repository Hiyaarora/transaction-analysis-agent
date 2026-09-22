"""Executor: a validated plan and a DataFrame in, exact facts out. No LLM."""

import pandas as pd
import pytest

from app.data_profile import build_profile
from app.executor import ExecutionError, ExecutionResult, StepRecord, execute
from app.schemas import parse_plan
from app.validator import validate


@pytest.fixture
def tx() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": ["T1", "T2", "T3", "T4"],
            "date": pd.to_datetime(["2026-01-03", "2026-01-20", "2026-02-05", "2026-03-01"]),
            "region": ["UK", "DE", "UK", "FR"],
            "product": ["Alpha", "Beta", "Beta", "Gamma"],
            "units": [10.0, 5.0, 8.0, 2.0],
            "unit_price": [100.0, 200.0, 200.0, 500.0],
            "discount": [0.10, 0.00, 0.20, 0.20],  # revenue 900, 1000, 1280, 800
        }
    )
    for c in ("id", "region", "product"):
        df[c] = df[c].astype("str")
    return df


def run(tx, *steps) -> ExecutionResult:
    plan = parse_plan({"status": "success", "intent": "t", "steps": list(steps)})
    outcome = validate(plan, build_profile(tx))
    assert outcome.status == "valid", outcome
    return execute(outcome.validated_plan, tx)


F = lambda column, op, value=None: {"tool": "filter_rows", "column": column, "op": op, "value": value}  # noqa: E731
COMPUTE = {"tool": "compute_metric", "metric": "revenue"}
COUNT = {"tool": "aggregate", "column": "id", "func": "count"}


# --- results ----------------------------------------------------------------------


def test_scalar_result(tx):
    r = run(tx, F("region", "eq", "UK"), COMPUTE, {"tool": "aggregate", "column": "revenue", "func": "sum"})
    assert r.kind == "scalar"
    assert r.scalar.value == 2180.0
    assert (r.scalar.rows_total, r.scalar.rows_used) == (2, 2)
    assert r.no_data is False


def test_count_result(tx):
    r = run(tx, F("units", "gt", 5), COUNT)
    assert r.scalar.value == 2


def test_grouped_result(tx):
    r = run(tx, COMPUTE, {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"})
    assert r.kind == "groups"
    assert r.groups.rows == [("DE", 1000.0, 1, 1), ("FR", 800.0, 1, 1), ("UK", 2180.0, 2, 2)]


def test_filter_groups_then_extreme(tx):
    r = run(
        tx,
        COMPUTE,
        {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
        {"tool": "filter_groups", "op": "lt", "value": 2000},
        {"tool": "select_extreme", "mode": "highest"},
    )
    assert r.kind == "extreme"
    assert r.extreme.groups == ("DE",)
    assert r.extreme.value == 1000.0


def test_group_count_without_column(tx):
    r = run(tx, {"tool": "group_by", "by": "product", "func": "count"})
    assert r.groups.rows == [("Alpha", 1, 1, 1), ("Beta", 2, 2, 2), ("Gamma", 1, 1, 1)]


# --- no_data ----------------------------------------------------------------------


def test_sum_over_no_rows_is_no_data(tx):
    r = run(tx, F("date", "between", ["2020-01-01", "2020-12-31"]), COMPUTE,
            {"tool": "aggregate", "column": "revenue", "func": "sum"})
    assert r.no_data is True
    assert r.scalar.value is None


def test_count_over_no_rows_is_zero_not_no_data(tx):
    r = run(tx, F("region", "eq", "FR"), F("product", "eq", "Alpha"), COUNT)
    assert r.scalar.value == 0
    assert r.no_data is False


def test_empty_group_filter_is_no_data(tx):
    r = run(tx, {"tool": "group_by", "by": "region", "func": "count"},
            {"tool": "filter_groups", "op": "gt", "value": 100})
    assert r.no_data is True
    assert r.groups.rows == []


def test_extreme_with_nothing_to_compare_is_no_data():
    df = pd.DataFrame(
        {"id": ["A"], "date": pd.to_datetime(["2026-01-01"]), "region": ["UK"], "product": ["Alpha"],
         "units": [float("nan")], "unit_price": [1.0], "discount": [0.0]}
    )
    r = run(df, {"tool": "group_by", "by": "region", "func": "sum", "column": "units"},
            {"tool": "select_extreme", "mode": "highest"})
    assert r.no_data is True


# --- step records ---------------------------------------------------------------


def test_step_records_carry_row_counts(tx):
    r = run(tx, F("region", "eq", "UK"), COMPUTE, {"tool": "aggregate", "column": "revenue", "func": "sum"})
    assert [s.tool for s in r.steps] == ["filter_rows", "compute_metric", "aggregate"]
    assert r.steps[0] == StepRecord(tool="filter_rows", args={"column": "region", "op": "eq", "value": "UK"},
                                    rows_in=4, rows_out=2)
    assert (r.steps[1].rows_in, r.steps[1].rows_out) == (2, 2)
    assert r.steps[2].rows_in == 2 and r.steps[2].rows_out is None


def test_group_step_records_carry_group_counts(tx):
    r = run(tx, {"tool": "group_by", "by": "region", "func": "count"},
            {"tool": "filter_groups", "op": "gt", "value": 1})
    assert (r.steps[0].rows_in, r.steps[0].groups_out) == (4, 3)
    assert (r.steps[1].groups_in, r.steps[1].groups_out) == (3, 1)


# --- guards -----------------------------------------------------------------------


def test_input_frame_is_not_mutated(tx):
    before = tx.copy()
    run(tx, F("region", "eq", "UK"), COMPUTE, {"tool": "aggregate", "column": "revenue", "func": "sum"})
    pd.testing.assert_frame_equal(tx, before)


def test_non_success_plan_is_refused(tx):
    plan = parse_plan({"status": "rejected", "intent": "x", "rejection_reason": "no"})
    with pytest.raises(ExecutionError, match="success"):
        execute(plan, tx)


def test_tool_error_after_validation_is_an_execution_error(tx):
    # Validated against a frame that has 'units', executed against one that does not:
    # the tool layer's own check must turn into ExecutionError, never a crash or a guess.
    plan = parse_plan({"status": "success", "intent": "t", "steps": [{"tool": "aggregate", "column": "units", "func": "sum"}]})
    validated = validate(plan, build_profile(tx)).validated_plan
    with pytest.raises(ExecutionError, match="units"):
        execute(validated, tx.drop(columns=["units"]))
