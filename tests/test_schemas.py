"""Structural validation of the analysis plan (Level 1).

These tests prove what Pydantic alone rejects — before the semantic validator
or any tool sees the plan. Nothing here depends on a dataset.
"""

import json

import pytest

from app.schemas import (
    AggregateStep,
    AnalysisPlan,
    ComputeMetricStep,
    FilterGroupsStep,
    FilterRowsStep,
    GroupByStep,
    PlanParseError,
    SelectExtremeStep,
    parse_plan,
)

UK_REVENUE = {
    "status": "success",
    "intent": "Sum of revenue for transactions in region UK",
    "steps": [
        {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "aggregate", "column": "revenue", "func": "sum"},
    ],
}


# --- happy paths ---------------------------------------------------------------


def test_success_plan_parses_into_typed_steps():
    plan = parse_plan(UK_REVENUE)
    assert plan.status == "success"
    assert [type(s) for s in plan.steps] == [FilterRowsStep, ComputeMetricStep, AggregateStep]
    assert plan.steps[0].value == "UK"
    assert plan.steps[2].func == "sum"


def test_plan_parses_from_json_string():
    plan = parse_plan(json.dumps(UK_REVENUE))
    assert plan == parse_plan(UK_REVENUE)


def test_every_tool_type_in_one_plan():
    plan = parse_plan(
        {
            "status": "success",
            "intent": "Highest-revenue region among regions with revenue above 4000 in Q1",
            "steps": [
                {"tool": "filter_rows", "column": "date", "op": "between", "value": ["2026-01-01", "2026-03-31"]},
                {"tool": "compute_metric", "metric": "revenue"},
                {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
                {"tool": "filter_groups", "op": "gt", "value": 4000},
                {"tool": "select_extreme", "mode": "highest"},
            ],
        }
    )
    assert [type(s) for s in plan.steps] == [
        FilterRowsStep, ComputeMetricStep, GroupByStep, FilterGroupsStep, SelectExtremeStep
    ]


def test_group_by_count_needs_no_column():
    step = parse_plan(_success([{"tool": "group_by", "by": "product", "func": "count"}])).steps[0]
    assert isinstance(step, GroupByStep)
    assert step.column is None


def test_filter_value_is_polymorphic():
    for value in ("UK", 5, 0.2, ["Beta", "Gamma"], [1, 10], None):
        step = parse_plan(_success([{"tool": "filter_rows", "column": "x", "op": "eq", "value": value}])).steps[0]
        assert step.value == value


def test_filter_value_defaults_to_none_for_null_checks():
    step = parse_plan(_success([{"tool": "filter_rows", "column": "discount", "op": "is_null"}])).steps[0]
    assert step.value is None


def test_clarification_plan():
    plan = parse_plan(
        {
            "status": "clarification_required",
            "intent": "Revenue for an unrecognised period 'Mars'",
            "clarification_question": "I couldn't interpret 'Mars' as a month. Did you mean March?",
        }
    )
    assert plan.steps == []
    assert "March" in plan.clarification_question


def test_rejected_plan():
    plan = parse_plan(
        {
            "status": "rejected",
            "intent": "Request to execute code",
            "rejection_reason": "I can only analyse the loaded dataset; I cannot run code or read files.",
        }
    )
    assert plan.steps == []
    assert plan.rejection_reason


# --- structural rejections -----------------------------------------------------


def test_unknown_tool_is_rejected():
    with pytest.raises(PlanParseError, match="execute_python"):
        parse_plan(_success([{"tool": "execute_python", "code": "print(1)"}]))


@pytest.mark.parametrize(
    "step",
    [
        {"tool": "filter_rows", "column": "region", "op": "like", "value": "U%"},
        {"tool": "aggregate", "column": "units", "func": "variance"},
        {"tool": "group_by", "by": "region", "func": "stddev"},
        {"tool": "filter_groups", "op": "in", "value": [1]},
        {"tool": "filter_groups", "op": "is_null"},
        {"tool": "select_extreme", "mode": "biggest"},
    ],
)
def test_unknown_option_is_rejected(step):
    with pytest.raises(PlanParseError):
        parse_plan(_success([step]))


@pytest.mark.parametrize(
    "step",
    [
        {"tool": "filter_rows", "column": "units", "op": "gt", "value": 5, "expression": "os.system('ls')"},
        {"tool": "compute_metric", "metric": "revenue", "formula": "units*price"},
        {"tool": "aggregate", "column": "units", "func": "sum", "path": "/etc/passwd"},
    ],
)
def test_extra_fields_are_rejected(step):
    with pytest.raises(PlanParseError):
        parse_plan(_success([step]))


def test_extra_top_level_field_is_rejected():
    with pytest.raises(PlanParseError):
        parse_plan({**UK_REVENUE, "python": "import os"})


def test_missing_required_field_is_rejected():
    with pytest.raises(PlanParseError, match="column"):
        parse_plan(_success([{"tool": "aggregate", "func": "sum"}]))


def test_filter_groups_value_must_be_numeric():
    with pytest.raises(PlanParseError):
        parse_plan(_success([{"tool": "filter_groups", "op": "gt", "value": "many"}]))


def test_unknown_status_is_rejected():
    with pytest.raises(PlanParseError, match="status"):
        parse_plan({"status": "maybe", "intent": "?", "steps": []})


# --- status / field consistency --------------------------------------------------


def test_success_requires_at_least_one_step():
    with pytest.raises(PlanParseError, match="step"):
        parse_plan({"status": "success", "intent": "nothing", "steps": []})


def test_success_must_not_carry_rejection_or_clarification_text():
    with pytest.raises(PlanParseError):
        parse_plan({**UK_REVENUE, "rejection_reason": "but also no"})
    with pytest.raises(PlanParseError):
        parse_plan({**UK_REVENUE, "clarification_question": "are you sure?"})


def test_clarification_requires_question_and_no_steps():
    with pytest.raises(PlanParseError, match="clarification_question"):
        parse_plan({"status": "clarification_required", "intent": "?"})
    with pytest.raises(PlanParseError, match="steps"):
        parse_plan(
            {"status": "clarification_required", "intent": "?", "clarification_question": "which?", "steps": UK_REVENUE["steps"]}
        )


def test_rejected_requires_reason_and_no_steps():
    with pytest.raises(PlanParseError, match="rejection_reason"):
        parse_plan({"status": "rejected", "intent": "?"})
    with pytest.raises(PlanParseError, match="steps"):
        parse_plan({"status": "rejected", "intent": "?", "rejection_reason": "no", "steps": UK_REVENUE["steps"]})


def test_blank_reason_text_counts_as_missing():
    with pytest.raises(PlanParseError):
        parse_plan({"status": "rejected", "intent": "?", "rejection_reason": "   "})


# --- malformed input -----------------------------------------------------------


def test_malformed_json_is_a_parse_error():
    with pytest.raises(PlanParseError, match="JSON"):
        parse_plan('{"status": "success", ')


def test_non_object_json_is_a_parse_error():
    with pytest.raises(PlanParseError):
        parse_plan("[1, 2, 3]")
    with pytest.raises(PlanParseError):
        parse_plan('"just a string"')


# --- the schema the planner will be shown -------------------------------------


def test_json_schema_exposes_the_closed_option_sets():
    schema = json.dumps(AnalysisPlan.model_json_schema())
    for name in ("filter_rows", "compute_metric", "aggregate", "group_by", "filter_groups", "select_extreme"):
        assert f'"{name}"' in schema
    for option in ("is_null", "median", "highest", "clarification_required"):
        assert f'"{option}"' in schema
    assert "execute" not in schema


# --- helpers -------------------------------------------------------------------


def _success(steps: list[dict]) -> dict:
    return {"status": "success", "intent": "test", "steps": steps}
