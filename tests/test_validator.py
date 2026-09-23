"""Semantic validation (Level 2): does a well-formed plan make sense for THIS dataset?

Every plan here is structurally valid — Pydantic has already accepted it. The
validator's job is to check it against the profile of the loaded data and the
contract, and to say precisely why when it does not fit.
"""

import pandas as pd
import pytest

from app.data_profile import build_profile
from app.schemas import parse_plan
from app.validator import ValidationOutcome, validate


@pytest.fixture
def profile():
    df = pd.DataFrame(
        {
            "id": ["T1", "T2", "T3"],
            "date": pd.to_datetime(["2026-01-03", "2026-02-10", "2026-03-04"]),
            "region": ["UK", "DE", "FR"],
            "product": ["Alpha", "Beta", "Gamma"],
            "units": [10.0, 5.0, 2.0],
            "unit_price": [100.0, 200.0, 500.0],
            "discount": [0.1, 0.0, 0.2],
            "zone": ["north", "south", "east"],  # extra column: present, not supported
        }
    )
    return build_profile(df)


def success(*steps: dict) -> dict:
    return {"status": "success", "intent": "test", "steps": list(steps)}


def filt(column, op, value=None):
    return {"tool": "filter_rows", "column": column, "op": op, "value": value}


COMPUTE = {"tool": "compute_metric", "metric": "revenue"}
SUM_REV = {"tool": "aggregate", "column": "revenue", "func": "sum"}
COUNT = {"tool": "aggregate", "column": "id", "func": "count"}


def outcome(profile, *steps) -> ValidationOutcome:
    return validate(parse_plan(success(*steps)), profile)


# --- pass-through of the LLM's own decisions ------------------------------------


def test_llm_clarification_passes_through(profile):
    plan = parse_plan({"status": "clarification_required", "intent": "?", "clarification_question": "Did you mean March?"})
    out = validate(plan, profile)
    assert out.status == "clarification_required"
    assert out.clarification_question == "Did you mean March?"
    assert out.validated_plan is None


def test_llm_rejection_passes_through(profile):
    plan = parse_plan({"status": "rejected", "intent": "?", "rejection_reason": "Cannot run code."})
    out = validate(plan, profile)
    assert out.status == "rejected"
    assert out.rejection_reason == "Cannot run code."


# --- valid plans ------------------------------------------------------------------


def test_valid_plan_is_returned_with_original_preserved(profile):
    plan = parse_plan(success(filt("region", "eq", "UK"), COMPUTE, SUM_REV))
    out = validate(plan, profile)
    assert out.status == "valid"
    assert out.plan is plan
    assert out.validated_plan == plan
    assert out.rejection_reason is None and out.clarification_question is None


@pytest.mark.parametrize(
    "steps",
    [
        (COUNT,),
        ({"tool": "aggregate", "column": "units", "func": "mean"},),
        (filt("units", "gt", 5), COUNT),
        (filt("discount", "is_null"), COUNT),
        (filt("discount", "not_null"), {"tool": "aggregate", "column": "discount", "func": "median"}),
        (filt("product", "in", ["Alpha", "Beta"]), COUNT),
        (filt("date", "between", ["2026-01-01", "2026-02-15"]), COMPUTE, SUM_REV),
        (filt("date", "gte", "2026-02-01"), COUNT),
        (filt("units", "between", [2, 8]), COUNT),
        (filt("id", "eq", "T2"), COUNT),
        ({"tool": "group_by", "by": "region", "func": "count"},),
        (COMPUTE, {"tool": "group_by", "by": "product", "func": "median", "column": "revenue"}),
        (COMPUTE, {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
         {"tool": "filter_groups", "op": "gt", "value": 500}),
        (COMPUTE, {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
         {"tool": "select_extreme", "mode": "highest"}),
        (COMPUTE, {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
         {"tool": "filter_groups", "op": "between", "value": [100, 5000]}, {"tool": "select_extreme", "mode": "lowest"}),
        (COMPUTE, filt("revenue", "gt", 900), COUNT),  # derived column usable after compute
    ],
)
def test_valid_shapes(profile, steps):
    assert outcome(profile, *steps).status == "valid"


def test_date_outside_dataset_range_is_still_valid(profile):
    out = outcome(profile, filt("date", "between", ["2020-01-01", "2020-12-31"]), COUNT)
    assert out.status == "valid"


# --- categorical value resolution -------------------------------------------------


def test_case_insensitive_match_is_normalised(profile):
    out = outcome(profile, filt("region", "eq", "uk"), COUNT)
    assert out.status == "valid"
    assert out.validated_plan.steps[0].value == "UK"
    assert out.plan.steps[0].value == "uk"  # original untouched


def test_case_insensitive_match_in_list(profile):
    out = outcome(profile, filt("product", "in", ["alpha", "GAMMA"]), COUNT)
    assert out.status == "valid"
    assert out.validated_plan.steps[0].value == ["Alpha", "Gamma"]


def test_near_miss_asks_for_clarification_with_suggestion(profile):
    out = outcome(profile, filt("product", "eq", "Gama"), COUNT)
    assert out.status == "clarification_required"
    assert "Gama" in out.clarification_question
    assert "Gamma" in out.clarification_question
    assert out.validated_plan is None


def test_no_evidence_is_rejected_and_lists_valid_values(profile):
    out = outcome(profile, filt("region", "eq", "Britain"), COUNT)
    assert out.status == "rejected"
    assert "Britain" in out.rejection_reason
    for v in ("DE", "FR", "UK"):
        assert v in out.rejection_reason


def test_synonym_is_never_mapped(profile):
    out = outcome(profile, filt("region", "eq", "Germany"), COUNT)
    assert out.status != "valid"


def test_unknown_value_in_list_is_reported(profile):
    out = outcome(profile, filt("region", "in", ["UK", "Mars"]), COUNT)
    assert out.status == "rejected"
    assert "Mars" in out.rejection_reason


# --- fabricated / unsupported columns and metrics -------------------------------


def test_fabricated_column_is_rejected(profile):
    out = outcome(profile, filt("profit", "gt", 0), COUNT)
    assert out.status == "rejected"
    assert "profit" in out.rejection_reason


def test_fabricated_metric_is_rejected(profile):
    out = outcome(profile, {"tool": "compute_metric", "metric": "profit"}, COUNT)
    assert out.status == "rejected"
    assert "profit" in out.rejection_reason


def test_extra_dataset_column_is_not_addressable(profile):
    out = outcome(profile, filt("zone", "eq", "north"), COUNT)
    assert out.status == "rejected"
    assert "zone" in out.rejection_reason


def test_derived_column_before_compute_is_rejected(profile):
    out = outcome(profile, SUM_REV)
    assert out.status == "rejected"
    assert "revenue" in out.rejection_reason and "compute" in out.rejection_reason


# --- operator / value / type agreement -----------------------------------------


def test_null_checks_take_no_value(profile):
    out = outcome(profile, filt("discount", "is_null", 0), COUNT)
    assert out.status == "rejected"


def test_between_needs_two_ordered_values(profile):
    assert outcome(profile, filt("units", "between", [5]), COUNT).status == "rejected"
    out = outcome(profile, filt("units", "between", [8, 2]), COUNT)
    assert out.status == "rejected"
    assert "8" in out.rejection_reason and "2" in out.rejection_reason


def test_date_between_must_be_ordered(profile):
    out = outcome(profile, filt("date", "between", ["2026-03-01", "2026-01-01"]), COUNT)
    assert out.status == "rejected"


def test_in_needs_non_empty_list(profile):
    assert outcome(profile, filt("region", "in", []), COUNT).status == "rejected"
    assert outcome(profile, filt("region", "in", "UK"), COUNT).status == "rejected"


def test_numeric_column_rejects_text_value(profile):
    out = outcome(profile, filt("units", "gt", "many"), COUNT)
    assert out.status == "rejected"
    assert "units" in out.rejection_reason


def test_date_column_rejects_non_iso_value(profile):
    out = outcome(profile, filt("date", "gte", "Mars"), COUNT)
    assert out.status == "rejected"
    assert "Mars" in out.rejection_reason


def test_ordering_operator_on_categorical_is_rejected(profile):
    out = outcome(profile, filt("region", "gt", "A"), COUNT)
    assert out.status == "rejected"


def test_numeric_aggregation_on_text_column_is_rejected(profile):
    out = outcome(profile, {"tool": "aggregate", "column": "region", "func": "sum"})
    assert out.status == "rejected"
    assert "region" in out.rejection_reason


def test_group_by_key_must_be_categorical(profile):
    out = outcome(profile, {"tool": "group_by", "by": "units", "func": "count"})
    assert out.status == "rejected"
    assert "units" in out.rejection_reason


def test_group_by_non_count_needs_numeric_column(profile):
    assert outcome(profile, {"tool": "group_by", "by": "region", "func": "sum"}).status == "rejected"
    assert outcome(profile, {"tool": "group_by", "by": "region", "func": "sum", "column": "product"}).status == "rejected"


def test_filter_groups_between_must_be_ordered(profile):
    out = outcome(
        profile,
        {"tool": "group_by", "by": "region", "func": "count"},
        {"tool": "filter_groups", "op": "between", "value": [10, 1]},
    )
    assert out.status == "rejected"


# --- step ordering ----------------------------------------------------------------


def test_filter_groups_without_group_by_is_rejected(profile):
    out = outcome(profile, {"tool": "filter_groups", "op": "gt", "value": 1})
    assert out.status == "rejected"
    assert "group_by" in out.rejection_reason


def test_select_extreme_without_group_by_is_rejected(profile):
    out = outcome(profile, COUNT, {"tool": "select_extreme", "mode": "highest"})
    assert out.status == "rejected"


def test_row_step_after_aggregate_is_rejected(profile):
    out = outcome(profile, COUNT, filt("region", "eq", "UK"))
    assert out.status == "rejected"


def test_two_terminal_steps_are_rejected(profile):
    out = outcome(profile, COUNT, COUNT)
    assert out.status == "rejected"


def test_plan_that_produces_no_result_is_rejected(profile):
    out = outcome(profile, filt("region", "eq", "UK"))
    assert out.status == "rejected"
    assert "result" in out.rejection_reason


def test_first_problem_is_reported(profile):
    out = outcome(profile, filt("profit", "gt", 0), filt("region", "eq", "Mars"), COUNT)
    assert "profit" in out.rejection_reason
    assert "Mars" not in out.rejection_reason


# --- a category value must come from the question ------------------------------------
#
# The planner is told never to map a synonym onto a category value, but a rule
# in a prompt is a tendency, not a guarantee: gpt-oss-120b answered "revenue
# for products from Germany" by filtering region = DE, three times out of three.
# The semantic checks above cannot see that, because DE is a perfectly valid
# region - the substitution happened before the plan was built. So the plan is
# also checked against the words the user actually used.


def ask(profile, question, *steps):
    return validate(parse_plan(success(*steps)), profile, question=question)


def test_a_value_the_user_named_is_accepted(profile):
    assert ask(profile, "What is the total revenue for UK transactions?",
               filt("region", "eq", "UK"), COUNT).status == "valid"


def test_the_match_ignores_case(profile):
    assert ask(profile, "how much revenue came from uk?", filt("region", "eq", "UK"), COUNT).status == "valid"


def test_a_value_the_user_never_named_asks_for_clarification(profile):
    outcome = ask(profile, "Can you tell me the total revenue for products from Germany?",
                  filt("region", "eq", "DE"), COUNT)
    assert outcome.status == "clarification_required"
    assert "Germany" not in outcome.clarification_question  # we do not guess what they meant
    assert "DE" in outcome.clarification_question and "FR" in outcome.clarification_question


def test_every_value_in_a_list_must_be_named(profile):
    assert ask(profile, "revenue for UK and FR", filt("region", "in", ["UK", "FR"]), COUNT).status == "valid"
    assert ask(profile, "revenue for UK and Germany",
               filt("region", "in", ["UK", "DE"]), COUNT).status == "clarification_required"


def test_a_code_inside_another_word_does_not_count_as_naming_it(profile):
    # "DE" appears inside "Denmark"; the user did not name the region DE.
    outcome = ask(profile, "What is the total revenue for Denmark?", filt("region", "eq", "DE"), COUNT)
    assert outcome.status == "clarification_required"


def test_punctuation_around_a_value_is_fine(profile):
    assert ask(profile, "revenue for 'UK'?", filt("region", "eq", "UK"), COUNT).status == "valid"
    assert ask(profile, "What are Alpha's total units?", filt("product", "eq", "Alpha"), COUNT).status == "valid"


def test_numeric_and_date_filters_are_not_subject_to_this_check(profile):
    # "20%" becomes 0.2 and "January" becomes a date: neither appears literally.
    assert ask(profile, "how many transactions used a 20% discount?",
               filt("discount", "eq", 0.2), COUNT).status == "valid"
    assert ask(profile, "revenue between January 1 and February 15",
               filt("date", "between", ["2026-01-01", "2026-02-15"]), COMPUTE, SUM_REV).status == "valid"


def test_without_the_question_the_check_is_skipped(profile):
    # validate() is usable without a question; the agent always supplies one.
    assert validate(parse_plan(success(filt("region", "eq", "DE"), COUNT)), profile).status == "valid"
