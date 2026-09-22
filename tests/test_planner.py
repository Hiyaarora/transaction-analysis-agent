"""Planner: question -> AnalysisPlan through a scripted LLM.

Proves what the LLM is shown (and not shown), the one-retry policy, and that
the planner never interprets a reply itself.
"""

import json

import pandas as pd
import pytest

from app.data_profile import build_profile
from app.llm.base import LLMError
from app.llm.fake import FakeLLMClient
from app.planner import UNUSABLE_PLAN_REASON, Planner, build_context
from app.schemas import AnalysisPlan

GOOD = json.dumps(
    {
        "status": "success",
        "intent": "Count UK transactions",
        "steps": [
            {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
            {"tool": "aggregate", "column": "id", "func": "count"},
        ],
    }
)
BAD_JSON = '{"status": "success", '
BAD_PLAN = json.dumps({"status": "success", "intent": "x", "steps": [{"tool": "execute_python", "code": "1"}]})


@pytest.fixture
def profile():
    df = pd.DataFrame(
        {
            "id": ["TX-SECRET-1", "TX-SECRET-2"],
            "date": pd.to_datetime(["2026-01-03", "2026-03-04"]),
            "region": ["UK", "DE"],
            "product": ["Alpha", "Gamma"],
            "units": [10.0, None],
            "unit_price": [100.0, 500.0],
            "discount": [0.1, 0.2],
            "zone": ["north", "south"],
        }
    )
    return build_profile(df, parse_errors={"units": 1})


def make(profile, *responses):
    fake = FakeLLMClient(responses=list(responses))
    return Planner(fake, profile), fake


# --- happy path -------------------------------------------------------------------


def test_returns_the_parsed_plan(profile):
    planner, fake = make(profile, GOOD)
    plan = planner.plan("How many UK transactions?")
    assert isinstance(plan, AnalysisPlan)
    assert plan.status == "success"
    assert plan.steps[0].value == "UK"
    assert len(fake.calls) == 1


def test_question_is_the_user_turn_and_schema_is_the_plan_schema(profile):
    planner, fake = make(profile, GOOD)
    planner.plan("How many UK transactions?")
    call = fake.calls[0]
    assert "How many UK transactions?" in call.user
    assert call.schema == AnalysisPlan.model_json_schema()


def test_non_success_plans_pass_through(profile):
    clar = json.dumps({"status": "clarification_required", "intent": "?", "clarification_question": "March?"})
    rej = json.dumps({"status": "rejected", "intent": "?", "rejection_reason": "no code"})
    planner, _ = make(profile, clar, rej)
    assert planner.plan("Mars?").clarification_question == "March?"
    assert planner.plan("run python").rejection_reason == "no code"


# --- what the LLM is shown ------------------------------------------------------


def test_context_describes_columns_values_dates_and_missing(profile):
    ctx = build_context(profile)
    for needle in (
        "region", "product", "units", "unit_price", "discount", "date",   # columns
        "DE", "UK", "Alpha", "Gamma",                                     # discovered category values
        "2026-01-03", "2026-03-04",                                        # date range
        "revenue",                                                         # derived metric
        "2 transactions",                                                  # row count
        "units: 1 missing",                                                # null counts
    ):
        assert needle in ctx, needle


def test_context_names_unsupported_extra_columns(profile):
    assert "zone" in build_context(profile)
    assert "not supported" in build_context(profile).lower()


def test_system_prompt_contains_tools_options_and_the_six_rules(profile):
    planner, fake = make(profile, GOOD)
    planner.plan("q")
    system = fake.calls[0].system
    for tool in ("filter_rows", "compute_metric", "aggregate", "group_by", "filter_groups", "select_extreme"):
        assert tool in system
    for option in ("is_null", "between", "median", "highest", "clarification_required", "rejected"):
        assert option in system
    for rule in ("never compute", "synonym", "clarification", "code", "YYYY-MM-DD", "fraction"):
        assert rule.lower() in system.lower(), rule


def test_llm_never_sees_rows_or_paths(profile):
    planner, fake = make(profile, GOOD)
    planner.plan("q")
    sent = fake.calls[0].system + fake.calls[0].user
    assert "TX-SECRET" not in sent
    assert ".csv" not in sent and "data/" not in sent
    assert "north" not in sent  # row values of the extra column


# --- retry policy -------------------------------------------------------------------


def test_malformed_first_reply_is_retried_once_with_the_error(profile):
    planner, fake = make(profile, BAD_JSON, GOOD)
    plan = planner.plan("How many UK transactions?")
    assert plan.status == "success"
    assert len(fake.calls) == 2
    retry = fake.calls[1].user
    assert "How many UK transactions?" in retry
    assert "not valid JSON" in retry
    assert fake.calls[1].system == fake.calls[0].system


def test_structurally_invalid_first_reply_is_retried_with_the_error(profile):
    planner, fake = make(profile, BAD_PLAN, GOOD)
    planner.plan("q")
    assert "execute_python" in fake.calls[1].user


def test_two_failures_give_the_exact_rejection(profile):
    planner, fake = make(profile, BAD_JSON, BAD_PLAN)
    plan = planner.plan("q")
    assert plan.status == "rejected"
    assert plan.rejection_reason == UNUSABLE_PLAN_REASON == "planner produced an unusable plan"
    assert len(fake.calls) == 2


def test_never_a_third_attempt(profile):
    planner, fake = make(profile, BAD_JSON, BAD_JSON, GOOD)
    plan = planner.plan("q")
    assert plan.status == "rejected"
    assert len(fake.calls) == 2


def test_provider_error_propagates(profile):
    planner, _ = make(profile, LLMError("quota"))
    with pytest.raises(LLMError, match="quota"):
        planner.plan("q")


def test_prompt_is_ascii_safe_for_windows_consoles(profile):
    planner, fake = make(profile, GOOD)
    planner.plan("q")
    (fake.calls[0].system + fake.calls[0].user).encode("ascii")  # raises if not


def test_rules_cover_year_inference_unsupported_metrics_and_date_ranges(profile):
    planner, fake = make(profile, GOOD)
    planner.plan("q")
    system = " ".join(fake.calls[0].system.lower().split())  # collapse line wraps
    assert "more than one year" in system          # ask for the year only when it is genuinely ambiguous
    assert "outside" in system and "range" in system  # never ask just because dates fall outside the data
    assert "not a synonym" in system                 # unsupported metric -> rejected, not "did you mean revenue"
