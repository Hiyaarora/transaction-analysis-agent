"""Opt-in live check against Gemini. Skipped unless RUN_LIVE_LLM=1.

Proves two things the stubbed tests cannot: the API accepts our transformed
plan schema, and a real reply parses as an AnalysisPlan. Costs one request.
"""

import os

import pytest

from app.config import load_settings
from app.llm.gemini import GeminiClient
from app.schemas import AnalysisPlan, parse_plan

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call Gemini")


def test_gemini_accepts_plan_schema_and_returns_a_parseable_plan():
    settings = load_settings()
    client = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.gemini_fallback_model)
    text = client.complete_json(
        system=(
            "You translate questions into an analysis plan as JSON matching the schema. "
            "Columns: region (categorical: DE, FR, UK), units (numeric). Never compute answers."
        ),
        user="How many transactions are there for UK?",
        schema=AnalysisPlan.model_json_schema(),
    )
    plan = parse_plan(text)
    assert plan.status == "success"
    assert plan.steps[-1].tool == "aggregate"
