"""The LLMClient contract, proven on the fake that every planner test uses."""

import pytest

from app.llm.base import LLMCall, LLMClient, LLMError
from app.llm.fake import FakeLLMClient


def test_fake_is_an_llm_client():
    assert isinstance(FakeLLMClient(responses=[]), LLMClient)


def test_returns_scripted_responses_in_order():
    fake = FakeLLMClient(responses=['{"a": 1}', '{"b": 2}'])
    assert fake.complete_json(system="s", user="u", schema={}) == '{"a": 1}'
    assert fake.complete_json(system="s", user="u", schema={}) == '{"b": 2}'


def test_records_exactly_what_was_sent():
    fake = FakeLLMClient(responses=["ok"])
    fake.complete_json(system="SYSTEM", user="QUESTION", schema={"type": "object"})
    assert fake.calls == [LLMCall(system="SYSTEM", user="QUESTION", schema={"type": "object"})]


def test_scripted_exception_is_raised():
    fake = FakeLLMClient(responses=[LLMError("rate limited")])
    with pytest.raises(LLMError, match="rate limited"):
        fake.complete_json(system="s", user="u", schema={})


def test_exhausted_fake_is_a_test_bug():
    fake = FakeLLMClient(responses=[])
    with pytest.raises(RuntimeError, match="no scripted response"):
        fake.complete_json(system="s", user="u", schema={})


def test_fake_reports_name_and_model():
    fake = FakeLLMClient(responses=[])
    assert fake.name == "fake"
    assert fake.model == "scripted"
