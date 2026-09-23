"""GroqClient with a stubbed SDK: what we send, and how failures are translated.

No network. The provider exists because Gemini's free tier has been measured
at 16-93 seconds per plan; whether Groq is actually faster is a question for a
benchmark, not for these tests.
"""

import groq
import httpx
import pytest

from app.llm.base import LLMClient, LLMError
from app.llm.groq import GroqClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


class _StubCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = type("M", (), {"content": outcome})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


class _StubSDK:
    def __init__(self, outcomes):
        self.completions = _StubCompletions(outcomes)
        self.chat = type("Chat", (), {"completions": self.completions})


def make(outcomes, **kwargs):
    sdk = _StubSDK(outcomes)
    client = GroqClient(api_key="k", model="test-model", sdk_factory=lambda key: sdk, **kwargs)
    return client, sdk.completions


def _status_error(status: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": f"http {status}"}})
    return groq.APIStatusError(f"http {status}", response=response, body=None)


# --- request shape ------------------------------------------------------------------


def test_is_an_llm_client():
    client, _ = make(["{}"])
    assert isinstance(client, LLMClient)
    assert (client.name, client.model) == ("groq", "test-model")


def test_sends_the_prompt_as_two_messages_and_returns_the_content():
    client, completions = make(['{"a": 1}'])
    assert client.complete_json(system="SYS", user="Q?", schema=SCHEMA) == '{"a": 1}'

    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q?"},
    ]
    assert call["temperature"] == 0  # planning should be repeatable


def test_asks_for_json_matching_the_plan_schema():
    client, completions = make(["{}"])
    client.complete_json(system="s", user="u", schema=SCHEMA)

    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == SCHEMA
    assert response_format["json_schema"]["name"]


def test_missing_api_key_is_an_llm_error():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient(api_key="", model="m", sdk_factory=lambda key: _StubSDK([]))


# --- error translation ----------------------------------------------------------------


def test_an_empty_reply_is_an_llm_error():
    client, _ = make([None])
    with pytest.raises(LLMError, match="empty"):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_an_api_error_is_translated_with_its_status():
    client, _ = make([_status_error(400)])
    with pytest.raises(LLMError, match="400"):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_no_vendor_exception_type_reaches_the_caller():
    for failure in (httpx.ConnectError("refused"), RuntimeError("sdk exploded")):
        client, _ = make([failure])
        with pytest.raises(LLMError):
            client.complete_json(system="s", user="u", schema=SCHEMA)


def test_a_rate_limit_is_retried_once_then_surfaced():
    client, completions = make([_status_error(429), '{"a": 2}'])
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 2}'
    assert len(completions.calls) == 2


def test_the_retry_chain_is_bounded():
    from app.llm.groq import MAX_ATTEMPTS

    client, completions = make([_status_error(429)] * 5)
    with pytest.raises(LLMError, match="429"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(completions.calls) == MAX_ATTEMPTS
    assert MAX_ATTEMPTS <= 2


def test_a_bad_request_is_not_retried():
    client, completions = make([_status_error(400), '{"a": 2}'])
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(completions.calls) == 1


def test_error_messages_never_carry_key_material():
    secret = "gsk_super_secret_value"
    sdk = _StubSDK([_status_error(400)])
    client = GroqClient(api_key=secret, model="m", sdk_factory=lambda key: sdk)
    with pytest.raises(LLMError) as raised:
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert secret not in str(raised.value)
