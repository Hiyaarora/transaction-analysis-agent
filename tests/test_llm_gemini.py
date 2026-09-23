"""GeminiClient with a stubbed SDK: what we send, what we translate, when we fall back.

No network. The live check lives in test_llm_live.py and is opt-in.
"""

import pytest
from google.genai import errors as genai_errors

from app.llm.base import LLMClient, LLMError
from app.llm.gemini import GeminiClient, _gemini_schema


class _Response:
    def __init__(self, text):
        self.text = text


class _StubModels:
    """Stands in for `client.models`. Scripted outcomes per call, records every call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _StubSDK:
    def __init__(self, outcomes, key="k"):
        self.models = _StubModels(outcomes)
        self.key = key


def _api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": f"http {code}", "status": "x"}})


def make(outcomes, fallback="gemini-2.5-flash-lite", backup_key=None, backup_outcomes=None):
    """Build a client over one or two stubbed SDKs, and return the call log per key."""
    sdks = {"k": _StubSDK(outcomes, key="k")}
    if backup_key:
        sdks[backup_key] = _StubSDK(backup_outcomes or [], key=backup_key)
    client = GeminiClient(
        api_key="k",
        model="gemini-2.5-flash",
        fallback_model=fallback,
        backup_api_key=backup_key,
        sdk_factory=lambda key: sdks[key],
    )
    return client, sdks["k"].models


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


# --- request shape ----------------------------------------------------------------


def test_is_an_llm_client():
    client, _ = make(["{}"])
    assert isinstance(client, LLMClient)
    assert (client.name, client.model) == ("gemini", "gemini-2.5-flash")


def test_sends_system_user_and_schema_and_returns_text():
    client, models = make(['{"a": 1}'])
    assert client.complete_json(system="SYS", user="Q?", schema=SCHEMA) == '{"a": 1}'
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "Q?"
    cfg = call["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == SCHEMA
    assert cfg.temperature == 0
    assert cfg.thinking_config.thinking_level == "MINIMAL"


def test_missing_api_key_is_an_llm_error():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="", model="m", fallback_model=None, sdk_factory=lambda key: _StubSDK([]))


# --- error translation ------------------------------------------------------------


def test_empty_reply_is_an_llm_error():
    client, _ = make([None])
    with pytest.raises(LLMError, match="empty"):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_api_error_is_translated_with_status_code():
    client, models = make([_api_error(400)])
    with pytest.raises(LLMError, match="400"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(models.calls) == 1  # no fallback for a non-rate-limit error


# --- fallback model ---------------------------------------------------------------


def test_rate_limit_falls_back_once_to_the_fallback_model():
    client, models = make([_api_error(429), '{"a": 2}'])
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 2}'
    assert [c["model"] for c in models.calls] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_rate_limit_on_both_models_is_an_llm_error():
    client, models = make([_api_error(429), _api_error(429)])
    with pytest.raises(LLMError, match="429"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(models.calls) == 2


def test_no_fallback_configured_means_no_second_call():
    client, models = make([_api_error(429)], fallback=None)
    with pytest.raises(LLMError, match="429"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(models.calls) == 1


# --- schema transform -------------------------------------------------------------


def test_schema_transform_rewrites_const_and_drops_discriminator():
    schema = {
        "$defs": {"Step": {"properties": {"tool": {"const": "filter_rows", "type": "string"}}}},
        "properties": {
            "steps": {"items": {"anyOf": [{"$ref": "#/$defs/Step"}], "discriminator": {"propertyName": "tool"}}}
        },
    }
    out = _gemini_schema(schema)
    assert out["$defs"]["Step"]["properties"]["tool"] == {"enum": ["filter_rows"], "type": "string"}
    assert "discriminator" not in out["properties"]["steps"]["items"]
    assert out["properties"]["steps"]["items"]["anyOf"] == [{"$ref": "#/$defs/Step"}]
    assert "const" in str(schema)  # input untouched


def test_schema_transform_handles_the_real_plan_schema():
    from app.schemas import AnalysisPlan

    out = _gemini_schema(AnalysisPlan.model_json_schema())
    text = str(out)
    assert "const" not in text and "discriminator" not in text
    assert "filter_rows" in text and "select_extreme" in text


def test_automatic_function_calling_is_disabled():
    # The SDK logs a warning about AFC on every generate_content call unless it is
    # explicitly disabled; we use no tools, and CLI output must stay clean.
    client, models = make(["{}"])
    client.complete_json(system="s", user="u", schema=SCHEMA)
    assert models.calls[0]["config"].automatic_function_calling.disable is True


# --- second API key ------------------------------------------------------------------


def _client_with_two_keys(primary_outcomes, backup_outcomes):
    sdks = {"key1": _StubSDK(primary_outcomes, key="key1"), "key2": _StubSDK(backup_outcomes, key="key2")}
    client = GeminiClient(
        api_key="key1",
        model="gemini-2.5-flash",
        fallback_model="gemini-2.5-flash-lite",
        backup_api_key="key2",
        sdk_factory=lambda key: sdks[key],
    )
    return client, sdks


def test_second_key_is_used_only_after_the_first_key_is_rate_limited():
    client, sdks = _client_with_two_keys([_api_error(429), _api_error(429)], ['{"a": 3}'])
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 3}'
    # First key: both models tried. Second key: primary model, which succeeded.
    assert [c["model"] for c in sdks["key1"].models.calls] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert [c["model"] for c in sdks["key2"].models.calls] == ["gemini-2.5-flash"]


def test_second_key_is_not_touched_when_the_first_succeeds():
    client, sdks = _client_with_two_keys(['{"a": 1}'], ['{"a": 2}'])
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 1}'
    assert sdks["key2"].models.calls == []


def test_second_key_is_not_tried_for_a_non_rate_limit_error():
    client, sdks = _client_with_two_keys([_api_error(400)], ['{"a": 2}'])
    with pytest.raises(LLMError, match="400"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert sdks["key2"].models.calls == []


def test_exhausting_every_key_and_model_raises_the_last_rate_limit():
    client, sdks = _client_with_two_keys([_api_error(429)] * 2, [_api_error(429)] * 2)
    with pytest.raises(LLMError, match="429"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(sdks["key1"].models.calls) == 2
    assert len(sdks["key2"].models.calls) == 2


def test_a_blank_backup_key_is_ignored():
    client, _ = make(["{}"], backup_key=None)
    assert client.backup_configured is False


def test_high_demand_is_retried_like_a_rate_limit():
    # 503 "model is currently experiencing high demand" is transient; observed
    # on several Gemini flash models. Treat it as retryable, not as a failure.
    # This helper configures a fallback model, so the first key gets two attempts.
    client, sdks = _client_with_two_keys([_api_error(503), _api_error(503)], ['{"a": 9}'])
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 9}'
    assert len(sdks["key1"].models.calls) == 2
    assert len(sdks["key2"].models.calls) == 1


def test_not_found_is_not_retried():
    # 404 means the model is unavailable to this account; retrying wastes quota.
    client, sdks = _client_with_two_keys([_api_error(404)], ['{"a": 9}'])
    with pytest.raises(LLMError, match="404"):
        client.complete_json(system="s", user="u", schema=SCHEMA)
    assert sdks["key2"].models.calls == []


# --- transport failures ----------------------------------------------------------------


def test_a_transport_failure_is_translated_not_leaked():
    # A dropped connection is not an APIError; without translation it would
    # escape the provider abstraction and crash the caller.
    import httpx

    # A dropped connection is retryable, so both attempts on this key see it.
    dropped = lambda: httpx.RemoteProtocolError("Server disconnected without sending a response.")  # noqa: E731
    client, _ = make([dropped(), dropped()])
    with pytest.raises(LLMError, match="disconnected"):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_a_transport_failure_is_retried_on_the_next_key():
    import httpx

    client, sdks = _client_with_two_keys(
        [httpx.ConnectError("connection refused"), httpx.ConnectError("connection refused")],
        ['{"a": 5}'],
    )
    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 5}'
    assert len(sdks["key2"].models.calls) == 1


def test_no_vendor_exception_type_reaches_the_caller():
    # Whatever the SDK throws, callers only ever have to know about LLMError.
    for failure in (RuntimeError("sdk exploded"), OSError("socket gone")):
        client, _ = make([failure])
        with pytest.raises(LLMError):
            client.complete_json(system="s", user="u", schema=SCHEMA)


# --- thinking configuration --------------------------------------------------------


def test_request_uses_thinking_level_not_the_2x_budget_parameter():
    # `thinking_budget` is the Gemini 2.x parameter. On 3.x models it is either
    # rejected (400 on some flash-lite models) or silently ignored, in which
    # case the model reasons at length and a plan takes ~50s instead of ~10s.
    client, models = make(["{}"])
    client.complete_json(system="s", user="u", schema=SCHEMA)
    thinking = models.calls[0]["config"].thinking_config
    assert thinking.thinking_level == "MINIMAL"
    assert thinking.thinking_budget is None


def test_thinking_level_is_configurable():
    sdk = _StubSDK(["{}"])
    client = GeminiClient(
        api_key="k", model="m", thinking_level="LOW", sdk_factory=lambda key: sdk
    )
    client.complete_json(system="s", user="u", schema=SCHEMA)
    assert sdk.models.calls[0]["config"].thinking_config.thinking_level == "LOW"


def test_a_busy_model_is_escaped_by_trying_a_different_model_first():
    # 503 means the model is saturated for everyone, so retrying it on another
    # key repeats the failure. The chain must reach a different model before it
    # reaches a different key.
    sdks = {"key1": _StubSDK([_api_error(503), '{"a": 7}'], key="key1"),
            "key2": _StubSDK(['{"a": 9}'], key="key2")}
    client = GeminiClient(api_key="key1", model="primary", fallback_model="secondary",
                          backup_api_key="key2", sdk_factory=lambda key: sdks[key])

    assert client.complete_json(system="s", user="u", schema=SCHEMA) == '{"a": 7}'
    assert [c["model"] for c in sdks["key1"].models.calls] == ["primary", "secondary"]
    assert sdks["key2"].models.calls == []  # the second key was never needed


def test_the_attempt_chain_covers_both_models_on_both_keys():
    sdks = {"key1": _StubSDK([], key="key1"), "key2": _StubSDK([], key="key2")}
    client = GeminiClient(api_key="key1", model="primary", fallback_model="secondary",
                          backup_api_key="key2", sdk_factory=lambda key: sdks[key])
    assert [(sdk.key, model) for sdk, model in client._attempts()] == [
        ("key1", "primary"), ("key1", "secondary"),
        ("key2", "primary"), ("key2", "secondary"),
    ]


def test_a_fallback_model_is_configured_by_default():
    # Without one, a 503 on the primary model has nowhere to go: every attempt
    # in the chain would use the model that is already overloaded.
    from app.config import load_settings

    assert load_settings(dotenv_path=None).gemini_fallback_model
