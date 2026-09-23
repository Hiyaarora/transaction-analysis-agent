"""Cross-provider fallback: if the primary provider cannot answer, ask the other.

This covers a wider failure than any single provider's retry chain - the whole
service being unreachable, not just one model or one key.
"""

import pytest

from app.llm.base import LLMClient, LLMError
from app.llm.chain import FallbackLLMClient


class _Recorder(LLMClient):
    def __init__(self, name, outcome):
        self.name = name
        self.model = f"{name}-model"
        self._outcome = outcome
        self.calls = 0

    def complete_json(self, *, system, user, schema):
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def chain(primary_outcome, backup_outcome):
    primary = _Recorder("groq", primary_outcome)
    backup = _Recorder("gemini", backup_outcome)
    return FallbackLLMClient(primary, backup), primary, backup


def ask(client):
    return client.complete_json(system="s", user="u", schema={})


def test_it_is_an_llm_client_named_after_both():
    client, _, _ = chain('{"a": 1}', '{"a": 2}')
    assert isinstance(client, LLMClient)
    assert "groq" in client.name and "gemini" in client.name


def test_the_backup_is_untouched_while_the_primary_works():
    client, primary, backup = chain('{"a": 1}', '{"a": 2}')
    assert ask(client) == '{"a": 1}'
    assert (primary.calls, backup.calls) == (1, 0)


def test_the_backup_answers_when_the_primary_fails():
    client, primary, backup = chain(LLMError("groq is down"), '{"a": 2}')
    assert ask(client) == '{"a": 2}'
    assert (primary.calls, backup.calls) == (1, 1)


def test_the_model_reported_is_whichever_one_would_be_asked_first():
    client, primary, _ = chain('{"a": 1}', '{"a": 2}')
    assert client.model == primary.model


def test_both_failing_raises_one_error_naming_both():
    client, primary, backup = chain(LLMError("groq exhausted"), LLMError("gemini exhausted"))
    with pytest.raises(LLMError) as raised:
        ask(client)
    message = str(raised.value)
    assert "groq exhausted" in message
    assert "gemini exhausted" in message
    assert (primary.calls, backup.calls) == (1, 1)


def test_the_chain_never_loops():
    # Exactly two providers are tried, once each - no retry on top of a retry.
    client, primary, backup = chain(LLMError("a"), LLMError("b"))
    with pytest.raises(LLMError):
        ask(client)
    assert primary.calls == 1 and backup.calls == 1
