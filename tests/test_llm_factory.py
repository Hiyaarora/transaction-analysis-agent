"""Provider selection: one place decides which LLM the application uses."""

import pytest

from app.config import Settings
from app.llm.base import LLMError
from app.llm.factory import build_llm
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient


def settings(**overrides) -> Settings:
    base = dict(
        llm_provider="gemini",
        llm_fallback_provider=None,
        gemini_api_key="g-key",
        gemini_api_key_2=None,
        gemini_model="gemini-model",
        gemini_fallback_model=None,
        gemini_thinking_level="MINIMAL",
        groq_api_key="q-key",
        groq_model="groq-model",
    )
    base.update(overrides)
    return Settings(**base)


def test_gemini_is_the_default_provider():
    client = build_llm(settings())
    assert isinstance(client, GeminiClient)
    assert client.name == "gemini"


def test_groq_is_selected_by_configuration():
    client = build_llm(settings(llm_provider="groq"))
    assert isinstance(client, GroqClient)
    assert (client.name, client.model) == ("groq", "groq-model")


def test_provider_choice_is_case_insensitive():
    assert isinstance(build_llm(settings(llm_provider="GROQ")), GroqClient)


def test_an_unknown_provider_says_what_is_supported():
    with pytest.raises(LLMError) as raised:
        build_llm(settings(llm_provider="wishful-thinking"))
    assert "wishful-thinking" in str(raised.value)
    assert "gemini" in str(raised.value) and "groq" in str(raised.value)


def test_a_missing_key_for_the_chosen_provider_is_reported():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        build_llm(settings(llm_provider="groq", groq_api_key=None))


def test_choosing_groq_does_not_require_a_gemini_key():
    assert isinstance(build_llm(settings(llm_provider="groq", gemini_api_key=None)), GroqClient)


# --- cross-provider fallback ------------------------------------------------------


def test_no_fallback_provider_means_a_single_client():
    from app.llm.chain import FallbackLLMClient

    client = build_llm(settings(llm_provider="groq", llm_fallback_provider=None))
    assert not isinstance(client, FallbackLLMClient)


def test_a_fallback_provider_wraps_both():
    from app.llm.chain import FallbackLLMClient

    client = build_llm(settings(llm_provider="groq", llm_fallback_provider="gemini"))
    assert isinstance(client, FallbackLLMClient)
    assert client.name == "groq+gemini"
    assert client.model == "groq-model"  # the one a question is planned with


def test_a_fallback_equal_to_the_primary_is_ignored():
    from app.llm.chain import FallbackLLMClient

    client = build_llm(settings(llm_provider="groq", llm_fallback_provider="groq"))
    assert not isinstance(client, FallbackLLMClient)


def test_an_unusable_fallback_does_not_stop_the_primary_from_working():
    # No Gemini key: the fallback cannot be built, but that must not prevent
    # questions being answered by the provider that is configured.
    from app.llm.groq import GroqClient

    client = build_llm(settings(llm_provider="groq", llm_fallback_provider="gemini", gemini_api_key=None))
    assert isinstance(client, GroqClient)


def test_an_unknown_fallback_provider_is_reported():
    with pytest.raises(LLMError, match="wishful"):
        build_llm(settings(llm_provider="groq", llm_fallback_provider="wishful"))
