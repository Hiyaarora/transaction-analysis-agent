"""Provider selection.

One function decides which client the whole application uses, so switching
provider is a configuration change rather than a code change. The CLI and the
web API both come through here, which is what keeps them answering from the
same engine.

A second provider can be named as a fallback. Each provider already retries
within itself; the fallback covers what that cannot - the provider as a whole
being unreachable or out of quota for the day.
"""

from app.config import Settings
from app.llm.base import LLMClient, LLMError
from app.llm.chain import FallbackLLMClient
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient

PROVIDERS = ("gemini", "groq")


def build_llm(settings: Settings) -> LLMClient:
    primary = _build(settings.llm_provider, settings)

    fallback_name = (settings.llm_fallback_provider or "").strip().lower()
    if not fallback_name or fallback_name == settings.llm_provider.strip().lower():
        return primary

    try:
        backup = _build(fallback_name, settings)
    except LLMError:
        # A fallback that cannot be built - usually a missing key - must not
        # stop the provider that IS configured from answering questions.
        # Raising here would turn an optional extra into a hard requirement.
        if fallback_name not in PROVIDERS:
            raise
        return primary

    return FallbackLLMClient(primary, backup)


def _build(provider: str, settings: Settings) -> LLMClient:
    name = provider.strip().lower()

    if name == "gemini":
        return GeminiClient(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_fallback_model,
            backup_api_key=settings.gemini_api_key_2,
            thinking_level=settings.gemini_thinking_level,
        )

    if name == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model)

    raise LLMError(f"Unknown LLM provider '{provider}'. Supported: {', '.join(PROVIDERS)}.")
