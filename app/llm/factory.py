"""Provider selection.

One function decides which client the whole application uses, so switching
provider is a configuration change rather than a code change. The CLI and the
web API both come through here, which is what keeps them answering from the
same engine.
"""

from app.config import Settings
from app.llm.base import LLMClient, LLMError
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient

PROVIDERS = ("gemini", "groq")


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.strip().lower()

    if provider == "gemini":
        return GeminiClient(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_fallback_model,
            backup_api_key=settings.gemini_api_key_2,
            thinking_level=settings.gemini_thinking_level,
        )

    if provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model)

    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Supported: {', '.join(PROVIDERS)}.")
