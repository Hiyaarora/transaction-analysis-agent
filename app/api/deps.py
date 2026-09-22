"""Shared dependencies. Overridable in tests via `app.dependency_overrides`."""

from functools import lru_cache

from app.api.sessions import SessionStore
from app.config import load_settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient


@lru_cache(maxsize=1)
def get_store() -> SessionStore:
    return SessionStore()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """One client for the whole process: it is stateless per call and owns a pool."""
    settings = load_settings()
    return GeminiClient(
        settings.gemini_api_key,
        settings.gemini_model,
        settings.gemini_fallback_model,
        backup_api_key=settings.gemini_api_key_2,
    )


def llm_is_configured() -> bool:
    return bool(load_settings().gemini_api_key)
