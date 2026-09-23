"""Shared dependencies. Overridable in tests via `app.dependency_overrides`."""

from functools import lru_cache

from app.api.sessions import SessionStore
from app.config import load_settings
from app.llm.base import LLMClient
from app.llm.factory import build_llm


@lru_cache(maxsize=1)
def get_store() -> SessionStore:
    return SessionStore()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """One client for the whole process: it is stateless per call and owns a pool."""
    settings = load_settings()
    return build_llm(settings)


def llm_is_configured() -> bool:
    """Whether the *selected* provider has a key, not whichever one is default."""
    settings = load_settings()
    keys = {"gemini": settings.gemini_api_key, "groq": settings.groq_api_key}
    return bool(keys.get(settings.llm_provider.strip().lower()))
