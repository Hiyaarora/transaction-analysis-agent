"""Runtime settings, read from the environment.

Only the LLM needs configuration. The dataset path is a CLI argument, not a
setting, so that the planner can never learn it from here either.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str | None
    #: Optional second key, tried when the first is rate limited.
    gemini_api_key_2: str | None
    gemini_model: str
    gemini_fallback_model: str | None
    #: MINIMAL, LOW, MEDIUM or HIGH. Some models accept only a subset.
    gemini_thinking_level: str


def load_settings(dotenv_path: str | Path | None = ".env") -> Settings:
    """Read `.env` (if given and present) into the process environment, then build Settings."""
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_api_key_2=os.getenv("GEMINI_API_KEY_2") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-3.6-flash",
        # Empty string opts out of the fallback; unset keeps the default.
        gemini_fallback_model=_fallback(os.getenv("GEMINI_FALLBACK_MODEL")),
        gemini_thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or "MINIMAL",
    )


def _fallback(raw: str | None) -> str | None:
    """No second model by default: resilience comes from the optional second key.

    Set GEMINI_FALLBACK_MODEL to add one. Empty string also means "none".
    """
    return raw or None
