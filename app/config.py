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
    #: Asked when the primary provider cannot answer at all. None disables it.
    llm_fallback_provider: str | None
    gemini_api_key: str | None
    #: Optional second key, tried when the first is rate limited.
    gemini_api_key_2: str | None
    gemini_model: str
    gemini_fallback_model: str | None
    #: MINIMAL, LOW, MEDIUM or HIGH. Some models accept only a subset.
    gemini_thinking_level: str
    groq_api_key: str | None
    groq_model: str


def load_settings(dotenv_path: str | Path | None = ".env") -> Settings:
    """Read `.env` (if given and present) into the process environment, then build Settings."""
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        llm_fallback_provider=os.getenv("LLM_FALLBACK_PROVIDER") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_api_key_2=os.getenv("GEMINI_API_KEY_2") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-3.6-flash",
        # Empty string opts out of the fallback; unset keeps the default.
        gemini_fallback_model=_fallback(os.getenv("GEMINI_FALLBACK_MODEL")),
        gemini_thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or "MINIMAL",
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
    )


#: Tried when the primary model is rate limited (429) or saturated (503).
#: A 503 is a property of the model, not of the key, so a second key cannot
#: escape one - only a different model can. This is the only current flash
#: model measured to accept our request config besides the primary, so it is
#: the emergency path rather than a preference: it is markedly slower.
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash-lite"

#: Groq's larger open model: better at multi-step planning than the 20B,
#: and one of the models that supports JSON-schema responses.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _fallback(raw: str | None) -> str | None:
    """Empty string disables the fallback; unset takes the default."""
    if raw is None:
        return DEFAULT_FALLBACK_MODEL
    return raw or None
