"""Gemini provider. The only module allowed to import `google.genai`.

Sends the planner's prompt with a JSON schema for constrained decoding and
returns the raw text. Every SDK failure surfaces as `LLMError`.

Three failures are worth retrying, and which escape helps depends on which
one occurred:

    429  this key's quota is spent        -> try the other KEY, same model
    503  the model is busy for everyone   -> try the other MODEL, same key
    dropped connection                    -> like 503; the model may be sick

Retrying a 503 on a second key just repeats the request that failed, because
capacity is a property of the model rather than of the key. Sending a 429 to
a different model on the same key may or may not help, since quota can be
shared across models, so the alternate key is tried first and the other model
only if no alternate key is configured.

Each escape falls back to the other kind when its preferred one is not
configured, and no combination of key and model is attempted twice. The chain
is capped at MAX_ATTEMPTS: a provider having a bad minute must not turn one
question into a burst of requests. Anything else - a bad request, or a 404
because the model is not available to this account - is raised immediately
rather than burning quota on a retry.

Whatever the SDK raises, callers only ever see LLMError. A transport failure
is not an APIError, so without translation an httpx exception would escape
the abstraction and reach the agent, which catches only LLMError - turning a
dropped connection into a crash instead of an honest "the model is
unavailable".
"""

from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.base import LLMClient, LLMError

# How hard the model may reason before answering. Building a plan is a
# translation task, not a reasoning one, so the cheapest setting is right.
#
# This must be `thinking_level`, not the Gemini 2.x `thinking_budget`: 3.x
# models either reject that parameter outright or ignore it and reason at
# full depth, which measured 30-55s per plan against 10-12s here.
DEFAULT_THINKING_LEVEL = "MINIMAL"

# HTTP statuses worth escaping from rather than surfacing straight away.
_RETRYABLE_CODES = (429, 503)

# A question is one request plus, at most, two escapes. A provider having a
# bad minute must not turn one question into a burst of traffic.
MAX_ATTEMPTS = 3

# (index into self._sdks, model name)
_Attempt = tuple[int, str]


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        fallback_model: str | None = None,
        backup_api_key: str | None = None,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env.")
        self.model = model
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        self.backup_configured = bool(backup_api_key and backup_api_key != api_key)
        self.thinking_level = thinking_level

        build = sdk_factory or (lambda key: genai.Client(api_key=key))
        keys = [api_key] + ([backup_api_key] if self.backup_configured else [])
        # Clients are built once: each owns a connection pool.
        self._sdks = [build(key) for key in keys]

    def complete_json(self, *, system: str, user: str, schema: dict) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=_gemini_schema(schema),
            temperature=0,  # planning should be repeatable
            # Not `thinking_budget`: see DEFAULT_THINKING_LEVEL.
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            # We pass no tools; without this the SDK logs an AFC warning on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        attempt: _Attempt = (0, self.model)
        tried: set[_Attempt] = set()
        transient: LLMError | None = None

        for _ in range(MAX_ATTEMPTS):
            tried.add(attempt)
            key_index, model = attempt
            try:
                return self._generate(self._sdks[key_index], model, user, config)
            except _LLMApiError as exc:
                if not exc.retryable:
                    raise
                transient = exc
                following = self._escape_from(exc, attempt, tried)
                if following is None:
                    break  # nothing left that has not already failed
                attempt = following

        raise transient  # every escape was exhausted

    def _escape_from(self, failure: "_LLMApiError", attempt: _Attempt, tried: set[_Attempt]) -> _Attempt | None:
        """Where to go after `failure`, or None when there is nowhere useful left."""
        key_index, model = attempt
        other_key = 1 - key_index if len(self._sdks) > 1 else None
        other_model = self.fallback_model if model == self.model else self.model

        change_key = (other_key, model) if other_key is not None else None
        change_model = (key_index, other_model) if other_model else None
        # A spent quota is a property of the key; a busy model is a property of
        # the model. Prefer the escape that matches the failure, then take the
        # other one, then any combination not yet tried.
        preferred = [change_key, change_model] if failure.code == 429 else [change_model, change_key]
        both_changed = (other_key, other_model) if other_key is not None and other_model else None

        for candidate in [*preferred, both_changed]:
            if candidate and candidate not in tried:
                return candidate
        return None

    def _generate(self, sdk: Any, model: str, user: str, config: types.GenerateContentConfig) -> str:
        try:
            response = sdk.models.generate_content(model=model, contents=user, config=config)
        except genai_errors.APIError as exc:
            raise _LLMApiError(
                f"Gemini error {exc.code} on {model}: {exc.message}",
                code=exc.code,
                retryable=exc.code in _RETRYABLE_CODES,
            ) from exc
        except Exception as exc:
            # Connection resets, timeouts, anything else the SDK or its HTTP
            # stack throws. Wrapped rather than enumerated: the contract is
            # that no vendor exception type reaches the caller. The original
            # is chained, so nothing is hidden from a traceback.
            raise _LLMApiError(f"Gemini request to {model} failed: {exc}", retryable=True) from exc

        text = response.text
        if not text:
            # A reply with no content is a real failure, not a transient one.
            raise _LLMApiError(f"Gemini returned an empty reply on {model}.")
        return text


class _LLMApiError(LLMError):
    """LLMError that remembers whether trying another key or model is worthwhile."""

    def __init__(self, message: str, code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _gemini_schema(schema: dict) -> dict:
    """Rewrite a Pydantic JSON schema into the subset Gemini documents.

    `const` becomes a one-value `enum` (same meaning) and the OpenAPI-style
    `discriminator` hint is dropped (the `anyOf` it annotates stays). The
    input is not modified.
    """

    def walk(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key == "discriminator":
                    continue
                if key == "const":
                    out["enum"] = [value]
                    continue
                out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)
