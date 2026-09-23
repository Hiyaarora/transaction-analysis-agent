"""Gemini provider. The only module allowed to import `google.genai`.

Sends the planner's prompt with a JSON schema for constrained decoding and
returns the raw text. Every SDK failure surfaces as `LLMError`.

Three failures are worth retrying: a rate limit (429), because the free tier
is small; "this model is experiencing high demand" (503), which is transient
and common on the flash models; and a dropped connection, which one model was
observed to do mid-request. The client walks a fixed sequence of attempts and
stops at the first success:

    (primary key, primary model) -> (primary key, fallback model)
    -> (backup key, primary model) -> (backup key, fallback model)

The fallback model is optional; with none configured the chain is just the
two keys. Anything else - a bad request, or a 404 because the model is not
available to this account - is a real failure and is raised immediately
rather than burning quota on a retry.

Whatever the SDK raises, callers only ever see LLMError. A transport failure
is not an APIError, so without translation an httpx exception would escape
the abstraction and reach the agent, which catches only LLMError - turning a
dropped connection into a crash instead of an honest "the model is
unavailable".
"""

from collections.abc import Callable, Iterator
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

# HTTP statuses worth trying the next key or model for.
_RETRYABLE_CODES = (429, 503)


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

        transient: LLMError | None = None
        for sdk, model in self._attempts():
            try:
                return self._generate(sdk, model, user, config)
            except _LLMApiError as exc:
                if not exc.retryable:
                    raise
                transient = exc  # try the next key or model
        raise transient  # every attempt hit a transient failure

    def _attempts(self) -> Iterator[tuple[Any, str]]:
        for sdk in self._sdks:
            yield sdk, self.model
            if self.fallback_model:
                yield sdk, self.fallback_model

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
