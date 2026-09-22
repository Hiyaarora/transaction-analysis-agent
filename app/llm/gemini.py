"""Gemini provider. The only module allowed to import `google.genai`.

Sends the planner's prompt with a JSON schema for constrained decoding and
returns the raw text. Every SDK failure surfaces as `LLMError`.

Two failures are worth retrying: a rate limit (429), because the free tier is
small, and "this model is experiencing high demand" (503), which is transient
and common on the flash models. The client walks a fixed sequence of attempts
and stops at the first success:

    (primary key, primary model) -> (primary key, fallback model)
    -> (backup key, primary model) -> (backup key, fallback model)

The fallback model is optional; with none configured the chain is just the
two keys. Anything else - a bad request, or a 404 because the model is not
available to this account - is a real failure and is raised immediately
rather than burning quota on a retry.
"""

from collections.abc import Callable, Iterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.base import LLMClient, LLMError

# Transient conditions worth trying the next key or model for.
_RETRYABLE = (429, 503)


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        fallback_model: str | None = None,
        backup_api_key: str | None = None,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env.")
        self.model = model
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        self.backup_configured = bool(backup_api_key and backup_api_key != api_key)

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
            # Flash reasons before answering and bills it as output; a plan needs none of that.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # We pass no tools; without this the SDK logs an AFC warning on every call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        transient: LLMError | None = None
        for sdk, model in self._attempts():
            try:
                return self._generate(sdk, model, user, config)
            except _LLMApiError as exc:
                if exc.code not in _RETRYABLE:
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
            raise _LLMApiError(f"Gemini error {exc.code} on {model}: {exc.message}", code=exc.code) from exc
        text = response.text
        if not text:
            raise _LLMApiError(f"Gemini returned an empty reply on {model}.", code=None)
        return text


class _LLMApiError(LLMError):
    """LLMError that remembers the HTTP status, so the retry decision can see it."""

    def __init__(self, message: str, code: int | None) -> None:
        super().__init__(message)
        self.code = code


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
