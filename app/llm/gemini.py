"""Gemini provider. The only module allowed to import `google.genai`.

Sends the planner's prompt with a JSON schema for constrained decoding and
returns the raw text. On a rate limit it retries once on the fallback model.
Every SDK failure surfaces as `LLMError`.
"""

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.base import LLMClient, LLMError

_RATE_LIMITED = 429


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        fallback_model: str | None = None,
        sdk: genai.Client | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env.")
        self.model = model
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else None
        # Injectable so tests can stub the SDK; built once because it owns a connection pool.
        self._sdk = sdk or genai.Client(api_key=api_key)

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
        try:
            return self._generate(self.model, user, config)
        except LLMError as primary:
            if self.fallback_model is None or getattr(primary, "code", None) != _RATE_LIMITED:
                raise
            return self._generate(self.fallback_model, user, config)

    def _generate(self, model: str, user: str, config: types.GenerateContentConfig) -> str:
        try:
            response = self._sdk.models.generate_content(model=model, contents=user, config=config)
        except genai_errors.APIError as exc:
            raise _LLMApiError(f"Gemini error {exc.code} on {model}: {exc.message}", code=exc.code) from exc
        text = response.text
        if not text:
            raise _LLMApiError(f"Gemini returned an empty reply on {model}.", code=None)
        return text


class _LLMApiError(LLMError):
    """LLMError that remembers the HTTP status, so the fallback decision can see it."""

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
