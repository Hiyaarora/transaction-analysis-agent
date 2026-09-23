"""Groq provider. The only module allowed to import `groq`.

Exists because planning latency on Gemini's free tier was measured between 16
and 93 seconds, with roughly a quarter of calls returning 503. Groq's platform
is built around low-latency inference, so it is worth measuring as an
alternative. Whether it is actually faster here is a question for a benchmark.

The retry story is deliberately smaller than the Gemini client's. There is one
key and one model, so the only escape from a transient failure is to try the
same request again, once. Anything else - a bad request, an unknown model - is
raised immediately.

Structured output uses `response_format: json_schema` in best-effort mode
rather than Groq's strict mode. Strict mode requires every property to be
required and `additionalProperties: false` throughout, which the plan schema
does not satisfy: it has genuinely optional fields (a clarification carries no
steps, a filter on `is_null` carries no value). Correctness does not depend on
the provider honouring the schema in any case - `parse_plan` validates every
reply and the planner retries once on a malformed one.
"""

from collections.abc import Callable
from typing import Any

import groq

from app.llm.base import LLMClient, LLMError

# Statuses worth one more attempt: a rate limit, or the service being busy.
_RETRYABLE_CODES = (429, 500, 502, 503, 504)

# One request plus, at most, one retry. With a single key and a single model
# there is nothing else to vary, so further attempts would just repeat.
MAX_ATTEMPTS = 2


class GroqClient(LLMClient):
    name = "groq"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        sdk_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env.")
        self.model = model
        build = sdk_factory or (lambda key: groq.Groq(api_key=key).chat)
        self._sdk = build(api_key)

    def complete_json(self, *, system: str, user: str, schema: dict) -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,  # planning should be repeatable
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "analysis_plan", "schema": schema},
            },
        }

        transient: LLMError | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                return self._generate(request)
            except _GroqApiError as exc:
                if not exc.retryable:
                    raise
                transient = exc
        raise transient

    def _generate(self, request: dict) -> str:
        try:
            response = self._sdk.completions.create(**request)
        except groq.APIStatusError as exc:
            raise _GroqApiError(
                f"Groq error {exc.status_code} on {self.model}: {_message(exc)}",
                retryable=exc.status_code in _RETRYABLE_CODES,
            ) from exc
        except Exception as exc:
            # Connection resets, timeouts, anything else the SDK throws.
            # Wrapped rather than enumerated: no vendor exception type reaches
            # the caller. The original is chained, so a traceback keeps it.
            raise _GroqApiError(f"Groq request to {self.model} failed: {exc}", retryable=True) from exc

        text = response.choices[0].message.content
        if not text:
            raise _GroqApiError(f"Groq returned an empty reply on {self.model}.")
        return text


def _message(exc: groq.APIStatusError) -> str:
    """The provider's own explanation, without the request details around it."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(getattr(exc, "message", exc))


class _GroqApiError(LLMError):
    """LLMError that remembers whether another attempt is worthwhile."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
