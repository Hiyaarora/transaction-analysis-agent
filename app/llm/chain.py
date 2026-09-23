"""Ask a second provider when the first cannot answer.

Each provider already retries within itself - another key, another model. This
covers what that cannot: the provider as a whole being unreachable, out of
quota for the day, or refusing every model. Groq and Gemini fail for unrelated
reasons, so one being unavailable says little about the other.

Exactly two attempts, one per provider. The providers' own retry logic sits
underneath; nothing here loops.
"""

from app.llm.base import LLMClient, LLMError


class FallbackLLMClient(LLMClient):
    def __init__(self, primary: LLMClient, backup: LLMClient) -> None:
        self._primary = primary
        self._backup = backup
        self.name = f"{primary.name}+{backup.name}"
        # The model a question will actually be planned with, absent a failure.
        self.model = primary.model

    def complete_json(self, *, system: str, user: str, schema: dict) -> str:
        try:
            return self._primary.complete_json(system=system, user=user, schema=schema)
        except LLMError as primary_failure:
            try:
                return self._backup.complete_json(system=system, user=user, schema=schema)
            except LLMError as backup_failure:
                # One error naming both, so the report says what was actually
                # tried rather than only how the last attempt ended.
                raise LLMError(
                    f"Both planning providers failed. "
                    f"{self._primary.name}: {primary_failure} | "
                    f"{self._backup.name}: {backup_failure}"
                ) from backup_failure
