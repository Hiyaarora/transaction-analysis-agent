"""A scripted LLM for tests: returns what it was told to, remembers what it was asked."""

from app.llm.base import LLMCall, LLMClient


class FakeLLMClient(LLMClient):
    name = "fake"
    model = "scripted"

    def __init__(self, responses: list[str | Exception]) -> None:
        # Consumed front to back. An Exception entry is raised instead of returned,
        # which is how tests simulate a provider failure.
        self._responses = list(responses)
        self.calls: list[LLMCall] = []

    def complete_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls.append(LLMCall(system=system, user=user, schema=schema))
        if not self._responses:
            raise RuntimeError("FakeLLMClient has no scripted response left for this call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
