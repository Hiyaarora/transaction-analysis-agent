"""The contract every LLM provider implements.

The planner depends on `LLMClient` and nothing else. A provider is one module
that turns (system prompt, user prompt, JSON schema) into the model's raw text
and translates its own errors into `LLMError`. Nothing vendor-specific leaks
upward: no SDK response objects, no vendor exceptions.

The method returns text, not a parsed object, on purpose. Parsing and
validating the plan is `schemas.parse_plan`'s job; a client that parsed would
be a second place where the plan could be "helpfully" reshaped.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(Exception):
    """The provider could not produce a response (auth, network, rate limit, empty reply)."""


@dataclass(frozen=True)
class LLMCall:
    """One request as the planner sent it. Recorded by the fake so tests can inspect it."""

    system: str
    user: str
    schema: dict


class LLMClient(ABC):
    #: Short identifier for logs, e.g. "gemini".
    name: str
    #: The concrete model in use, e.g. "gemini-2.5-flash".
    model: str

    @abstractmethod
    def complete_json(self, *, system: str, user: str, schema: dict) -> str:
        """Return the model's reply as raw text, expected to be JSON matching `schema`.

        Providers that support constrained decoding pass `schema` through;
        others may ignore it — the schema is also described in the prompt.
        Raises LLMError on any provider failure.
        """
