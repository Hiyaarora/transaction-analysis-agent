"""The agent: one question in, one response out. This is where the boundary is wired.

    question
      -> prescreen           (deterministic; obvious code/file/secret requests stop here)
      -> Planner.plan        (LLM: probabilistic; sees the profile, never the rows)
      -> validate            (deterministic; checks the plan against the profile)
      -> execute             (deterministic; pandas over the active transactions)
      -> AgentResponse

The agent owns the ACTIVE DATASET for a session. Every question runs against
it; `load()` replaces it and rebuilds the planner's context. Nothing else in
the application holds a DataFrame.
"""

from dataclasses import dataclass
from typing import Literal

from app.data_loader import ActiveDataset
from app.executor import ExecutionError, ExecutionResult, execute
from app.llm.base import LLMClient, LLMError
from app.planner import Planner
from app.prescreen import prescreen
from app.schemas import AnalysisPlan
from app.validator import validate

Status = Literal["success", "no_data", "clarification_required", "rejected", "error"]


@dataclass(frozen=True)
class AgentResponse:
    status: Status
    question: str
    plan: AnalysisPlan | None = None  # what the planner produced (original, unnormalised)
    execution: ExecutionResult | None = None  # present for success and no_data
    message: str | None = None  # clarification question, rejection reason, or error text


class Agent:
    def __init__(self, dataset: ActiveDataset, llm: LLMClient) -> None:
        self._llm = llm
        self.load(dataset)

    def load(self, dataset: ActiveDataset) -> None:
        """Replace the active dataset. The planner is rebuilt so its context matches."""
        self.dataset = dataset
        self._planner = Planner(self._llm, dataset.profile)

    def ask(self, question: str) -> AgentResponse:
        screen = prescreen(question)
        if screen.blocked:
            return AgentResponse("rejected", question, message=screen.reason)

        try:
            plan = self._planner.plan(question)
        except LLMError as exc:
            return AgentResponse("error", question, message=f"The planning model is unavailable: {exc}")

        # The question goes in too: a category value the user never wrote
        # is how a silently mapped synonym shows up.
        outcome = validate(plan, self.dataset.profile, question=question)
        if outcome.status == "clarification_required":
            return AgentResponse("clarification_required", question, plan, message=outcome.clarification_question)
        if outcome.status == "rejected":
            return AgentResponse("rejected", question, plan, message=outcome.rejection_reason)

        try:
            result = execute(outcome.validated_plan, self.dataset.transactions)
        except ExecutionError as exc:
            return AgentResponse("error", question, plan, message=str(exc))

        return AgentResponse("no_data" if result.no_data else "success", question, plan, result)
