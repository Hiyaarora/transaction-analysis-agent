"""The analysis plan: the contract between the LLM and the executor.

The planner (LLM) produces JSON; this module decides whether that JSON is a
*well-formed* plan. It is Level 1 of two validation levels:

    Level 1  structural  (here)      is this the right shape, with allowed options?
    Level 2  semantic    (validator) does it make sense for the loaded dataset?

One rule decides what is a Literal and what is a plain string:

  * options that do not depend on the data — tool names, operators,
    aggregation functions, extreme modes, plan status — are Literals built
    from the tuples in `tools.py`. A wrong one is rejected here, by name.
  * names that depend on the data — columns, group keys, metrics — are plain
    strings. Whether `region` exists or `revenue` is available yet depends on
    the file and on step order, so the semantic validator checks them and can
    say why.

Every model forbids extra fields. A step that carries `"expression"`,
`"code"` or `"path"` is not a step with a suspicious extra; it is not a step.
"""

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.tools import AGG_FUNCS, EXTREME_MODES, FILTER_OPS, GROUP_FILTER_OPS

FilterOp = Literal[*FILTER_OPS]
GroupFilterOp = Literal[*GROUP_FILTER_OPS]
AggFunc = Literal[*AGG_FUNCS]
ExtremeMode = Literal[*EXTREME_MODES]
PlanStatus = Literal["success", "clarification_required", "rejected"]

# What a row filter may compare against. Which shape fits which operator
# (a list for `in`/`between`, nothing for `is_null`) is a semantic question.
Scalar = str | float
FilterValue = Scalar | list[Scalar] | None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- one model per tool --------------------------------------------------------


class FilterRowsStep(_Strict):
    tool: Literal["filter_rows"]
    column: str
    op: FilterOp
    value: FilterValue = None


class ComputeMetricStep(_Strict):
    tool: Literal["compute_metric"]
    metric: str


class AggregateStep(_Strict):
    tool: Literal["aggregate"]
    column: str
    func: AggFunc


class GroupByStep(_Strict):
    tool: Literal["group_by"]
    by: str
    func: AggFunc
    column: str | None = None  # not needed for count


class FilterGroupsStep(_Strict):
    tool: Literal["filter_groups"]
    op: GroupFilterOp
    value: float | list[float]  # a group aggregate is always a number


class SelectExtremeStep(_Strict):
    tool: Literal["select_extreme"]
    mode: ExtremeMode


# The discriminator makes `tool` decide which model applies, so an aggregate
# step cannot smuggle in a filter operator and an unknown tool fails by name.
Step = Annotated[
    Union[FilterRowsStep, ComputeMetricStep, AggregateStep, GroupByStep, FilterGroupsStep, SelectExtremeStep],
    Field(discriminator="tool"),
]


# --- the plan ------------------------------------------------------------------


class AnalysisPlan(_Strict):
    status: PlanStatus
    intent: str = Field(description="One sentence: what the question was understood to ask.")
    steps: list[Step] = Field(default_factory=list)
    clarification_question: str | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _status_fields_agree(self) -> "AnalysisPlan":
        has_question = bool(self.clarification_question and self.clarification_question.strip())
        has_reason = bool(self.rejection_reason and self.rejection_reason.strip())

        if self.status == "success":
            if not self.steps:
                raise ValueError("a success plan needs at least one step")
            if self.clarification_question is not None or self.rejection_reason is not None:
                raise ValueError("a success plan must not carry clarification_question or rejection_reason")
        elif self.status == "clarification_required":
            if not has_question:
                raise ValueError("clarification_required needs a clarification_question")
            if self.steps:
                raise ValueError("clarification_required must not carry steps")
        else:  # rejected
            if not has_reason:
                raise ValueError("rejected needs a rejection_reason")
            if self.steps:
                raise ValueError("rejected must not carry steps")
        return self


# --- entry point ---------------------------------------------------------------


class PlanParseError(Exception):
    """The LLM's output is not a well-formed plan. Carries a readable reason."""


def parse_plan(raw: str | dict) -> AnalysisPlan:
    """Turn the planner's raw output into an AnalysisPlan or raise PlanParseError."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Plan is not valid JSON: {exc.msg} at position {exc.pos}") from exc
    if not isinstance(raw, dict):
        raise PlanParseError(f"Plan must be a JSON object, got {type(raw).__name__}")
    try:
        return AnalysisPlan.model_validate(raw)
    except ValidationError as exc:
        raise PlanParseError(_summarise(exc)) from exc


def _summarise(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "plan"
        detail = err["msg"]
        if err["type"] == "literal_error":
            detail += f" (got {err['input']!r})"
        lines.append(f"{location}: {detail}")
    return "Plan failed structural validation: " + "; ".join(lines)
