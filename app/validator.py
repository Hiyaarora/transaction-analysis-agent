"""Semantic validation: does a well-formed plan make sense for the loaded data?

Level 1 (`schemas.py`) proved the plan has the right shape and only uses
allowed options. This module is Level 2. It walks the steps in order, against
the contract and the dataset's profile, and answers three questions no schema
can:

  * Is every name real *here* — column, metric, category value — and is a
    derived column only used after the step that creates it?
  * Do operator, value and column type agree (a date column gets ISO dates, a
    numeric column gets numbers, `between` gets two ordered bounds)?
  * Do the steps chain — filters and grouping on rows, group filters and
    extremes on groups, one result at the end?

Category values follow one deterministic rule and never involve semantics:
exact match accepted; a case-insensitive match to exactly one value is
normalised; a close misspelling asks for clarification with the candidate;
anything else is rejected with the list of real values. Synonyms ("Britain"
for "UK") have no string evidence and are therefore rejected here — deciding
that a word *means* a category is the planner's job, and its only allowed
move is to ask.

The validator never mutates the plan it was given. A valid outcome carries a
normalised copy; every other outcome carries the original plus a reason.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app import contract
from app.data_profile import DataProfile
from app.schemas import (
    AggregateStep,
    AnalysisPlan,
    ComputeMetricStep,
    FilterGroupsStep,
    FilterRowsStep,
    GroupByStep,
    SelectExtremeStep,
)

OutcomeStatus = Literal["valid", "clarification_required", "rejected"]
Stage = Literal["rows", "scalar", "groups", "extreme"]

_NEAR_MISS_CUTOFF = 0.75


@dataclass(frozen=True)
class ValidationOutcome:
    status: OutcomeStatus
    plan: AnalysisPlan  # exactly what the planner produced, untouched
    validated_plan: AnalysisPlan | None = None  # normalised copy, only when valid
    clarification_question: str | None = None
    rejection_reason: str | None = None


class _Reject(Exception):
    pass


class _Clarify(Exception):
    pass


def validate(plan: AnalysisPlan, profile: DataProfile, question: str = "") -> ValidationOutcome:
    """Check `plan` against the dataset, and against the words of `question`.

    `question` is optional so the validator can be exercised on its own, but
    the agent always supplies it: without it, a value the user never wrote
    cannot be detected.
    """
    # The planner's own decision to ask or refuse is honoured as-is.
    if plan.status == "clarification_required":
        return ValidationOutcome("clarification_required", plan, clarification_question=plan.clarification_question)
    if plan.status == "rejected":
        return ValidationOutcome("rejected", plan, rejection_reason=plan.rejection_reason)

    try:
        steps = _Walk(profile, question).run(plan)
    except _Reject as exc:
        return ValidationOutcome("rejected", plan, rejection_reason=str(exc))
    except _Clarify as exc:
        return ValidationOutcome("clarification_required", plan, clarification_question=str(exc))

    return ValidationOutcome("valid", plan, validated_plan=plan.model_copy(update={"steps": steps}))


# --- the step walk ---------------------------------------------------------------


class _Walk:
    """State carried from one step to the next: which columns exist, what stage we are at."""

    def __init__(self, profile: DataProfile, question: str = "") -> None:
        self.profile = profile
        self.question = question
        self.stage: Stage = "rows"
        # Only contract columns are addressable. Extra file columns exist but
        # are not supported for analysis; derived metrics appear after compute.
        self.kinds: dict[str, str] = {
            contract.ID_COLUMN: "text",
            contract.DATE_COLUMN: "date",
            **{c: "categorical" for c in contract.CATEGORICAL_COLUMNS},
            **{c: "numeric" for c in contract.NUMERIC_COLUMNS},
        }

    def run(self, plan: AnalysisPlan) -> list:
        handlers = {
            FilterRowsStep: self.filter_rows,
            ComputeMetricStep: self.compute_metric,
            AggregateStep: self.aggregate,
            GroupByStep: self.group_by,
            FilterGroupsStep: self.filter_groups,
            SelectExtremeStep: self.select_extreme,
        }
        out = []
        for index, step in enumerate(plan.steps, start=1):
            out.append(handlers[type(step)](step, index) or step)
        if self.stage == "rows":
            raise _Reject("The plan filters rows but never produces a result; it must end with an aggregation, grouping or comparison.")
        return out

    # --- row-stage steps ---

    def filter_rows(self, step: FilterRowsStep, index: int):
        self._need_stage("rows", step.tool, index)
        kind = self._column_kind(step.column, index)
        op, value = step.op, step.value

        if op in ("is_null", "not_null"):
            if value is not None:
                raise _Reject(f"Step {index}: '{op}' takes no value.")
            return step

        if kind == "categorical" and op in ("gt", "gte", "lt", "lte", "between"):
            raise _Reject(f"Step {index}: '{op}' cannot order the category column '{step.column}'.")
        if kind == "text" and op not in ("eq", "neq", "in"):
            raise _Reject(f"Step {index}: '{op}' is not supported on '{step.column}'.")

        if op == "in":
            if not isinstance(value, list) or not value:
                raise _Reject(f"Step {index}: 'in' needs a non-empty list of values.")
            return step.model_copy(update={"value": [self._typed(v, kind, step.column, index) for v in value]})

        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise _Reject(f"Step {index}: 'between' needs exactly two values [low, high].")
            low, high = (self._typed(v, kind, step.column, index) for v in value)
            if low > high:  # floats and ISO date strings both order correctly as-is
                raise _Reject(f"Step {index}: 'between' bounds are out of order ({low} is after {high}).")
            return step.model_copy(update={"value": [low, high]})

        if isinstance(value, list) or value is None:
            raise _Reject(f"Step {index}: '{op}' needs a single value.")
        return step.model_copy(update={"value": self._typed(value, kind, step.column, index)})

    def compute_metric(self, step: ComputeMetricStep, index: int):
        self._need_stage("rows", step.tool, index)
        if step.metric not in contract.DERIVED_METRICS:
            raise _Reject(
                f"Step {index}: '{step.metric}' is not a supported metric. Supported: {', '.join(contract.DERIVED_METRICS)}."
            )
        missing = [c for c in contract.DERIVED_METRICS[step.metric] if c not in self.kinds]
        if missing:
            raise _Reject(f"Step {index}: '{step.metric}' needs column(s) {', '.join(missing)}.")
        self.kinds[step.metric] = "numeric"

    def aggregate(self, step: AggregateStep, index: int):
        self._need_stage("rows", step.tool, index)
        kind = self._column_kind(step.column, index)
        if step.func != "count" and kind != "numeric":
            raise _Reject(f"Step {index}: '{step.func}' needs a numeric column; '{step.column}' is {kind}.")
        self.stage = "scalar"

    def group_by(self, step: GroupByStep, index: int):
        self._need_stage("rows", step.tool, index)
        if step.by not in contract.CATEGORICAL_COLUMNS:
            raise _Reject(
                f"Step {index}: cannot group by '{step.by}'. Group columns: {', '.join(contract.CATEGORICAL_COLUMNS)}."
            )
        if step.func != "count":
            if step.column is None:
                raise _Reject(f"Step {index}: '{step.func}' needs a column to aggregate.")
            kind = self._column_kind(step.column, index)
            if kind != "numeric":
                raise _Reject(f"Step {index}: '{step.func}' needs a numeric column; '{step.column}' is {kind}.")
        self.stage = "groups"

    # --- group-stage steps ---

    def filter_groups(self, step: FilterGroupsStep, index: int):
        self._need_stage("groups", step.tool, index)
        if step.op == "between":
            if not isinstance(step.value, list) or len(step.value) != 2:
                raise _Reject(f"Step {index}: 'between' needs exactly two values [low, high].")
            low, high = step.value
            if low > high:
                raise _Reject(f"Step {index}: 'between' bounds are out of order ({low} > {high}).")
        elif isinstance(step.value, list):
            raise _Reject(f"Step {index}: '{step.op}' needs a single number.")

    def select_extreme(self, step: SelectExtremeStep, index: int):
        self._need_stage("groups", step.tool, index)
        self.stage = "extreme"

    # --- helpers ---

    def _need_stage(self, stage: Stage, tool: str, index: int) -> None:
        if self.stage == stage:
            return
        if stage == "groups":
            raise _Reject(f"Step {index}: '{tool}' must follow a group_by step.")
        raise _Reject(f"Step {index}: '{tool}' operates on rows, but the plan already produced a result at an earlier step.")

    def _column_kind(self, column: str, index: int) -> str:
        if column in self.kinds:
            return self.kinds[column]
        if column in contract.DERIVED_METRICS:
            raise _Reject(f"Step {index}: '{column}' is not available yet; add a compute_metric step for it first.")
        supported = ", ".join(list(self.kinds) + [m for m in contract.DERIVED_METRICS if m not in self.kinds])
        raise _Reject(f"Step {index}: '{column}' is not a column or metric of this dataset. Available: {supported}.")

    def _typed(self, value, kind: str, column: str, index: int):
        """Check a filter value against the column's kind; return it normalised."""
        if kind == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _Reject(f"Step {index}: '{column}' is numeric; {value!r} is not a number.")
            return float(value)
        if kind == "date":
            parsed = pd.to_datetime(value, errors="coerce", format="ISO8601") if isinstance(value, str) else pd.NaT
            if pd.isna(parsed):
                raise _Reject(f"Step {index}: '{value}' is not a date in YYYY-MM-DD form for column '{column}'.")
            return parsed.date().isoformat()
        if not isinstance(value, str):
            raise _Reject(f"Step {index}: '{column}' holds text; {value!r} is not text.")
        if kind == "categorical":
            return self._resolve_category(value, column, index)
        return value

    def _resolve_category(self, value: str, column: str, index: int) -> str:
        valid = self.profile.categorical_values.get(column, ())
        if value in valid:
            self._require_the_user_said_it(value, column, valid)
            return value
        folded = [v for v in valid if v.casefold() == value.strip().casefold()]
        if len(folded) == 1:
            self._require_the_user_said_it(value, column, valid)
            return folded[0]
        close = difflib.get_close_matches(value, valid, n=3, cutoff=_NEAR_MISS_CUTOFF)
        if close:
            options = " or ".join(f"'{c}'" for c in close)
            raise _Clarify(f"'{value}' is not a {column} in the dataset. Did you mean {options}?")
        raise _Reject(f"Step {index}: '{value}' is not a {column} in this dataset. Values: {', '.join(valid)}.")

    def _require_the_user_said_it(self, value: str, column: str, valid: tuple[str, ...]) -> None:
        """A valid value the user never wrote means the planner substituted one.

        Matching is on whole words, so a two-letter code is not counted as
        named because it happens to sit inside another word ("DE" in
        "Denmark"). The clarification lists the real values and does not repeat
        the planner's guess: what the user meant is the thing being asked.
        """
        if not self.question:
            return
        if re.search(rf"\b{re.escape(value)}\b", self.question, re.IGNORECASE):
            return
        raise _Clarify(
            f"Your question does not name a {column} from this dataset. "
            f"The available {column} values are {', '.join(valid)}. Which did you mean?"
        )

