"""Run a validated plan against the active transactions. Deterministic, pandas only.

This is the entry point of the deterministic half. It is deliberately dumb: a
dispatch table from step type to the tool in `tools.py`, plus typed state
that moves through three stages:

    DataFrame  --aggregate-->  AggregateResult                 (scalar)
    DataFrame  --group_by--->  GroupResult  --filter_groups--> GroupResult  --select_extreme--> ExtremeResult

It records facts about each step (rows in, rows out, groups in, groups out)
and never writes prose; the renderer turns records into sentences.

Only a plan the validator approved should arrive here. If a tool still
objects, that is a validation gap: it surfaces as ExecutionError, never as a
crash and never as a made-up number.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from app import tools
from app.schemas import (
    AggregateStep,
    AnalysisPlan,
    ComputeMetricStep,
    FilterGroupsStep,
    FilterRowsStep,
    GroupByStep,
    SelectExtremeStep,
)
from app.tools import AggregateResult, ExtremeResult, GroupResult, ToolError


class ExecutionError(Exception):
    """The plan could not be executed as validated."""


@dataclass(frozen=True)
class StepRecord:
    tool: str
    args: dict[str, Any]
    rows_in: int | None = None
    rows_out: int | None = None
    groups_in: int | None = None
    groups_out: int | None = None


@dataclass(frozen=True)
class ExecutionResult:
    kind: Literal["scalar", "groups", "extreme"]
    steps: list[StepRecord] = field(default_factory=list)
    scalar: AggregateResult | None = None
    groups: GroupResult | None = None
    extreme: ExtremeResult | None = None

    @property
    def no_data(self) -> bool:
        """True when there was nothing to compute from. A count of zero is data."""
        if self.kind == "scalar":
            return self.scalar.value is None
        if self.kind == "groups":
            return not self.groups.rows
        return not self.extreme.groups


def execute(plan: AnalysisPlan, transactions: pd.DataFrame) -> ExecutionResult:
    if plan.status != "success":
        raise ExecutionError(f"Only a success plan can be executed (got status '{plan.status}').")

    state: pd.DataFrame | AggregateResult | GroupResult | ExtremeResult = transactions
    records: list[StepRecord] = []
    try:
        for step in plan.steps:
            state, record = _run_step(step, state)
            records.append(record)
    except ToolError as exc:
        raise ExecutionError(f"Step '{step.tool}' could not run: {exc}") from exc

    if isinstance(state, AggregateResult):
        return ExecutionResult(kind="scalar", steps=records, scalar=state)
    if isinstance(state, GroupResult):
        return ExecutionResult(kind="groups", steps=records, groups=state)
    if isinstance(state, ExtremeResult):
        return ExecutionResult(kind="extreme", steps=records, extreme=state)
    raise ExecutionError("The plan produced no result.")


def _run_step(step, state):
    args = step.model_dump(exclude={"tool"}, exclude_none=True)

    if isinstance(step, FilterRowsStep):
        out = tools.filter_rows(state, step.column, step.op, step.value)
        return out, StepRecord(step.tool, args, rows_in=len(state), rows_out=len(out))

    if isinstance(step, ComputeMetricStep):
        out = tools.compute_metric(state, step.metric)
        return out, StepRecord(step.tool, args, rows_in=len(state), rows_out=len(out))

    if isinstance(step, AggregateStep):
        out = tools.aggregate(state, step.column, step.func)
        return out, StepRecord(step.tool, args, rows_in=len(state))

    if isinstance(step, GroupByStep):
        out = tools.group_by(state, step.by, step.column, step.func)
        return out, StepRecord(step.tool, args, rows_in=len(state), groups_out=len(out.rows))

    if isinstance(step, FilterGroupsStep):
        out = tools.filter_groups(state, step.op, step.value)
        return out, StepRecord(step.tool, args, groups_in=len(state.rows), groups_out=len(out.rows))

    if isinstance(step, SelectExtremeStep):
        out = tools.select_extreme(state, step.mode)
        return out, StepRecord(step.tool, args, groups_in=len(state.rows), groups_out=len(out.groups))

    raise ExecutionError(f"No executor for step type {type(step).__name__}.")  # unreachable by construction
