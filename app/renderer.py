"""Turn an AgentResponse into the text the user reads.

This is the only module that writes prose, and it writes it from facts that
were already computed: the StepRecords the executor produced and the values
in the result objects. It has no access to the DataFrame, so it cannot
produce a number that Python did not calculate.

The answer line pairs the plan's `intent` (the model's wording) with the
value (Python's arithmetic). That is the project's boundary in one sentence.

Rounding happens here and nowhere else: computation keeps full precision,
presentation shows two decimals.
"""

from app import contract
from app.agent import AgentResponse
from app.data_profile import DataProfile
from app.executor import ExecutionResult, StepRecord
from app.tools import GroupResult

_OP_SYMBOL = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=" }
_OP_WORD = {"gt": "greater than", "gte": "at least", "lt": "less than", "lte": "at most",
            "eq": "equal to", "neq": "not equal to"}


def render(response: AgentResponse, profile: DataProfile) -> str:
    if response.status == "clarification_required":
        return f"Status: Clarification required\n{response.message}"
    if response.status == "rejected":
        return f"Status: Rejected\nReason: {response.message}"
    if response.status == "error":
        return f"Status: Error\n{response.message}"

    execution = response.execution
    sections = [
        f"Answer:\n{_answer(response, execution)}",
        "Operations performed:\n" + "\n".join(
            f"{i}. {_describe(step)}" for i, step in enumerate(execution.steps, start=1)
        ),
        f"Explanation:\n{_explanation(response, execution, profile)}",
    ]
    return "\n\n".join(sections)


# --- answer ---------------------------------------------------------------------------


def _answer(response: AgentResponse, execution: ExecutionResult) -> str:
    intent = (response.plan.intent or "Result").rstrip(".")

    if execution.no_data:
        return f"{intent}: No data. No transactions matched, so there is nothing to report."

    if execution.kind == "scalar":
        return f"{intent}: {_number(execution.scalar.value)}"

    if execution.kind == "groups":
        lines = [f"{intent}:"]
        width = max(len(str(name)) for name, *_ in execution.groups.rows)
        for name, value, _, _ in execution.groups.rows:
            shown = _number(value) if value is not None else "no value"
            lines.append(f"  {str(name).ljust(width)}  {shown}")
        return "\n".join(lines)

    extreme = execution.extreme
    names = ", ".join(extreme.groups)
    if len(extreme.groups) > 1:
        return f"{intent}: {names} (tied at {_number(extreme.value)})"
    return f"{intent}: {names} ({_number(extreme.value)})"


def _number(value) -> str:
    if isinstance(value, int):  # counts stay whole
        return f"{value:,}"
    return f"{value:,.2f}"


# --- operations -------------------------------------------------------------------------


def _describe(step: StepRecord) -> str:
    args = step.args

    if step.tool == "filter_rows":
        return (
            f"Filtered transactions where {_condition(args['column'], args['op'], args.get('value'))} "
            f"({step.rows_out} of {step.rows_in} rows kept)."
        )

    if step.tool == "compute_metric":
        metric = args["metric"]
        return (
            f"Computed {metric} for each of the {_rows(step.rows_in)} "
            f"({metric} = {contract.METRIC_FORMULAS[metric]})."
        )

    if step.tool == "aggregate":
        if args["func"] == "count":
            return f"Counted the {_rows(step.rows_in)}."
        return f"{_verb(args['func'])} {args['column']} over {_rows(step.rows_in)}."

    if step.tool == "group_by":
        if args["func"] == "count":
            body = f"counted the transactions in each group"
        else:
            body = f"{_verb(args['func']).lower()} {args['column']} within each group"
        return f"Grouped {step.rows_in} transactions by {args['by']} and {body} ({step.groups_out} groups)."

    if step.tool == "filter_groups":
        return (
            f"Kept groups whose value is {_group_condition(args['op'], args['value'])} "
            f"({step.groups_out} of {step.groups_in} groups kept)."
        )

    return f"Selected the group with the {step.args['mode']} value ({step.groups_in} groups compared)."


def _rows(count: int) -> str:
    return f"{count} matching transaction" + ("" if count == 1 else "s")


def _verb(func: str) -> str:
    return {"sum": "Summed", "mean": "Averaged", "median": "Took the median of",
            "min": "Took the minimum of", "max": "Took the maximum of"}[func]


def _condition(column: str, op: str, value) -> str:
    if op == "is_null":
        return f"{column} is missing"
    if op == "not_null":
        return f"{column} is present"
    if op == "in":
        return f"{column} is one of {', '.join(_plain(v) for v in value)}"
    if op == "between":
        return f"{column} between {_plain(value[0])} and {_plain(value[1])}"
    return f"{column} {_OP_SYMBOL[op]} {_plain(value)}"


def _group_condition(op: str, value) -> str:
    if op == "between":
        return f"between {_number(float(value[0]))} and {_number(float(value[1]))}"
    return f"{_OP_WORD[op]} {_number(float(value))}"


def _plain(value) -> str:
    """A filter value as the user would recognise it, without forcing 2 decimals."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --- explanation --------------------------------------------------------------------------


def _explanation(response: AgentResponse, execution: ExecutionResult, profile: DataProfile) -> str:
    lines = [
        f"The result was computed deterministically in Python from the active dataset "
        f"({profile.source_name}); the language model only chose which operations to run."
    ]

    if execution.no_data:
        if profile.date_range:
            start, end = profile.date_range
            lines.append(f"The dataset holds {profile.row_count} transactions spanning {start} to {end}.")
        else:
            lines.append(f"The dataset holds {profile.row_count} transactions.")

    if execution.kind == "scalar" and not execution.no_data:
        used, total = execution.scalar.rows_used, execution.scalar.rows_total
        if used != total:
            skipped = total - used
            lines.append(
                f"Computed from {used} of the {total} matching transactions; "
                f"{skipped} had a missing value and {'was' if skipped == 1 else 'were'} "
                f"left out rather than treated as zero."
            )

    if execution.kind in ("groups", "extreme"):
        groups = execution.groups if execution.kind == "groups" else None
        if groups is not None:
            lines.extend(_group_notes(groups))

    return "\n".join(lines)


def _group_notes(groups: GroupResult) -> list[str]:
    notes = []
    short = [(name, used, total) for name, _, total, used in groups.rows if used != total]
    if short:
        detail = "; ".join(f"{name}: {used} of {total}" for name, used, total in short)
        notes.append(f"Some groups had missing values and were computed from fewer rows ({detail}).")
    if groups.rows_without_group:
        count = groups.rows_without_group
        notes.append(
            f"{count} transaction{'' if count == 1 else 's'} had no group value and "
            f"{'was' if count == 1 else 'were'} excluded from the grouping."
        )
    if groups.groups_without_value:
        count = groups.groups_without_value
        notes.append(
            f"{count} group{'' if count == 1 else 's'} had no computable value and "
            f"{'was' if count == 1 else 'were'} not compared."
        )
    return notes
