"""Domain objects -> wire models. The only place tuples become named fields."""

from app.agent import AgentResponse
from app.api import schemas
from app.data_loader import ActiveDataset
from app.executor import ExecutionResult


def dataset_state(dataset: ActiveDataset) -> schemas.DatasetState:
    profile = dataset.profile
    return schemas.DatasetState(
        source_name=dataset.source_name,
        row_count=profile.row_count,
        columns=list(profile.columns),
        categorical_values={column: list(values) for column, values in profile.categorical_values.items()},
        numeric_ranges=profile.numeric_ranges,
        date_range=(
            schemas.DateRange(start=profile.date_range[0].isoformat(), end=profile.date_range[1].isoformat())
            if profile.date_range
            else None
        ),
        null_counts=profile.null_counts,
        parse_error_counts=profile.parse_error_counts,
        extra_columns=list(profile.extra_columns),
        supported_metrics=list(profile.supported_metrics),
        questions=list(dataset.questions),
    )


def ask_response(response: AgentResponse) -> schemas.AskResponse:
    if response.status in ("success", "no_data"):
        return schemas.AnswerResponse(
            status=response.status,
            question=response.question,
            intent=response.plan.intent if response.plan else "",
            execution=_execution(response.execution),
        )
    return schemas.NonAnswerResponse(
        status=response.status,
        question=response.question,
        message=response.message or "",
    )


def _execution(result: ExecutionResult) -> schemas.Execution:
    steps = [schemas.StepRecord(**vars(step)) for step in result.steps]

    if result.kind == "scalar":
        return schemas.ScalarExecution(
            kind="scalar",
            steps=steps,
            value=result.scalar.value,
            rows_total=result.scalar.rows_total,
            rows_used=result.scalar.rows_used,
        )

    if result.kind == "groups":
        groups = result.groups
        return schemas.GroupsExecution(
            kind="groups",
            steps=steps,
            by=groups.by,
            func=groups.func,
            column=groups.column,
            # The positional tuples the tools produce become named fields here.
            rows=[
                schemas.GroupRow(group=name, value=value, rows_total=total, rows_used=used)
                for name, value, total, used in groups.rows
            ],
            rows_without_group=groups.rows_without_group,
            groups_without_value=groups.groups_without_value,
        )

    extreme = result.extreme
    return schemas.ExtremeExecution(
        kind="extreme",
        steps=steps,
        mode=extreme.mode,
        groups=list(extreme.groups),
        value=extreme.value,
    )
