"""The wire contract: what the browser actually receives.

Domain objects use tuples and positional rows because that suits Python.
These models use named fields and plain JSON types because that suits a
typed client. Keeping them separate means the UI is not coupled to internal
representations, and the discriminated unions force the frontend to handle
every status and every result kind.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"]
    llm_configured: bool


class DateRange(BaseModel):
    start: str
    end: str


class DatasetState(BaseModel):
    source_name: str
    row_count: int
    columns: list[str]
    categorical_values: dict[str, list[str]]
    numeric_ranges: dict[str, tuple[float, float] | None]
    date_range: DateRange | None
    null_counts: dict[str, int]
    parse_error_counts: dict[str, int]
    extra_columns: list[str]
    supported_metrics: list[str]
    questions: list[str]


class StepRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    rows_in: int | None = None
    rows_out: int | None = None
    groups_in: int | None = None
    groups_out: int | None = None


class GroupRow(BaseModel):
    group: str
    value: float | int | None
    rows_total: int
    rows_used: int


class ScalarExecution(BaseModel):
    kind: Literal["scalar"]
    steps: list[StepRecord]
    value: float | int | None
    rows_total: int
    rows_used: int


class GroupsExecution(BaseModel):
    kind: Literal["groups"]
    steps: list[StepRecord]
    by: str
    func: str
    column: str | None
    rows: list[GroupRow]
    rows_without_group: int
    groups_without_value: int


class ExtremeExecution(BaseModel):
    kind: Literal["extreme"]
    steps: list[StepRecord]
    mode: str
    groups: list[str]
    value: float | int | None


Execution = ScalarExecution | GroupsExecution | ExtremeExecution


class AnswerResponse(BaseModel):
    status: Literal["success", "no_data"]
    question: str
    intent: str
    execution: Execution = Field(discriminator="kind")


class NonAnswerResponse(BaseModel):
    status: Literal["clarification_required", "rejected", "error"]
    question: str
    message: str


AskResponse = AnswerResponse | NonAnswerResponse


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ApiError(BaseModel):
    code: str
    message: str
