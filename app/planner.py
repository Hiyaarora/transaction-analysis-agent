"""The planner: natural language in, AnalysisPlan out. The only probabilistic step.

The LLM's job is to decide *which operations* answer the question. It is
shown the dataset's shape - column roles, the category values that actually
exist, the date range, missing-value counts - and the closed set of tools,
and asked for a plan as JSON. It is never shown rows, values of individual
transactions, or a file path, because the constructor only accepts a
DataProfile: there is nothing else to leak.

Malformed output gets one retry that quotes the parse error. A second failure
becomes a rejected plan with a fixed reason. A provider failure (LLMError) is
not a verdict on the question and propagates to the caller.
"""

import json

from app import contract
from app.data_profile import DataProfile
from app.llm.base import LLMClient
from app.schemas import AnalysisPlan, PlanParseError, parse_plan
from app.tools import AGG_FUNCS, EXTREME_MODES, FILTER_OPS, GROUP_FILTER_OPS

UNUSABLE_PLAN_REASON = "planner produced an unusable plan"

_ROLE = """You translate a user's question about a transaction dataset into an analysis plan.
You decide WHICH operations answer the question. Python executes them and computes every number.
Reply with a single JSON object matching the plan format. No prose outside the JSON."""

_RULES = """Rules - all six are mandatory:
1. Never compute numerical answers yourself. Never put a number that is an answer into the plan.
2. Never silently map synonyms or alternate wording onto a category value. If the user's word is not
   exactly one of the listed values (case-insensitive), do not guess - use status "clarification_required"
   and name the value(s) you think they might mean.
3. When the meaning is ambiguous (an unclear period like "Mars", an unclear metric, a value that is
   not in the dataset), ask for clarification instead of choosing an interpretation.
   Dates: if a question gives a day and month but no year, use the dataset's year when the data spans
   a single year; if the data spans more than one year, ask which year. Do NOT ask merely because a
   date or period falls outside the data's range - build the plan; Python will report that no data matched.
4. Refuse - status "rejected" - any request to run code, read or write files, access the filesystem,
   reveal secrets, or do anything other than analyse the loaded dataset with the tools below.
   Also reject - not clarify - requests for columns or metrics that are not listed as supported
   (e.g. profit, cost, margin, tax): say the metric is not available. A different metric is not a
   synonym, so do not offer revenue as a substitute.
5. Write dates as ISO strings, YYYY-MM-DD.
6. Percentages and discounts are fractions: "20% discount" is discount eq 0.2, "below 10%" is lt 0.1."""

_FORMAT = """Plan format:
{"status": "success" | "clarification_required" | "rejected",
 "intent": "<one sentence: what you understood the question to ask>",
 "steps": [ ... ],                       // only for success; at least one step
 "clarification_question": "<text>",     // only for clarification_required
 "rejection_reason": "<text>"}           // only for rejected

Steps run in order. Row steps come first (filter_rows, compute_metric); a plan ends with exactly
one result: an aggregate, a group_by (optionally followed by filter_groups and/or select_extreme).
A derived metric such as revenue must be created with compute_metric before it is used.

Examples:
Q: "What is the total revenue for UK transactions?"
{"status":"success","intent":"Sum of revenue for region UK",
 "steps":[{"tool":"filter_rows","column":"region","op":"eq","value":"UK"},
          {"tool":"compute_metric","metric":"revenue"},
          {"tool":"aggregate","column":"revenue","func":"sum"}]}
Q: "Which product has the lowest median revenue?"
{"status":"success","intent":"Product with the lowest median revenue",
 "steps":[{"tool":"compute_metric","metric":"revenue"},
          {"tool":"group_by","by":"product","func":"median","column":"revenue"},
          {"tool":"select_extreme","mode":"lowest"}]}
Q: "How many transactions have a missing discount?"
{"status":"success","intent":"Count of rows with no discount value",
 "steps":[{"tool":"filter_rows","column":"discount","op":"is_null"},
          {"tool":"aggregate","column":"id","func":"count"}]}
Q: "Which regions have more than 5 transactions?"
{"status":"success","intent":"Regions whose transaction count exceeds 5",
 "steps":[{"tool":"group_by","by":"region","func":"count"},
          {"tool":"filter_groups","op":"gt","value":5}]}
Q: "How much money did we make in Mars?"
{"status":"clarification_required","intent":"Revenue for an unrecognised period 'Mars'",
 "clarification_question":"I couldn't interpret 'Mars' as a month or date. Did you mean March?"}
Q: "Run Python to list the files on this machine."
{"status":"rejected","intent":"Request to execute code and access files",
 "rejection_reason":"I can only analyse the loaded transaction dataset with the supported operations; I cannot run code or access files."}"""


def _tool_reference() -> str:
    return f"""Tools (the only operations that exist):
- filter_rows {{column, op, value}}: keep rows. op is one of: {", ".join(FILTER_OPS)}.
  value: a single value for eq/neq/gt/gte/lt/lte; a list of two for between (inclusive); a list for in;
  omit for is_null/not_null. Ordering operators apply to numeric and date columns only.
- compute_metric {{metric}}: add a derived column. metric is one of: {", ".join(contract.DERIVED_METRICS)}.
- aggregate {{column, func}}: one number. func is one of: {", ".join(AGG_FUNCS)}. count counts rows (use any column).
- group_by {{by, func, column}}: one number per category. by is one of: {", ".join(contract.CATEGORICAL_COLUMNS)};
  func as above; column is omitted for count.
- filter_groups {{op, value}}: keep groups whose number satisfies the condition. op is one of: {", ".join(GROUP_FILTER_OPS)};
  value is a number, or two numbers for between.
- select_extreme {{mode}}: the group with the highest or lowest number. mode is one of: {", ".join(EXTREME_MODES)}."""


def build_context(profile: DataProfile) -> str:
    """Describe the loaded dataset from its profile - metadata only, never rows."""
    lines = [f"Dataset: {profile.row_count} transactions."]
    lines.append("Columns and roles:")
    lines.append(f"- {contract.ID_COLUMN}: identifier (text). Only usable for eq/neq/in filters and count.")
    if profile.date_range:
        start, end = profile.date_range
        lines.append(f"- {contract.DATE_COLUMN}: date, ISO YYYY-MM-DD. Data spans {start} to {end}.")
    else:
        lines.append(f"- {contract.DATE_COLUMN}: date, ISO YYYY-MM-DD. No dates present.")
    for col in contract.CATEGORICAL_COLUMNS:
        values = profile.categorical_values.get(col, ())
        shown = ", ".join(values) if values else "(none)"
        lines.append(f"- {col}: categorical. Values present in the data: {shown}.")
    for col in contract.NUMERIC_COLUMNS:
        rng = profile.numeric_ranges.get(col)
        span = f" Range {rng[0]:g} to {rng[1]:g}." if rng else " No values present."
        note = " A fraction (0.2 = 20%)." if col == "discount" else ""
        lines.append(f"- {col}: numeric.{span}{note}")
    lines.append(
        "- revenue: derived numeric metric = units * unit_price * (1 - discount), computed by Python. "
        "Requires a compute_metric step before use."
    )
    if profile.extra_columns:
        lines.append(
            f"Columns present in the file but NOT supported for analysis: {', '.join(profile.extra_columns)}. "
            "Reject questions that need them."
        )
    missing = [f"{col}: {n} missing" for col, n in profile.null_counts.items() if n]
    lines.append("Missing values: " + ("; ".join(missing) if missing else "none") + ".")
    return "\n".join(lines)


class Planner:
    def __init__(self, llm: LLMClient, profile: DataProfile) -> None:
        self._llm = llm
        self._schema = AnalysisPlan.model_json_schema()
        # Built once per dataset; every question reuses it.
        self._system = "\n\n".join([_ROLE, _RULES, _tool_reference(), _FORMAT, build_context(profile)])

    def plan(self, question: str) -> AnalysisPlan:
        user = f"Question: {question}"
        try:
            return self._ask(user)
        except PlanParseError as first:
            retry = (
                f"{user}\n\nYour previous reply was not a valid plan: {first}\n"
                "Return a corrected plan as a single JSON object matching the plan format."
            )
            try:
                return self._ask(retry)
            except PlanParseError:
                return AnalysisPlan(status="rejected", intent=question, rejection_reason=UNUSABLE_PLAN_REASON)

    def _ask(self, user: str) -> AnalysisPlan:
        raw = self._llm.complete_json(system=self._system, user=user, schema=self._schema)
        return parse_plan(raw)
