"""The semantic contract of the transaction domain.

This is the one place that says what a "transaction dataset" *is*: which
columns must exist and what role each plays. It is deliberately the only
schema knowledge in the codebase.

Two things are kept apart on purpose:

* the CONTRACT (here): column names and roles — fixed, because the assignment
  defines the domain;
* the PROFILE (`data_profile.py`): the values those columns actually hold —
  discovered from the loaded file at runtime, never written down in code.

So the categorical values, the date range and the row count are *not* here.
If a loaded file has different regions or five hundred rows, nothing in this
module changes.
"""

ID_COLUMN = "id"
DATE_COLUMN = "date"
CATEGORICAL_COLUMNS: tuple[str, ...] = ("region", "product")
NUMERIC_COLUMNS: tuple[str, ...] = ("units", "unit_price", "discount")

REQUIRED_COLUMNS: tuple[str, ...] = (ID_COLUMN, DATE_COLUMN, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)

# The supplied file mixes transaction rows with evaluation questions. A
# non-empty value in this column marks a row as a question, not a transaction.
# The column itself is optional: a file without it is all transactions.
QUESTION_COLUMN = "question"

# Metrics that do not exist as columns but can be computed deterministically
# from ones that do. The formula lives in the compute tool; here we only record
# which inputs each metric needs so the profile can say whether it is available.
DERIVED_METRICS: dict[str, tuple[str, ...]] = {
    "revenue": ("units", "unit_price", "discount"),
}

# How each formula reads, for the "operations performed" explanation. The
# arithmetic itself lives in `tools.compute_metric`; this is only the wording.
METRIC_FORMULAS: dict[str, str] = {
    "revenue": "units * unit_price * (1 - discount)",
}
