"""Load a CSV into an `ActiveDataset`.

This module is the only place in the application that touches the filesystem
for data. It receives a path from the CLI — never from the LLM — and produces
a typed, transactions-only DataFrame plus the evaluation questions embedded in
the file.

Reading strategy: the CSV is read with every cell as a string first. That lets
us observe each raw value before any coercion, which is what makes it possible
to tell "the cell was empty" (missing) apart from "the cell held something that
is not a number/date" (malformed). Both end up as NaN/NaT in the frame — no row
is ever dropped for either — but only the second is counted in `parse_errors`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app import contract
from app.data_profile import DataProfile, build_profile


class DatasetLoadError(Exception):
    """The file cannot be used as a transaction dataset. Fatal by design."""


@dataclass(frozen=True)
class ActiveDataset:
    """The dataset a session is answering questions about.

    `transactions` holds only transaction rows, typed. `questions` is the
    evaluation corpus the file carried along (may be empty). `parse_errors`
    counts, per column, values that were present but could not be parsed.
    """

    transactions: pd.DataFrame
    questions: list[str]
    profile: DataProfile
    source_name: str
    parse_errors: dict[str, int] = field(default_factory=dict)


def load_dataset(path: str | Path) -> ActiveDataset:
    path = Path(path)
    raw = _read_raw(path)
    _require_columns(raw)
    transactions, questions = _split_rows(raw)
    if transactions.empty:
        raise DatasetLoadError(f"{path.name} contains no transaction rows.")

    transactions, parse_errors = _coerce_types(transactions)
    profile = build_profile(transactions, parse_errors=parse_errors, source_name=path.name)
    return ActiveDataset(
        transactions=transactions,
        questions=questions,
        profile=profile,
        source_name=path.name,
        parse_errors=parse_errors,
    )


# --- steps -------------------------------------------------------------------


def _read_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DatasetLoadError(f"File not found: {path}")
    if path.suffix.lower() != ".csv":
        raise DatasetLoadError(f"Expected a .csv file, got '{path.suffix}'.")
    try:
        # dtype=str + keep_default_na=False: every cell arrives as the literal
        # text in the file ("" for an empty cell). Coercion happens later, on
        # our terms, so we can count what failed.
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError(f"{path.name} is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetLoadError(f"{path.name} could not be parsed as CSV: {exc}") from exc
    return raw.apply(lambda col: col.str.strip())


def _require_columns(raw: pd.DataFrame) -> None:
    missing = [c for c in contract.REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DatasetLoadError(f"Missing required column(s): {', '.join(missing)}")


def _split_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Separate rows on the `question` column alone.

    A row is a question iff that column is non-empty. Missing transaction
    fields never influence the split — a transaction with a blank discount is
    still a transaction.
    """
    if contract.QUESTION_COLUMN not in raw.columns:
        return raw.reset_index(drop=True), []

    is_question = raw[contract.QUESTION_COLUMN] != ""
    questions = raw.loc[is_question, contract.QUESTION_COLUMN].tolist()
    transactions = raw.loc[~is_question].drop(columns=[contract.QUESTION_COLUMN])
    return transactions.reset_index(drop=True), questions


def _coerce_types(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse dates and numerics; count present-but-unparseable values."""
    tx = tx.copy()
    parse_errors: dict[str, int] = {}

    def record(column: str, raw: pd.Series, parsed: pd.Series) -> None:
        failed = int(((raw != "") & parsed.isna()).sum())
        if failed:
            parse_errors[column] = failed

    raw_date = tx[contract.DATE_COLUMN]
    # ISO 8601 only. Accepting "03/01/2026" would force a silent day/month
    # guess, which is exactly the kind of ambiguity this project must not hide.
    parsed_date = pd.to_datetime(raw_date, errors="coerce", format="ISO8601")
    record(contract.DATE_COLUMN, raw_date, parsed_date)
    tx[contract.DATE_COLUMN] = parsed_date

    for column in contract.NUMERIC_COLUMNS:
        raw_num = tx[column]
        parsed_num = pd.to_numeric(raw_num, errors="coerce").astype("float64")
        record(column, raw_num, parsed_num)
        tx[column] = parsed_num

    # Remaining columns stay text. An empty cell becomes NA so that "missing"
    # means one thing everywhere: `isna()`, whatever the column type.
    for column in tx.columns:
        if pd.api.types.is_string_dtype(tx[column]):
            tx[column] = tx[column].replace("", pd.NA)

    return tx, parse_errors
