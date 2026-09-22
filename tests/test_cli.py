"""CLI: load a dataset once, answer many questions against it.

Streams and the LLM client are injected, so the whole REPL runs in-process
with no terminal and no network.
"""

import io
import json

import pytest

from app.cli import run
from app.llm.base import LLMError
from app.llm.fake import FakeLLMClient


@pytest.fixture
def data_file(write_csv):
    return write_csv(
        [
            "T1,2026-01-03,UK,Alpha,10,100,0.10,",
            "T2,2026-01-20,DE,Beta,5,200,0.00,",
            "T3,2026-02-05,UK,Beta,8,200,0.20,",
            'Q1,,,,,,,"What is the total revenue for UK transactions?"',
            'Q2,,,,,,,"How many Beta transactions are there?"',
        ]
    )


def plan(intent, *steps):
    return json.dumps({"status": "success", "intent": intent, "steps": list(steps)})


COUNT_UK = plan(
    "Number of transactions in region UK",
    {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
    {"tool": "aggregate", "column": "id", "func": "count"},
)
SUM_UNITS = plan("Total units across all transactions", {"tool": "aggregate", "column": "units", "func": "sum"})


def cli(data_file, lines, *replies):
    out = io.StringIO()
    code = run(
        ["--data", str(data_file)],
        stdin=io.StringIO("\n".join(lines) + "\n"),
        stdout=out,
        llm=FakeLLMClient(responses=list(replies)),
    )
    return code, out.getvalue()


# --- startup -------------------------------------------------------------------------


def test_startup_summary_describes_the_loaded_dataset(data_file):
    code, out = cli(data_file, ["/exit"])
    assert code == 0
    assert "3 transactions" in out
    assert "region" in out and "product" in out
    assert "2026-01-03" in out and "2026-02-05" in out
    assert "Ready" in out


def test_startup_reports_the_embedded_question_corpus(data_file):
    _, out = cli(data_file, ["/exit"])
    assert "2 questions" in out


def test_missing_file_exits_with_code_one(tmp_path):
    out = io.StringIO()
    code = run(["--data", str(tmp_path / "nope.csv")], stdin=io.StringIO(""), stdout=out,
               llm=FakeLLMClient(responses=[]))
    assert code == 1
    assert "not found" in out.getvalue().lower() or "File not found" in out.getvalue()


def test_invalid_dataset_exits_with_code_one(write_csv):
    bad = write_csv(["T1,2026-01-03,UK,10,100,0.1"], header="id,date,region,units,unit_price,discount")
    out = io.StringIO()
    code = run(["--data", str(bad)], stdin=io.StringIO(""), stdout=out, llm=FakeLLMClient(responses=[]))
    assert code == 1
    assert "product" in out.getvalue()


def test_missing_api_key_is_reported_clearly(data_file, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = io.StringIO()
    # No llm injected: the CLI must build one from settings and fail clearly.
    code = run(["--data", str(data_file), "--no-dotenv"], stdin=io.StringIO(""), stdout=out)
    assert code == 1
    assert "GEMINI_API_KEY" in out.getvalue()


# --- asking questions -------------------------------------------------------------------


def test_answers_a_question(data_file):
    _, out = cli(data_file, ["How many UK transactions?", "/exit"], COUNT_UK)
    assert "Answer:" in out
    assert "Number of transactions in region UK: 2" in out
    assert "Operations performed:" in out


def test_many_questions_run_against_one_active_dataset(data_file):
    _, out = cli(data_file, ["q1", "q2", "/exit"], COUNT_UK, SUM_UNITS)
    assert "Number of transactions in region UK: 2" in out
    assert "Total units across all transactions: 23.00" in out


def test_blank_lines_are_ignored(data_file):
    # No scripted replies: any LLM call would raise, so a blank line must not make one.
    code, out = cli(data_file, ["", "   ", "/exit"])
    assert code == 0


def test_provider_failure_does_not_end_the_session(data_file):
    _, out = cli(data_file, ["q1", "q2", "/exit"], LLMError("quota exceeded"), COUNT_UK)
    assert "Status: Error" in out
    assert "quota exceeded" in out
    assert "Number of transactions in region UK: 2" in out  # the session carried on


# --- commands ---------------------------------------------------------------------------


def test_help_lists_the_commands(data_file):
    _, out = cli(data_file, ["/help", "/exit"])
    for command in ("/help", "/profile", "/questions", "/load", "/exit"):
        assert command in out


def test_profile_command_shows_discovered_values(data_file):
    _, out = cli(data_file, ["/profile", "/exit"])
    assert "DE" in out and "UK" in out
    assert "Alpha" in out and "Beta" in out


def test_questions_command_lists_the_embedded_corpus(data_file):
    _, out = cli(data_file, ["/questions", "/exit"])
    assert "What is the total revenue for UK transactions?" in out
    assert "How many Beta transactions are there?" in out


def test_load_replaces_the_active_dataset(data_file, write_csv):
    other = write_csv(["T9,2026-05-05,UK,Alpha,1,100,0.00,"], name="v2.csv")
    _, out = cli(data_file, ["q1", f"/load {other}", "q2", "/exit"], SUM_UNITS, SUM_UNITS)
    assert "Total units across all transactions: 23.00" in out  # first dataset
    assert "Total units across all transactions: 1.00" in out   # after the swap
    assert "v2.csv" in out


def test_failed_load_keeps_the_previous_dataset(data_file, tmp_path):
    _, out = cli(data_file, [f"/load {tmp_path / 'missing.csv'}", "q", "/exit"], COUNT_UK)
    assert "could not be loaded" in out.lower() or "not found" in out.lower()
    assert "Number of transactions in region UK: 2" in out  # still the original dataset


def test_load_without_a_path_explains_usage(data_file):
    _, out = cli(data_file, ["/load", "/exit"])
    assert "/load" in out and "path" in out.lower()


def test_unknown_command_does_not_reach_the_llm(data_file):
    code, out = cli(data_file, ["/frobnicate", "/exit"])  # no replies scripted
    assert code == 0
    assert "Unknown command" in out


def test_eof_ends_the_session_cleanly(data_file):
    out = io.StringIO()
    code = run(["--data", str(data_file)], stdin=io.StringIO(""), stdout=out,
               llm=FakeLLMClient(responses=[]))
    assert code == 0


# --- presentation ---------------------------------------------------------------------------


def test_output_is_ascii_safe(data_file):
    _, out = cli(data_file, ["/help", "/profile", "q", "/exit"], COUNT_UK)
    out.encode("ascii")


def test_no_file_path_is_echoed_into_the_session_output(data_file):
    # The banner names the file; nothing else should leak a full path.
    _, out = cli(data_file, ["q", "/exit"], COUNT_UK)
    assert str(data_file.parent) not in out.split("Ready")[1]
