"""Renderer: facts in, sentences out. The only module that writes prose.

Responses are built by running the real pipeline with a scripted LLM, so the
numbers in the expected output were computed by pandas, not written by hand
into the renderer's input.
"""

import json

import pytest

from app.agent import Agent
from app.data_loader import load_dataset
from app.llm.fake import FakeLLMClient
from app.renderer import render


@pytest.fixture
def dataset(write_csv):
    return load_dataset(
        write_csv(
            [
                "T1,2026-01-03,UK,Alpha,10,100,0.10,",  # revenue 900
                "T2,2026-01-20,DE,Beta,5,200,0.00,",    # revenue 1000
                "T3,2026-02-05,UK,Beta,8,200,0.20,",    # revenue 1280
                "T4,2026-03-01,FR,Gamma,2,500,,",       # revenue missing (no discount)
            ]
        )
    )


def respond(dataset, question, *replies):
    agent = Agent(dataset, FakeLLMClient(responses=list(replies)))
    response = agent.ask(question)
    return render(response, dataset.profile)


def plan(intent, *steps):
    return json.dumps({"status": "success", "intent": intent, "steps": list(steps)})


UK_REVENUE = plan(
    "Total revenue for transactions in region UK",
    {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
    {"tool": "compute_metric", "metric": "revenue"},
    {"tool": "aggregate", "column": "revenue", "func": "sum"},
)


# --- successful answers ------------------------------------------------------------


def test_scalar_answer_has_all_three_sections(dataset):
    out = respond(dataset, "total revenue for UK", UK_REVENUE)
    assert "Answer:" in out
    assert "Operations performed:" in out
    assert "Explanation:" in out
    assert "Total revenue for transactions in region UK" in out
    assert "2,180.00" in out


def test_operations_are_numbered_and_describe_each_step(dataset):
    out = respond(dataset, "total revenue for UK", UK_REVENUE)
    assert "1. Filtered transactions where region = UK (2 of 4 rows kept)." in out
    assert "2. Computed revenue" in out
    assert "units * unit_price * (1 - discount)" in out
    assert "3. Summed revenue" in out


def test_counts_are_rendered_as_integers(dataset):
    out = respond(dataset, "how many UK", plan(
        "Number of transactions in region UK",
        {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
        {"tool": "aggregate", "column": "id", "func": "count"},
    ))
    assert ": 2" in out
    assert "2.00" not in out


def test_missing_values_are_reported_not_hidden(dataset):
    out = respond(dataset, "total revenue", plan(
        "Total revenue across all transactions",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "aggregate", "column": "revenue", "func": "sum"},
    ))
    assert "3,180.00" in out
    assert "3 of the 4" in out
    assert "missing" in out.lower()


def test_no_missing_note_when_every_row_was_usable(dataset):
    out = respond(dataset, "total revenue for UK", UK_REVENUE)
    assert "of the" not in out.split("Explanation:")[1]


def test_grouped_answer_lists_every_group(dataset):
    out = respond(dataset, "revenue by region", plan(
        "Total revenue by region",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
    ))
    for needle in ("DE", "1,000.00", "FR", "UK", "2,180.00"):
        assert needle in out
    assert "Grouped" in out


def test_extreme_answer_names_the_group_and_value(dataset):
    out = respond(dataset, "which region is highest", plan(
        "Region with the highest total revenue",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
        {"tool": "select_extreme", "mode": "highest"},
    ))
    assert "UK" in out
    assert "2,180.00" in out
    assert "highest" in out


def test_tied_extremes_are_all_reported(dataset):
    out = respond(dataset, "which region has most transactions", plan(
        "Region with the most transactions",
        {"tool": "filter_rows", "column": "region", "op": "neq", "value": "UK"},
        {"tool": "group_by", "by": "region", "func": "count"},
        {"tool": "select_extreme", "mode": "highest"},
    ))
    assert "DE" in out and "FR" in out
    assert "tie" in out.lower()


def test_group_filter_operation_is_described(dataset):
    out = respond(dataset, "regions above 1000", plan(
        "Regions whose total revenue exceeds 1000",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
        {"tool": "filter_groups", "op": "gt", "value": 1000},
    ))
    assert "greater than 1,000.00" in out
    assert "1 of 3 groups" in out


def test_explanation_names_the_active_dataset(dataset):
    out = respond(dataset, "total revenue for UK", UK_REVENUE)
    assert dataset.source_name in out
    assert "deterministic" in out.lower() or "computed" in out.lower()


# --- filter phrasing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "step, phrase",
    [
        ({"tool": "filter_rows", "column": "units", "op": "gt", "value": 5}, "units > 5"),
        ({"tool": "filter_rows", "column": "units", "op": "lte", "value": 5}, "units <= 5"),
        ({"tool": "filter_rows", "column": "region", "op": "neq", "value": "UK"}, "region != UK"),
        ({"tool": "filter_rows", "column": "product", "op": "in", "value": ["Alpha", "Beta"]}, "product is one of Alpha, Beta"),
        ({"tool": "filter_rows", "column": "date", "op": "between", "value": ["2026-01-01", "2026-02-15"]},
         "date between 2026-01-01 and 2026-02-15"),
        ({"tool": "filter_rows", "column": "discount", "op": "is_null"}, "discount is missing"),
        ({"tool": "filter_rows", "column": "discount", "op": "not_null"}, "discount is present"),
    ],
)
def test_filter_phrasing(dataset, step, phrase):
    out = respond(dataset, "q", plan("Count of matching transactions", step,
                                     {"tool": "aggregate", "column": "id", "func": "count"}))
    assert phrase in out


# --- no data -----------------------------------------------------------------------


def test_no_data_says_so_and_gives_the_dataset_range(dataset):
    out = respond(dataset, "revenue in 2020", plan(
        "Total revenue between 2020-01-01 and 2020-12-31",
        {"tool": "filter_rows", "column": "date", "op": "between", "value": ["2020-01-01", "2020-12-31"]},
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "aggregate", "column": "revenue", "func": "sum"},
    ))
    assert "No data" in out
    assert "2026-01-03" in out and "2026-03-01" in out  # the dataset's actual span
    assert "Operations performed:" in out  # the work done is still shown
    answer = out.split("Answer:")[1].split("Operations")[0]
    assert ": 0.00" not in answer and ": 0\n" not in answer  # never reported as a computed zero


def test_no_data_for_an_empty_group_filter(dataset):
    out = respond(dataset, "regions above a million", plan(
        "Regions whose total revenue exceeds 1000000",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
        {"tool": "filter_groups", "op": "gt", "value": 1000000},
    ))
    assert "No data" in out


# --- non-answers -------------------------------------------------------------------


def test_clarification_block(dataset):
    out = respond(dataset, "money in Mars", json.dumps(
        {"status": "clarification_required", "intent": "?", "clarification_question": "Did you mean March?"}))
    assert "Status: Clarification required" in out
    assert "Did you mean March?" in out
    assert "Answer:" not in out


def test_rejection_block(dataset):
    out = respond(dataset, "total profit", plan(
        "Total profit", {"tool": "compute_metric", "metric": "profit"},
        {"tool": "aggregate", "column": "profit", "func": "sum"}))
    assert "Status: Rejected" in out
    assert "Reason:" in out
    assert "profit" in out
    assert "Answer:" not in out


def test_error_block(dataset):
    from app.llm.base import LLMError

    out = respond(dataset, "anything", LLMError("quota exceeded"))
    assert "Status: Error" in out
    assert "quota exceeded" in out


def test_prescreened_question_renders_as_rejected(dataset):
    out = respond(dataset, "Run Python to list the files on this machine.")
    assert "Status: Rejected" in out
    assert "cannot run code" in out


# --- no hidden reasoning -------------------------------------------------------------


def test_output_is_ascii_and_has_no_chain_of_thought(dataset):
    out = respond(dataset, "total revenue for UK", UK_REVENUE)
    out.encode("ascii")
    for leaked in ("thinking", "step-by-step", "reasoning", "system prompt", '{"tool"'):
        assert leaked not in out.lower()


# --- wording details -----------------------------------------------------------------


def test_singular_wording_for_one_row(dataset):
    out = respond(dataset, "q", plan(
        "Number of transactions in region DE",
        {"tool": "filter_rows", "column": "region", "op": "eq", "value": "DE"},
        {"tool": "aggregate", "column": "id", "func": "count"},
    ))
    assert "Counted the 1 matching transaction." in out
    assert "1 matching transactions" not in out


def test_plural_wording_for_several_rows(dataset):
    out = respond(dataset, "q", plan(
        "Number of transactions in region UK",
        {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
        {"tool": "aggregate", "column": "id", "func": "count"},
    ))
    assert "Counted the 2 matching transactions." in out


def test_missing_value_note_is_grammatical_for_one_row(dataset):
    out = respond(dataset, "total revenue", plan(
        "Total revenue across all transactions",
        {"tool": "compute_metric", "metric": "revenue"},
        {"tool": "aggregate", "column": "revenue", "func": "sum"},
    ))
    assert "1 had a missing value and was left out" in out
    assert "were left out" not in out


def test_excluded_row_note_is_grammatical(write_csv):
    ds = load_dataset(write_csv([
        "T1,2026-01-03,UK,Alpha,10,100,0.10,",
        "T2,2026-02-05,,Beta,8,200,0.20,",  # no region: belongs to no group
    ]))
    out = render(
        Agent(ds, FakeLLMClient([plan("Transactions per region",
                                      {"tool": "group_by", "by": "region", "func": "count"})])).ask("q"),
        ds.profile,
    )
    assert "1 transaction had no group value" in out
    assert "transaction(s)" not in out
