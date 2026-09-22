"""Loader behaviour: path -> ActiveDataset.

The loader is the only component that ever sees a filesystem path.
"""

import numpy as np
import pandas as pd
import pytest

from app.data_loader import ActiveDataset, DatasetLoadError, load_dataset


# --- row discrimination -------------------------------------------------------


def test_splits_transaction_and_question_rows(mixed_csv):
    ds = load_dataset(mixed_csv)
    assert isinstance(ds, ActiveDataset)
    assert len(ds.transactions) == 3
    assert ds.questions == ["What is the total revenue?", "How many Beta transactions?"]


def test_question_column_is_not_part_of_transactions(mixed_csv):
    ds = load_dataset(mixed_csv)
    assert "question" not in ds.transactions.columns


def test_row_with_missing_fields_is_still_a_transaction(write_csv):
    # blank region AND blank discount, but question is blank -> transaction
    path = write_csv(["T1,2026-01-03,,Alpha,10,100,,"])
    ds = load_dataset(path)
    assert len(ds.transactions) == 1
    assert ds.questions == []


def test_whitespace_only_question_counts_as_empty(write_csv):
    path = write_csv(["T1,2026-01-03,UK,Alpha,10,100,0.1,   "])
    ds = load_dataset(path)
    assert len(ds.transactions) == 1
    assert ds.questions == []


def test_question_column_absent_means_all_rows_are_transactions(write_csv):
    path = write_csv(
        ["T1,2026-01-03,UK,Alpha,10,100,0.1", "T2,2026-01-04,DE,Beta,5,200,0.0"],
        header="id,date,region,product,units,unit_price,discount",
    )
    ds = load_dataset(path)
    assert len(ds.transactions) == 2
    assert ds.questions == []


# --- type parsing --------------------------------------------------------------


def test_types_are_parsed(mixed_csv):
    tx = load_dataset(mixed_csv).transactions
    assert pd.api.types.is_datetime64_any_dtype(tx["date"])
    for col in ("units", "unit_price", "discount"):
        assert pd.api.types.is_float_dtype(tx[col]), col
    assert tx["units"].tolist() == [10.0, 5.0, 8.0]


# --- missing vs malformed values ----------------------------------------------


def test_missing_values_are_kept_as_nan_not_dropped(write_csv):
    path = write_csv(
        [
            "T1,2026-01-03,UK,Alpha,10,100,,",   # discount missing
            "T2,,DE,Beta,5,200,0.0,",            # date missing
        ]
    )
    ds = load_dataset(path)
    assert len(ds.transactions) == 2
    assert np.isnan(ds.transactions.loc[0, "discount"])
    assert pd.isna(ds.transactions.loc[1, "date"])
    assert ds.parse_errors == {}


def test_malformed_values_are_coerced_and_counted_separately(write_csv):
    path = write_csv(
        [
            "T1,Jan 3rd,UK,Alpha,ten,100,0.1,",   # bad date, bad units
            "T2,2026-01-04,DE,Beta,5,200,,",      # missing discount (not an error)
        ]
    )
    ds = load_dataset(path)
    assert len(ds.transactions) == 2
    assert pd.isna(ds.transactions.loc[0, "date"])
    assert np.isnan(ds.transactions.loc[0, "units"])
    assert ds.parse_errors == {"date": 1, "units": 1}


# --- fatal errors --------------------------------------------------------------


def test_missing_required_column_is_fatal(write_csv):
    path = write_csv(["T1,2026-01-03,UK,10,100,0.1"], header="id,date,region,units,unit_price,discount")
    with pytest.raises(DatasetLoadError, match="product"):
        load_dataset(path)


def test_missing_file_is_fatal(tmp_path):
    with pytest.raises(DatasetLoadError):
        load_dataset(tmp_path / "nope.csv")


def test_non_csv_extension_is_fatal(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_text("id,date\n")
    with pytest.raises(DatasetLoadError, match="csv"):
        load_dataset(path)


def test_zero_transaction_rows_is_fatal(write_csv):
    path = write_csv(['Q1,,,,,,,"Only a question"'])
    with pytest.raises(DatasetLoadError, match="no transaction rows"):
        load_dataset(path)


def test_empty_file_is_fatal(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetLoadError):
        load_dataset(path)


# --- extra columns -------------------------------------------------------------


def test_extra_column_is_kept_in_transactions(write_csv):
    path = write_csv(
        ["T1,2026-01-03,UK,Alpha,10,100,0.1,north,"],
        header="id,date,region,product,units,unit_price,discount,zone,question",
    )
    ds = load_dataset(path)
    assert "zone" in ds.transactions.columns
    assert ds.transactions.loc[0, "zone"] == "north"


# --- the supplied file (structure only) ---------------------------------------


def test_supplied_dataset_loads():
    ds = load_dataset("data/project_4.csv")
    assert ds.source_name == "project_4.csv"
    assert len(ds.transactions) > 0
    assert len(ds.questions) > 0
    assert ds.parse_errors == {}
    assert ds.transactions.isna().sum().sum() == 0  # regression: no leakage from question rows


def test_missing_categorical_is_na_not_empty_string(write_csv):
    # One definition of "missing" across all columns: pandas NA, never "".
    path = write_csv(["T1,2026-01-03,,Alpha,10,100,0.1,"])
    tx = load_dataset(path).transactions
    assert pd.isna(tx.loc[0, "region"])
    assert tx.loc[0, "id"] == "T1"
