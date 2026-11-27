"""Tests for the pre-computed Cohort Summary tables and their version stamp.

Hermetic: a tiny hand-built frame and tmp files. The key property is that reading the
saved tables gives exactly what summarising the raw rows would.
"""

import json

import numpy as np
import pandas as pd
import pytest

from services.summary import (
    build_summary_tables,
    categorical_from_tables,
    categorical_summary,
    continuous_from_tables,
    continuous_summary,
)
from services.summary_store import (
    build_meta,
    data_fingerprint,
    load_summary_meta,
    load_summary_tables,
    save_summary_tables,
    summary_is_stale,
)

DF = pd.DataFrame({
    "vdate": pd.to_datetime(["2012-01-05", "2012-03-01", "2012-11-20", "2012-12-30"]),
    "lengthofstay": [1, 2, 3, 6],
    "bmi": [20.0, 30.0, np.nan, 40.0],
    "gender": ["F", "F", "M", "M"],
    "rcount": ["0", "5+", "0", "1"],
    "asthma": [1, 0, 0, 0],
    "facid": ["A", "A", "B", "B"],
})
CONT = ["lengthofstay", "bmi"]
CAT = ["gender", "rcount", "asthma", "facid"]
BYS = ["none", "facid", "gender"]


@pytest.fixture
def saved(tmp_path):
    """Tables built from DF, written to disk, and read back (as the app does)."""
    csv = tmp_path / "los.csv"
    DF.to_csv(csv, index=False)
    out = tmp_path / "summary"
    tables = build_summary_tables(DF, CONT, CAT, BYS)
    save_summary_tables(tables, build_meta(DF, csv), out)
    yield out, csv
    load_summary_tables.cache_clear()
    summary_is_stale.cache_clear()


@pytest.mark.parametrize("by", [None, "facid", "gender"])
@pytest.mark.parametrize("variables", [CONT, CONT[::-1]])
def test_continuous_from_saved_tables_equals_direct_computation(saved, by, variables):
    out, _ = saved
    got = continuous_from_tables(load_summary_tables(out), variables, by)
    want = continuous_summary(DF, variables, by)
    pd.testing.assert_frame_equal(got, want)


@pytest.mark.parametrize("by", [None, "facid", "gender"])
def test_categorical_from_saved_tables_equals_direct_computation(saved, by):
    out, _ = saved
    variables = [v for v in CAT if v != by]          # the tab never tabulates the grouping variable
    got = categorical_from_tables(load_summary_tables(out), variables, by)
    want = categorical_summary(DF, variables, by)
    pd.testing.assert_frame_equal(got, want)


def test_categorical_respects_the_order_variables_were_chosen_in(saved):
    out, _ = saved
    got = categorical_from_tables(load_summary_tables(out), ["rcount", "gender"], None)
    assert list(dict.fromkeys(got["Variable"])) == ["Readmission count", "Gender"]


def test_text_levels_survive_the_csv_round_trip(saved):
    out, _ = saved
    wide = categorical_from_tables(load_summary_tables(out), ["rcount"], None)
    assert list(wide["Level"]) == ["0", "1", "2", "3", "4", "5+"]


def test_a_variable_is_not_tabulated_against_itself(saved):
    out, _ = saved
    cat = load_summary_tables(out).categorical
    assert "Gender" not in set(cat[cat["By"] == "gender"]["Variable"])


def test_stamp_describes_the_data(saved):
    out, csv = saved
    meta = load_summary_meta(out)
    assert meta["n_rows"] == 4
    assert (meta["first_visit"], meta["last_visit"]) == ("2012-01-05", "2012-12-30")
    assert meta["data_sha256"] == data_fingerprint(csv)
    assert {"schema_version", "split_cutoff", "built_at", "git_commit"} <= set(meta)
    json.dumps(meta)                                  # plain JSON types only


def test_fingerprint_changes_when_the_data_changes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x\n1\n")
    b.write_text("x\n1\n")
    assert data_fingerprint(a) == data_fingerprint(b)
    b.write_text("x\n2\n")
    assert data_fingerprint(a) != data_fingerprint(b)


def test_stale_is_detected_when_the_source_changes(saved):
    out, csv = saved
    assert summary_is_stale(out, csv) is False
    summary_is_stale.cache_clear()
    csv.write_text(csv.read_text() + "2013-01-01,9,1.0,F,0,0,A\n")   # "more data accumulated"
    assert summary_is_stale(out, csv) is True


def test_staleness_unknown_when_raw_data_is_not_shipped(saved, tmp_path):
    out, _ = saved
    assert summary_is_stale(out, tmp_path / "missing.csv") is None


def test_missing_tables_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_summary.py"):
        load_summary_tables(tmp_path / "nope")
