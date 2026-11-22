"""Tests for db_app.services.summary — the descriptive tables behind the Group summary tab.

Hermetic: a tiny hand-built frame, no CSV or Shiny needed (db_app is put on the
path via [tool.pytest.ini_options] in pyproject.toml).
"""

import numpy as np
import pandas as pd
import pytest

from services.summary import categorical_long, categorical_summary, continuous_summary

# 4 rows, hand-checkable numbers.
DF = pd.DataFrame({
    "lengthofstay": [1, 2, 3, 6],
    "bmi": [20.0, 30.0, np.nan, 40.0],
    "gender": ["F", "F", "M", "M"],
    "rcount": ["0", "5+", "0", "1"],
    "asthma": [1, 0, 0, 0],
    "facid": ["A", "A", "B", "B"],
})


def test_continuous_overall_stats():
    out = continuous_summary(DF, ["lengthofstay"])
    row = out.iloc[0]
    assert list(out.columns) == ["Variable", "N", "Mean", "SD", "Median", "Q1", "Q3"]
    assert row["N"] == 4
    assert row["Mean"] == pytest.approx(3.0)
    assert row["Median"] == pytest.approx(2.5)
    assert row["Q1"] == pytest.approx(1.75)
    assert row["Q3"] == pytest.approx(3.75)


def test_continuous_ignores_missing_values():
    row = continuous_summary(DF, ["bmi"]).iloc[0]
    assert row["N"] == 3                      # NaN not counted
    assert row["Mean"] == pytest.approx(30.0)


def test_continuous_by_group_has_one_row_per_group_and_variable():
    out = continuous_summary(DF, ["lengthofstay", "bmi"], by="facid")
    assert list(out["Group"]) == ["A", "A", "B", "B"]
    a_los = out[(out["Group"] == "A") & (out["Variable"] == "Length of stay")].iloc[0]
    assert a_los["Mean"] == pytest.approx(1.5)


def test_categorical_proportions_sum_to_100_per_variable():
    long = categorical_long(DF, ["gender", "rcount"])
    for var, grp in long.groupby("Variable"):
        assert grp["pct"].sum() == pytest.approx(100.0), var


def test_binary_flag_reports_only_the_yes_row():
    long = categorical_long(DF, ["asthma"])
    assert list(long["Level"]) == ["Yes"]
    assert long.iloc[0]["n"] == 1 and long.iloc[0]["pct"] == pytest.approx(25.0)


def test_rcount_levels_are_ordered_not_alphabetical():
    long = categorical_long(DF, ["rcount"])
    assert list(long["Level"]) == ["0", "1", "2", "3", "4", "5+"]   # incl. empty levels


def test_categorical_wide_has_a_column_per_group_and_formats_cells():
    wide = categorical_summary(DF, ["gender"], by="facid")
    assert list(wide.columns) == ["Variable", "Level", "A", "B"]
    f_row = wide[wide["Level"] == "F"].iloc[0]
    assert f_row["A"] == "2 (100.0%)" and f_row["B"] == "0 (0.0%)"


def test_categorical_wide_overall_when_not_grouped():
    wide = categorical_summary(DF, ["gender"])
    assert list(wide.columns) == ["Variable", "Level", "Overall"]
    assert list(wide["Overall"]) == ["2 (50.0%)", "2 (50.0%)"]
