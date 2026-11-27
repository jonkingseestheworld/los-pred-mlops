"""Tests for db_app.services.summary — the descriptive tables behind the Group summary tab.

Hermetic: a tiny hand-built frame, no CSV or Shiny needed (db_app is put on the
path via [tool.pytest.ini_options] in pyproject.toml).
"""

import numpy as np
import pandas as pd
import pytest

from services.summary import categorical_long, categorical_summary, categorical_table1, continuous_display, continuous_summary

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
    a_los = out[(out["Group"] == "A") & (out["Variable"] == "Length of stay (days)")].iloc[0]
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


def test_continuous_display_layout_and_formatting():
    out = continuous_display(continuous_summary(DF, ["lengthofstay"]))
    assert list(out.columns) == ["Variable", "Mean (SD)", "Median (Q1, Q3)"]   # N column is switched off
    row = out.iloc[0]
    assert row["Mean (SD)"] == "3.00 (2.16)"
    assert row["Median (Q1, Q3)"] == "2.50 (1.75, 3.75)"


def test_continuous_display_keeps_the_group_column_and_handles_missing_sd():
    one = pd.DataFrame({"lengthofstay": [5], "facid": ["A"]})        # a single value has no SD
    out = continuous_display(continuous_summary(one, ["lengthofstay"], by="facid"))
    assert list(out.columns) == ["Group", "Variable", "Mean (SD)", "Median (Q1, Q3)"]
    assert out.iloc[0]["Mean (SD)"] == "5.00 (–)"


def test_table1_layout_variable_row_then_indented_levels_and_single_row_flags():
    t = categorical_table1(categorical_summary(DF, ["gender", "asthma"]))
    assert list(t.columns) == ["Variable", "kind", "Overall"]
    assert list(t["Variable"]) == ["Gender", "F", "M", "Asthma"]
    assert list(t["kind"]) == ["variable", "level", "level", "variable"]
    assert list(t["Overall"]) == ["", "2 (50.0%)", "2 (50.0%)", "1 (25.0%)"]   # header row has no count; flag keeps its own


def test_table1_has_a_column_per_group():
    t = categorical_table1(categorical_summary(DF, ["gender"], by="facid"))
    assert list(t.columns) == ["Variable", "kind", "A", "B"]
    assert t.iloc[1]["A"] == "2 (100.0%)" and t.iloc[1]["B"] == "0 (0.0%)"      # level F


def test_train_comes_before_test_in_the_split_grouping():
    df = DF.assign(split=["test", "train", "train", "test"])
    assert list(continuous_summary(df, ["lengthofstay"], by="split")["Group"]) == ["train", "test"]
    wide = categorical_summary(df, ["gender"], by="split")
    assert list(wide.columns) == ["Variable", "Level", "train", "test"]
