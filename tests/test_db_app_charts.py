"""Tests for the pre-computed length-of-stay counts and the SVG histogram drawn from them."""

import re

import numpy as np
import pandas as pd
import pytest

from services.charts import histogram_html, histogram_svg, nice_step
from services.summary import build_summary_tables, los_counts, los_from_tables
from services.summary_store import build_meta, load_summary_tables, save_summary_tables, summary_is_stale

DF = pd.DataFrame({
    "vdate": pd.to_datetime(["2012-01-05", "2012-03-01", "2012-11-20", "2012-12-30", "2012-02-02"]),
    "lengthofstay": [1, 2, 2, 6, 6],
    "bmi": [20.0, 30.0, 25.0, 40.0, 22.0],
    "gender": ["F", "F", "M", "M", "F"],
    "rcount": ["0", "5+", "0", "1", "0"],
    "asthma": [1, 0, 0, 0, 0],
    "facid": ["A", "A", "B", "B", "B"],
})


# --- the day counts --------------------------------------------------------
def test_counts_per_day_include_empty_days():
    out = los_counts(DF)
    assert list(out.columns) == ["Days", "n"]
    assert list(out["Days"]) == [1, 2, 3, 4, 5, 6]
    assert list(out["n"]) == [1, 2, 0, 0, 0, 2]
    assert out["n"].sum() == len(DF)


def test_saved_counts_equal_direct_counts(tmp_path):
    csv = tmp_path / "los.csv"
    DF.to_csv(csv, index=False)
    tables = build_summary_tables(DF, ["lengthofstay"], ["gender"], ["none", "facid"])
    save_summary_tables(tables, build_meta(DF, csv), tmp_path / "out")
    try:
        pd.testing.assert_frame_equal(los_from_tables(load_summary_tables(tmp_path / "out")), los_counts(DF))
    finally:
        load_summary_tables.cache_clear()
        summary_is_stale.cache_clear()


def test_histogram_does_not_depend_on_the_grouping():
    # one whole-population table, however many groupings the other tables have
    t = build_summary_tables(DF, ["lengthofstay"], ["gender"], ["none", "facid", "gender"])
    assert list(t.los.columns) == ["Days", "n"] and t.los["n"].sum() == len(DF)


# --- the chart ---------------------------------------------------------------
def test_nice_step_is_a_round_number():
    assert nice_step(0) == 1
    assert nice_step(17_979) == 5_000
    assert nice_step(1_900) == 500
    assert nice_step(3) == 1


def test_one_bar_per_nonzero_day_and_a_tooltip_for_every_day():
    svg = histogram_svg([1, 2, 3, 4], [5, 0, 3, 1])
    assert svg.count('class="hist-bar"') == 3                 # the empty day draws no bar
    assert svg.count("<title>") == 4                          # ...but still has a tooltip
    assert "1 day: 5 (55.6%)" in svg
    assert "2 days: 0 (0.0%)" in svg


def test_only_the_peak_is_labelled():
    svg = histogram_svg([1, 2, 3], [10, 40, 20])
    assert svg.count('class="hist-value"') == 1 and ">40<" in svg


def test_all_zero_counts_do_not_crash():
    svg = histogram_svg([1, 2], [0, 0])
    assert 'class="hist-bar"' not in svg and "NaN" not in svg


def test_large_counts_use_compact_ticks_small_use_full():
    assert "20K" in histogram_svg([1, 2], [17_979, 100])
    assert ">1,000<" in histogram_svg([1, 2], [900, 100])


def test_many_days_thin_out_the_x_labels():
    days = list(range(1, 41))
    svg = histogram_svg(days, [1] * 40)
    assert len(re.findall(r'class="hist-tick" x="[\d.]+" y="\d+" text-anchor="middle"', svg)) < 40


def test_histogram_html_has_one_panel_total_and_a_transposed_table_view():
    html = histogram_html(los_counts(DF))
    assert html.count("<svg") == 1
    assert "n = 5" in html and "View as table" in html
    # transposed: a header cell per day, then a single row of counts
    assert [int(d) for d in re.findall(r"<th class='text-end'>(\d+)</th>", html)] == [1, 2, 3, 4, 5, 6]
    assert html.count("<tbody><tr>") == 1 and html.count("<tr>") == 2          # header row + one body row
    row = re.search(r"<tbody>(.*?)</tbody>", html).group(1)
    assert re.findall(r"<td class='text-end'>([\d,]+)</td>", row) == ["1", "2", "0", "0", "0", "2"]
