"""Tests for db_app.services.cleaning — the audit behind the Data cleaning tab.

Hermetic: a small synthetic LOS-shaped CSV with known, injected problems (an
impossible glucose in a *test-period* row, an extreme neutrophil count in a train
row, and the 2013-01-01 stub day), loaded through the real `load_dataset`.
"""

import numpy as np
import pandas as pd
import pytest

import services.data as data_mod
from los_pred.features import FLAGS
from services.cleaning import build_cleaning_report

N = 400


def _fmt(ts: pd.Series) -> list[str]:
    return [f"{d.month}/{d.day}/{d.year}" for d in ts]


@pytest.fixture
def report(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    vdate = pd.Series(pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 350, N), "D"))
    vdate.iloc[0] = pd.Timestamp("2013-01-01")      # the stub day -> merged into Dec
    vdate.iloc[1] = pd.Timestamp("2012-12-15")      # a test-period row ...
    vdate.iloc[2] = pd.Timestamp("2012-03-01")      # ... and a train row
    los = rng.integers(1, 10, N)

    df = pd.DataFrame({
        "eid": np.arange(1, N + 1),
        "vdate": _fmt(vdate),
        "rcount": rng.choice(["0", "1", "5+"], N),
        "gender": rng.choice(["M", "F"], N),
        **{f: rng.integers(0, 2, N) for f in FLAGS},
        "hematocrit": rng.normal(12, 2, N),
        "neutrophils": rng.normal(10, 3, N).clip(0.1),
        "sodium": rng.normal(138, 3, N),
        "glucose": rng.normal(140, 20, N),
        "bloodureanitro": rng.normal(12, 3, N).clip(1),
        "creatinine": rng.normal(1.1, 0.2, N),
        "bmi": rng.normal(30, 2, N),
        "pulse": rng.integers(55, 95, N),
        "respiration": 6.5,
        "secondarydiagnosisnonicd9": rng.integers(0, 6, N),
        "discharged": _fmt(vdate + pd.to_timedelta(los, "D")),
        "facid": rng.choice(["A", "B", "C"], N),
        "lengthofstay": los,
    })
    df.loc[1, "glucose"] = -1.0          # impossible value, in the test period
    df.loc[2, "neutrophils"] = 200.0     # extreme outlier, in the train period
    path = tmp_path / "los.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(data_mod, "DATA_CSV", path)
    data_mod.load_dataset.cache_clear()
    yield build_cleaning_report(data_mod.load_dataset())
    data_mod.load_dataset.cache_clear()


def test_clean_input_checks(report):
    assert report.n_rows == N
    assert report.n_duplicates == 0 and report.n_missing == 0


def test_cleaning_never_drops_rows(report):
    assert report.n_removed == 0


def test_leakage_is_detected(report):
    assert report.leakage_match_pct == pytest.approx(100.0)   # discharged - visit == LOS


def test_stub_day_is_counted(report):
    assert report.n_stub_rows == 1


def test_impossible_glucose_replaced_by_training_median(report):
    row = report.effect.set_index("Variable").loc["glucose"]
    assert report.n_glucose_fixed == 1
    assert row["Raw min"] == pytest.approx(-1.0)
    assert row["Cleaned min"] > 0


def test_outliers_are_capped_at_the_learned_training_cap(report):
    row = report.effect.set_index("Variable").loc["neutrophils"]
    assert report.n_neutrophils_capped >= 1
    assert row["Raw max"] == pytest.approx(200.0)
    assert row["Cleaned max"] == pytest.approx(report.params["neutrophils_cap"])


def test_params_are_learned_from_the_training_split_only(report):
    # the impossible glucose sits in the test period, so the train median is untouched by it
    assert report.params["glucose_median"] > 100
    assert report.params["facid_categories"] == ["A", "B", "C"]


def test_split_partitions_all_rows_by_time(report):
    s = report.split.set_index("split")
    assert s["rows"].sum() == N
    assert s["share"].sum() == pytest.approx(1.0)
    assert s.loc["train", "last"] == "2012-10"
    assert s.loc["test", "first"] == "2012-11"


def test_final_features_exclude_leaky_and_dropped_columns(report):
    for col in ("eid", "vdate", "discharged", "respiration", "secondarydiagnosisnonicd9", "lengthofstay"):
        assert col not in report.features
    assert {"isMale", "comorbidity_count", "fac_other"} <= set(report.features)
