"""Tests for los_pred.features — the feature-engineering transform.

Uses a small synthetic frame (not the real CSV, which is gitignored) so the
suite is self-contained and reproducible anywhere. Edge cases are baked into
the fixture: an rcount "5+", a negative glucose, an extreme neutrophils value,
an unseen test facility, and rows either side of the temporal cutoff.
"""

import pandas as pd
import pytest

from los_pred import features
from los_pred.features import (
    DROP,
    add_batch_month,
    fit_transformer,
    temporal_split,
    transform,
)


def _row(eid, vdate, facid, los, *, rcount="0", glucose=120.0,
         neutrophils=9.0, bloodureanitro=12.0, flags=0):
    """Build one raw admission row with sensible defaults."""
    row = {
        "eid": eid, "vdate": vdate, "rcount": rcount, "gender": "M" if eid % 2 else "F",
        "hematocrit": 12.0, "neutrophils": neutrophils, "sodium": 138.0,
        "glucose": glucose, "bloodureanitro": bloodureanitro, "creatinine": 1.0,
        "bmi": 30.0, "pulse": 70, "respiration": 6.5,
        "secondarydiagnosisnonicd9": 1, "discharged": vdate, "facid": facid,
        "lengthofstay": los,
    }
    for f in features.FLAGS:
        row[f] = 0
    # Turn on `flags` comorbidity flags to exercise comorbidity_count.
    for f in features.FLAGS[:flags]:
        row[f] = 1
    return row


@pytest.fixture
def raw():
    """Raw frame spanning two months, two train facilities (A, B) and an
    unseen test facility (Z), plus DQ edge cases."""
    rows = [
        # --- train months (<= 2012-10) ---
        _row(1, "9/3/2012", "A", 3, rcount="5+", flags=2),
        _row(2, "9/15/2012", "B", 5, glucose=-1.0),            # impossible glucose
        _row(3, "10/2/2012", "A", 4, neutrophils=999.0),       # extreme outlier
        _row(4, "10/20/2012", "B", 7, flags=11),
        # --- test months (> 2012-10) ---
        _row(5, "11/8/2012", "A", 2),
        _row(6, "12/24/2012", "Z", 6),                         # unseen facility
        _row(7, "1/1/2013", "A", 3),                           # stub -> merges into Dec
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def split(raw):
    df = add_batch_month(raw)
    return temporal_split(df, cutoff="2012-10")


@pytest.fixture
def fitted(split):
    train_df, test_df = split
    params = fit_transformer(train_df)
    return train_df, test_df, params


# --- temporal batching ----------------------------------------------------
def test_stub_month_merged_into_december(raw):
    df = add_batch_month(raw)
    jan_stub = df.loc[df["eid"] == 7, "batch_month"].iloc[0]
    assert str(jan_stub) == "2012-12"


def test_temporal_split_is_by_time(split):
    train_df, test_df = split
    assert train_df["batch_month"].max() <= pd.Period("2012-10", freq="M")
    assert test_df["batch_month"].min() > pd.Period("2012-10", freq="M")


# --- transform: leakage / schema -----------------------------------------
def test_transform_drops_leakage_and_raw_columns(fitted):
    train_df, _, params = fitted
    X = transform(train_df, params)
    for col in DROP:
        assert col not in X.columns


def test_train_test_schema_identical(fitted):
    train_df, test_df, params = fitted
    X_train = transform(train_df, params)
    X_test = transform(test_df, params)
    assert list(X_train.columns) == list(X_test.columns)


# --- transform: encodings / engineered features --------------------------
def test_rcount_numeric_after_mapping(fitted):
    train_df, _, params = fitted
    X = transform(train_df, params)
    assert pd.api.types.is_numeric_dtype(X["rcount"])
    assert X["rcount"].max() == 5   # "5+" mapped to 5


def test_comorbidity_count_in_range_and_no_nan(fitted):
    train_df, _, params = fitted
    X = transform(train_df, params)
    assert X["comorbidity_count"].between(0, 11).all()
    assert not X.isna().any().any()


# --- transform: data-quality fixes (fitted on train) ---------------------
def test_glucose_lows_imputed_positive(fitted):
    train_df, _, params = fitted
    X = transform(train_df, params)
    assert (X["glucose"] > 0).all()


def test_outlier_caps_applied(fitted):
    train_df, _, params = fitted
    X = transform(train_df, params)
    assert X["neutrophils"].max() <= params["neutrophils_cap"]
    assert X["bloodureanitro"].max() <= params["bloodureanitro_cap"]


def test_params_learned_from_train_only(fitted):
    train_df, _, params = fitted
    # Train facilities are A and B; the unseen test facility Z must NOT appear.
    assert params["facid_categories"] == ["A", "B"]


# --- transform: unknown-category (OOV) handling --------------------------
def test_unknown_facility_routes_to_other(fitted):
    train_df, test_df, params = fitted
    X_test = transform(test_df, params)
    z_idx = test_df.index[test_df["facid"] == "Z"][0]
    known_fac_cols = [c for c in X_test.columns if c.startswith("fac_") and c != "fac_other"]
    assert X_test.loc[z_idx, "fac_other"] == 1
    assert X_test.loc[z_idx, known_fac_cols].sum() == 0
