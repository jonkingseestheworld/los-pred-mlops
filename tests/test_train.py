"""Tests for los_pred.train — the training/evaluation pipeline.

Focus on the logic train.py owns on top of features.py:
  - evaluate(): the day-space scoring contract (metric keys, floor-clip at 1,
    log1p back-transform) — verified with tiny stub models so the maths is
    exact and CSV-free.
  - train_variant() + evaluate(): a smoke test that every target-treatment
    variant actually fits and scores (exercises the log1p and poisson paths).
  - load_and_prepare(): a leakage guard (train/test rows disjoint, same schema),
    skipped when the gitignored real CSV isn't present.
"""

import numpy as np
import pandas as pd
import pytest

from los_pred.train import (
    DATA_CSV,
    VARIANTS,
    evaluate,
    train_variant,
)


class ArrayModel:
    """Stub model: ignores X, returns a fixed prediction array (len must match)."""

    def __init__(self, preds):
        self.preds = np.asarray(preds, dtype=float)

    def predict(self, X):
        return self.preds


def _dummy_X(n):
    """A DataFrame of the right length; evaluate() only uses len(X) via predict."""
    return pd.DataFrame({"f": np.zeros(n)})


# --- evaluate(): metric contract -----------------------------------------
def test_evaluate_returns_expected_keys():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate(ArrayModel(np.ones(4)), {"log_target": False}, _dummy_X(4), y)
    assert set(metrics) == {"mae", "rmse", "residual_mean"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


# --- evaluate(): floor-clip at 1 -----------------------------------------
def test_evaluate_floor_clips_predictions():
    """Raw preds of 0 must be clipped to 1 before scoring."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate(ArrayModel(np.zeros(4)), {"log_target": False}, _dummy_X(4), y)
    # clipped preds = 1 -> residuals = y - 1 = [0, 1, 2, 3]
    assert metrics["mae"] == pytest.approx(1.5)          # mean(|y-1|)
    assert metrics["residual_mean"] == pytest.approx(1.5)  # mean(y-1), all under-predicted
    # sanity: without clipping (preds=0) MAE would be mean(y)=2.5, so clip mattered.


# --- evaluate(): log1p back-transform ------------------------------------
def test_evaluate_inverts_log_target():
    """A model predicting log1p(y) must score ~0 error once expm1-inverted."""
    y = pd.Series([1.0, 2.0, 4.0, 7.0])
    model = ArrayModel(np.log1p(y.to_numpy()))
    metrics = evaluate(model, {"log_target": True}, _dummy_X(len(y)), y)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["residual_mean"] == pytest.approx(0.0, abs=1e-9)


# --- train_variant() + evaluate(): all variants fit and score ------------
@pytest.fixture
def synthetic_split():
    """A CSV-free numeric train/test split — enough rows for the learners to fit."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(1, 10, size=200).astype(float))
    return X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:]


@pytest.mark.parametrize("name,config", list(VARIANTS.items()))
def test_variant_trains_and_evaluates(name, config, synthetic_split):
    X_train, y_train, X_test, y_test = synthetic_split
    model = train_variant(config, X_train, y_train)
    metrics = evaluate(model, config, X_test, y_test)
    assert np.isfinite(metrics["mae"]) and metrics["mae"] >= 0
    assert np.isfinite(metrics["rmse"]) and metrics["rmse"] >= 0


# --- load_and_prepare(): leakage guard (needs the real, gitignored CSV) ---
@pytest.mark.skipif(not DATA_CSV.exists(), reason="real CSV is gitignored / absent")
def test_load_and_prepare_no_row_overlap():
    from los_pred.train import load_and_prepare

    X_train, y_train, X_test, y_test = load_and_prepare()
    assert len(X_train) > 0 and len(X_test) > 0
    assert list(X_train.columns) == list(X_test.columns)      # identical schema
    assert set(X_train.index).isdisjoint(X_test.index)         # no shared rows
