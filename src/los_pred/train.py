"""Phase 2 — Experiment tracking & model training (hospital length-of-stay).

PLAN SUMMARY
============
Goal: replace ad-hoc model saving with a structured, reproducible MLflow
experiment, and use it to answer one concrete modelling question —

    How should we treat the right-skewed target (LOS, integer 1-17, mean 4)?

Skew handling is a modelling *choice* made via the objective / target transform,
not something gradient boosting fixes automatically. So we test it as a
controlled A/B/C experiment: three variants that differ in EXACTLY ONE thing
(how the target is treated), compared on the same held-out test set.

Controlled-experiment rule: same features, same temporal split, same algorithm
(LightGBM), same hyperparameters across all variants. Only the target treatment
changes, so any difference in error is attributable to that.

THE THREE VARIANTS (LightGBM objectives)
----------------------------------------
  (a) l2       — raw target, objective="regression" (L2, the control).
                 Squares errors -> over-weights the rare long stays.
  (b) log1p    — train on np.log1p(y), predict, invert with np.expm1().
                 Compresses the right tail -> optimises relative error.
                 Caveat: expm1 back-transform is slightly biased (Jensen).
  (c) poisson  — raw target, objective="poisson".
                 Principled match for a positive integer COUNT; never negative.

EVALUATION
----------
  - Score every variant in REAL DAYS: predict -> invert any transform ->
    clip at the floor (>= 1) -> then compute the metric. Never compare a
    log-space score against a day-space score.
  - MAE (days) = headline / decision metric. RMSE (days) = secondary.
  - Must beat the naive baseline from 01_eda: predict-the-mean MAE = 1.914.
  - Winner = lowest MAE that also beats the baseline. Registered in the
    MLflow Model Registry (Phase 2 deliverable).

Data / split contract comes from src/los_pred/features.py (single source of
truth): temporal split, train-only fit of encodings + DQ fixes, floor-clip at 1.
The 17 upper bound is a dataset artifact (logged as a known-limitation tag).

Deps (install hands-on): uv add mlflow lightgbm scikit-learn
Run:                     uv run python -m los_pred.train
View UI:                 uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
#import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from los_pred.features import (
    TARGET,
    add_batch_month,
    fit_transformer,
    temporal_split,
    transform,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "LengthOfStay.csv"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"   # sqlite backend enables the Model Registry locally
EXPERIMENT = "los-pred"
REGISTERED_MODEL = "los-pred-los"
DEFAULT_CUTOFF = "2012-10"

# The one thing that differs between runs; everything else is held constant.
VARIANTS: dict[str, dict] = {
    "l2":      {"objective": "regression", "log_target": False},
    "log1p":   {"objective": "regression", "log_target": True},
    "poisson": {"objective": "poisson",    "log_target": False},
}

# Shared hyperparameters — identical across variants (the controlled part).
HYPERPARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 0,
    "verbose": -1,
}

BASELINE_MAE = 1.914  # predict-the-mean floor from notebooks/01_eda.ipynb


def load_and_prepare(cutoff: str = DEFAULT_CUTOFF):
    """Load raw data, temporal-split, fit the transform on TRAIN only, and
    return (X_train, y_train, X_test, y_test)."""
    raw = add_batch_month(pd.read_csv(DATA_CSV))
    train_df, test_df = temporal_split(raw, cutoff=cutoff)
    params = fit_transformer(train_df)
    X_train, y_train = transform(train_df, params), train_df[TARGET]
    X_test, y_test = transform(test_df, params), test_df[TARGET]
    return X_train, y_train, X_test, y_test


def train_variant(config: dict, X_train, y_train) -> LGBMRegressor:
    """Train one variant. If config['log_target'], fit on np.log1p(y_train)."""
    y_fit = np.log1p(y_train) if config["log_target"] else y_train
    model = LGBMRegressor(objective=config["objective"], **HYPERPARAMS)
    model.fit(X_train, y_fit)
    return model


def evaluate(model, config: dict, X_test, y_test) -> dict:
    """Predict -> invert log1p if used -> clip at 1 -> metrics in real days.

    residual_mean = mean(actual - predicted) = model bias: >0 under-predicts,
    <0 over-predicts. Useful for spotting the expected log1p back-transform bias
    (Jensen) when comparing variants in the MLflow UI.
    """
    pred = model.predict(X_test)
    if config["log_target"]:
        pred = np.expm1(pred)
    pred = np.clip(pred, 1, None)
    residuals = y_test.to_numpy() - pred
    return {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "residual_mean": float(residuals.mean()),
    }


def main() -> None:
    """Loop VARIANTS, log each as an MLflow run, print an MAE/RMSE table vs the
    naive baseline, and register the best model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_test, y_test = load_and_prepare()

    results = []
    for name, config in VARIANTS.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "variant": name,
                "objective": config["objective"],
                "log_target": config["log_target"],
                "n_features": X_train.shape[1],
                **HYPERPARAMS,
            })
            mlflow.set_tag("known_limitation", "target capped at 17; under-predicts long stays")

            model = train_variant(config, X_train, y_train)
            metrics = evaluate(model, config, X_test, y_test)
            mlflow.log_metrics({**metrics, "baseline_mae": BASELINE_MAE})
            mlflow.lightgbm.log_model(model, name="model", input_example=X_train.head(3))
            #mlflow.sklearn.log_model(model, name="model") #, input_example=X_train.head(3))

            results.append({"variant": name, "run_id": run.info.run_id, **metrics})

    table = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    print("\n" + table.to_string(index=False))
    print(f"naive baseline MAE = {BASELINE_MAE}")

    best = table.iloc[0]
    if best["mae"] < BASELINE_MAE:
        mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL)
        print(f"\nregistered '{REGISTERED_MODEL}' <- variant '{best['variant']}' (MAE {best['mae']:.3f})")
    else:
        print("\nno variant beat the baseline — nothing registered")


if __name__ == "__main__":
    main()



## To run the script with uv: uv run python -m los_pred.train  
## To view the MLflow UI: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db