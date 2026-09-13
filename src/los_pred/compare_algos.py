"""Phase 2 (step 2) — algorithm comparison: LightGBM vs XGBoost.

PLAN SUMMARY
============
train.py answered "how should we treat the target?" (winner: log1p). This script
answers the follow-up "does the *algorithm* matter, once the target treatment is
fixed?" — a second controlled experiment, one variable changed (the learner).

Controlled-experiment rule: same features, same temporal split, same target
treatment, same shared hyperparameters (n_estimators, learning_rate, seed).
Only the algorithm changes. Pass --treatment to hold a different treatment fixed;
it defaults to `log1p`, the winner from train.py.

Honest caveat: tree-complexity isn't perfectly comparable across the two APIs —
LightGBM is leaf-wise (num_leaves), XGBoost is depth-wise (max_depth). We use
each library's conventional default (num_leaves=31 ≈ max_depth=6) rather than
forcing an artificial match. So read this as "each algo near its sensible
default," not a hyperparameter-tuned shoot-out.

Logging: each run uses its algorithm's NATIVE MLflow flavor
(mlflow.lightgbm / mlflow.xgboost). Native serialization avoids the restricted
unpickler that made mlflow.sklearn reject LightGBM's Booster types.

Logged to a SEPARATE experiment (`los-pred-algo`) so it doesn't mix with the
target-treatment runs in `los-pred`. This is a comparison, not a promotion — it
prints a winner but does NOT register a model (train.py owns the registry).

Deps (install hands-on):  uv add xgboost
Run:                      uv run python -m los_pred.compare_algos                 # log1p
                          uv run python -m los_pred.compare_algos --treatment l2
View UI:                  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from los_pred.train import (
    BASELINE_MAE,
    TRACKING_URI,
    evaluate,
    load_and_prepare,
)

EXPERIMENT = "los-pred-algo"

# Target treatment -> per-algorithm objective + whether to log1p the target.
# (Same three treatments as train.py, expressed for both learners.)
TREATMENTS: dict[str, dict] = {
    "l2":      {"lgb_objective": "regression", "xgb_objective": "reg:squarederror", "log_target": False},
    "log1p":   {"lgb_objective": "regression", "xgb_objective": "reg:squarederror", "log_target": True},
    "poisson": {"lgb_objective": "poisson",    "xgb_objective": "count:poisson",    "log_target": False},
}

# Shared across both algorithms — the controlled part of the experiment.
COMMON = {"n_estimators": 400, "learning_rate": 0.05, "random_state": 0}


def build_lgb(objective: str) -> LGBMRegressor:
    return LGBMRegressor(objective=objective, num_leaves=31, verbose=-1, **COMMON)


def build_xgb(objective: str) -> XGBRegressor:
    return XGBRegressor(objective=objective, max_depth=6, verbosity=0, **COMMON)


# name -> (builder, key into TREATMENTS for its objective, native log_model fn)
ALGOS = {
    "lightgbm": (build_lgb, "lgb_objective", mlflow.lightgbm.log_model),
    "xgboost":  (build_xgb, "xgb_objective", mlflow.xgboost.log_model),
}


def run(treatment: str) -> None:
    """Train LightGBM and XGBoost on the fixed `treatment`, log both, print a table."""
    spec = TREATMENTS[treatment]
    log_target = spec["log_target"]
    eval_config = {"log_target": log_target}   # evaluate() only reads this key

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_test, y_test = load_and_prepare()
    y_fit = np.log1p(y_train) if log_target else y_train

    results = []
    for algo, (build, obj_key, log_model) in ALGOS.items():
        with mlflow.start_run(run_name=f"{algo}-{treatment}") as run_ctx:
            model = build(spec[obj_key])
            model.fit(X_train, y_fit)
            metrics = evaluate(model, eval_config, X_test, y_test)

            mlflow.log_params({
                "algo": algo,
                "treatment": treatment,
                "objective": spec[obj_key],
                "log_target": log_target,
                "n_features": X_train.shape[1],
                **COMMON,
            })
            mlflow.set_tag("known_limitation", "target capped at 17; under-predicts long stays")
            mlflow.log_metrics({**metrics, "baseline_mae": BASELINE_MAE})
            log_model(model, name="model", input_example=X_train.head(3))

            results.append({"algo": algo, "run_id": run_ctx.info.run_id, **metrics})

    table = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    print(f"\ntreatment held fixed = {treatment}")
    print(table.to_string(index=False))
    print(f"naive baseline MAE = {BASELINE_MAE}")

    winner = table.iloc[0]
    print(f"\nbest algo: {winner['algo']} (MAE {winner['mae']:.3f}) "
          f"— not registered (comparison only; train.py owns the registry)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LightGBM vs XGBoost on one fixed target treatment.",
    )
    parser.add_argument(
        "--treatment",
        default="log1p",
        choices=list(TREATMENTS),
        help="target treatment to hold fixed (winner from train.py; default: log1p)",
    )
    args = parser.parse_args()
    run(args.treatment)


if __name__ == "__main__":
    main()




### "LightGBM and XGBoost perform equivalently at default settings (MAE 0.35x vs 0.35y, gap within seed noise). 
### LightGBM retained as the production algorithm for simplicity (incumbent, fewer deps, no OpenMP requirement); 
### the marginal MAE edge is consistent with but not the basis for this choice."