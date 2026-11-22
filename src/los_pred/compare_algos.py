"""Phase 2 (step 2) — algorithm comparison: LightGBM vs XGBoost vs linear baselines.

PLAN SUMMARY
============
train.py answered "how should we treat the target?" (winner: log1p). This script
answers the follow-up "does the *algorithm* matter, once the target treatment is
fixed?" — a second controlled experiment, one variable changed (the learner).

Controlled-experiment rule: same features, same temporal split, same target
treatment, same shared hyperparameters (n_estimators, learning_rate, seed).
Only the algorithm changes. Pass --treatment to hold a different treatment fixed;
it defaults to `log1p`, the winner from train.py.

Linear baselines (linear, ridge): added to answer "do we need boosted trees at all?".
They use the SAME features from features.transform() (already numeric, no NaNs),
with one extra step — feature SCALING (StandardScaler, inside a Pipeline):
  - Linear models are scale-sensitive: ridge's penalty shrinks all coefficients
    equally, so unscaled features (e.g. glucose in the 100s vs 0/1 flags) would be
    penalised unevenly; scaling also keeps the solver well-conditioned and makes
    coefficients comparable. OLS predictions are unchanged by scaling, but we scale
    both for a like-for-like pair.
  - Tree models (LightGBM/XGBoost) split on thresholds, so monotonic rescaling
    does not change their splits -> no scaler needed.
  - The scaler lives in the Pipeline, so it is fit on TRAIN only (no test leakage)
    and ships inside the logged model.
The `poisson` treatment is skipped for linear models (no Poisson objective here;
it would silently be plain OLS on raw LOS, duplicating `l2`).

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
import json
from datetime import datetime, timezone

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from los_pred.train import (
    BASELINE_MAE,
    PROJECT_ROOT,
    TRACKING_URI,
    evaluate,
    load_and_prepare,
)

EXPERIMENT = "los-pred-algo"
RESULTS_JSON = PROJECT_ROOT / "docs" / "algo_comparison.json"   # quick-reference copy of the MLflow metrics

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


def build_linear(_objective=None):
    # Scaling is REQUIRED for the linear family (see module docstring); trees skip it.
    return make_pipeline(StandardScaler(), LinearRegression())


def build_ridge(_objective=None):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=0))


# name -> (builder, key into TREATMENTS for its objective or None, native log_model fn)
# obj_key None = linear baseline: no objective, squared-error fit (log1p/raw per treatment).
ALGOS = {
    "lightgbm": (build_lgb,    "lgb_objective", mlflow.lightgbm.log_model),
    "xgboost":  (build_xgb,    "xgb_objective", mlflow.xgboost.log_model),
    "linear":   (build_linear, None,            mlflow.sklearn.log_model),
    "ridge":    (build_ridge,  None,            mlflow.sklearn.log_model),
}


def save_results(table: pd.DataFrame, treatment: str) -> None:
    """Write the comparison table to docs/algo_comparison.json for quick reference.

    Overwritten each run (latest comparison only); MLflow keeps the full history.
    """
    payload = {
        "experiment": EXPERIMENT,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "treatment": treatment,
        "baseline_mae": BASELINE_MAE,
        "results": table.round(4).to_dict(orient="records"),   # sorted best (lowest MAE) first
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved comparison -> {RESULTS_JSON}")


def run(treatment: str) -> None:
    """Train every algo in ALGOS on the fixed `treatment`, log each, print a table."""
    spec = TREATMENTS[treatment]
    log_target = spec["log_target"]
    eval_config = {"log_target": log_target}   # evaluate() only reads this key

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_test, y_test, _ = load_and_prepare()  # params unused here
    y_fit = np.log1p(y_train) if log_target else y_train

    results = []
    for algo, (build, obj_key, log_model) in ALGOS.items():
        if obj_key is None and treatment == "poisson":
            print(f"skipping {algo}: no Poisson objective for linear baselines")
            continue
        objective = spec[obj_key] if obj_key else "squared_error"
        with mlflow.start_run(run_name=f"{algo}-{treatment}") as run_ctx:
            model = build(objective)
            model.fit(X_train, y_fit)
            metrics = evaluate(model, eval_config, X_test, y_test)

            mlflow.log_params({
                "algo": algo,
                "treatment": treatment,
                "objective": objective,
                "scaled_features": obj_key is None,
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

    save_results(table, treatment)

    winner = table.iloc[0]
    print(f"\nbest algo: {winner['algo']} (MAE {winner['mae']:.3f}) "
          f"— not registered (comparison only; train.py owns the registry)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LightGBM, XGBoost and linear/ridge baselines on one fixed target treatment.",
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