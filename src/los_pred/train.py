"""Phase 2 — Experiment tracking & model training (hospital length-of-stay).

PLAN SUMMARY
============
Goal: replace ad-hoc model saving with a structured, reproducible MLflow
experiment, and use it to answer one concrete modelling question —

    How should we treat the right-skewed target (LOS, integer 1-17, mean 4)?

We do NOT assume "XGBoost handles skew". Skew handling is a modelling *choice*
made via the objective / target transform, so we test it as a controlled A/B/C
experiment: train three variants that differ in EXACTLY ONE thing (how the
target is treated) and compare them on the same held-out test set.

Controlled-experiment rule: same features, same temporal split, same algorithm,
same hyperparameters across all variants. The only thing that changes is the
target treatment below. Any difference in error is then attributable to that.

THE THREE VARIANTS
------------------
  (a) squarederror  — raw target, objective="reg:squarederror" (the control).
                      Squares errors -> over-weights the rare long stays.
  (b) log1p         — train on np.log1p(y), predict, invert with np.expm1().
                      Compresses the right tail -> optimises relative error.
                      Caveat: expm1 back-transform is slightly biased (Jensen).
  (c) poisson       — raw target, objective="count:poisson" (or reg:tweedie).
                      Principled match for a positive integer COUNT; never
                      predicts negatives.

EVALUATION
----------
  - Score every variant in REAL DAYS: predict -> invert any transform ->
    clip at the floor (>= 1) -> then compute the metric. Never compare a
    log-space score against a day-space score.
  - MAE (mean absolute error, days) = headline / decision metric (robust to
    skew, interpretable: "typically off by X days").
  - RMSE (days) = secondary (flags whether a variant wins on the bulk but
    loses on the tail).
  - Must beat the naive baseline from 01_eda: predict-the-mean MAE = 1.914.
  - Winner = lowest MAE that also beats the baseline. Register it in the
    MLflow Model Registry (Phase 2 deliverable).

DATA / SPLIT CONTRACT (from 01_eda §8)
--------------------------------------
  - Drop from features: eid, vdate, discharged  (discharged = leakage).
  - Encodings: gender -> isMale(0/1); rcount "5+"->5; facid -> one-hot;
    flags already 0/1; engineered comorbidity_count = sum of 11 flags.
  - DQ fixes fit on TRAIN ONLY: glucose > 0 bound; winsorize neutrophils &
    bloodureanitro; consider dropping near-constant respiration.
  - Split is TEMPORAL not random: train on earlier months -> test on later
    months. Batches = 12 months of 2012 (merge the 2013-01-01 stub into Dec).
  - Predictions clipped to the floor (>= 1). The 17 upper bound is a dataset
    artifact, NOT a real ceiling — do not clip-and-forget; log it as a known
    limitation (under-serves genuinely long stays).

MLflow: one run per variant, logging params (variant, objective, features,
hyperparameters), metrics (mae, rmse) and the fitted model artifact.

Deps (install hands-on when you reach this phase):
    uv add mlflow scikit-learn xgboost
Run:
    uv run python -m los_pred.train
"""

from __future__ import annotations

# --- Variant registry ------------------------------------------------------
# Each entry captures the ONE thing that differs between runs. Everything else
# (features, split, hyperparameters) is held constant by the training loop.
VARIANTS: dict[str, dict] = {
    "squarederror": {"objective": "reg:squarederror", "log_target": False},
    "log1p":        {"objective": "reg:squarederror", "log_target": True},
    "poisson":      {"objective": "count:poisson",     "log_target": False},
}

BASELINE_MAE = 1.914  # predict-the-mean floor from notebooks/01_eda.ipynb


# --- Skeleton (fill in during Phase 2) -------------------------------------
def load_and_prepare():
    """Load LengthOfStay.csv, apply the §8 encodings, return a feature frame
    plus the target and the temporal batch_month label. (DQ fixes fit on
    train only — do them after the split, not here.)"""
    raise NotImplementedError


def temporal_split(df):
    """Split by batch_month: earlier months -> train, later months -> test."""
    raise NotImplementedError


def train_variant(name, config, X_train, y_train):
    """Train one variant. If config['log_target'], fit on np.log1p(y_train)."""
    raise NotImplementedError


def evaluate(model, config, X_test, y_test):
    """Predict -> invert log1p (np.expm1) if used -> clip at 1 -> MAE & RMSE
    in real days. Returns {'mae': ..., 'rmse': ...}."""
    raise NotImplementedError


def main():
    """Loop VARIANTS, log each as an MLflow run, print an MAE/RMSE table vs
    BASELINE_MAE, and register the best model."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
