"""Monitoring for the length-of-stay model.

Reusable, scriptable version of `notebooks/03_monitoring.ipynb`: given a model
and the fitted feature params, generate Evidently **data-drift** and **regression
performance** reports comparing each monthly production batch against the
training reference.

Separation of concerns:
- `features.py` builds the feature matrix (shared with training + serving).
- `monitor.py` (here) turns a model + data into monitoring reports.
- Model *training* belongs in `train.py` (Phase 2). `main()` trains a simple
  stand-in baseline so this module is runnable today; once Phase 2 registers a
  real model, load that here instead of training one.

Run:
    uv run python -m los_pred.monitor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from lightgbm import LGBMRegressor


from los_pred.features import (
    FLAGS,
    TARGET,
    add_batch_month,
    fit_transformer,
    temporal_split,
    transform,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "LengthOfStay.csv"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
DEFAULT_CUTOFF = "2012-10"


# --- schema ---------------------------------------------------------------
def feature_schema(X: pd.DataFrame) -> DataDefinition:
    """Split the feature matrix into categorical (binary flags / one-hot / isMale)
    and numerical columns for Evidently drift detection."""
    cat = [c for c in X.columns if c.startswith("fac_") or c in FLAGS or c == "isMale"]
    num = [c for c in X.columns if c not in cat]
    return DataDefinition(numerical_columns=num, categorical_columns=cat)


# --- individual reports ---------------------------------------------------
def data_drift_report(reference_X, current_X, schema, out_path: Path) -> Path:
    """Save a data-drift report comparing a current batch to the reference."""
    reference_ds = Dataset.from_pandas(reference_X, data_definition=schema)
    current_ds = Dataset.from_pandas(current_X, data_definition=schema)
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_ds, reference_ds)   # (current, reference)
    snapshot.save_html(str(out_path))
    return out_path


def regression_report(y_ref, pred_ref, y_cur, pred_cur, out_path: Path) -> Path:
    """Save a regression performance report (current batch vs reference)."""
    reg_def = DataDefinition(regression=[Regression(target=TARGET, prediction="prediction")])

    def _ds(y, pred):
        frame = pd.DataFrame({TARGET: np.asarray(y), "prediction": np.asarray(pred)})
        return Dataset.from_pandas(frame, data_definition=reg_def)

    report = Report([RegressionPreset()])
    snapshot = report.run(_ds(y_cur, pred_cur), _ds(y_ref, pred_ref))   # (current, reference)
    snapshot.save_html(str(out_path))
    return out_path


# --- orchestration --------------------------------------------------------
def generate_reports(
    raw_df: pd.DataFrame,
    model,
    params: dict,
    cutoff: str = DEFAULT_CUTOFF,
    out_dir: Path = REPORTS_DIR,
) -> list[Path]:
    """For each month after `cutoff`, write a drift report and a performance
    report vs the training reference. Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = pd.Period(cutoff, freq="M")

    df = raw_df if "batch_month" in raw_df else add_batch_month(raw_df)
    X_all = transform(df, params)
    y_all = df[TARGET]
    pred_all = pd.Series(np.clip(model.predict(X_all), 1, None), index=X_all.index)

    ref_mask = df["batch_month"] <= cut
    schema = feature_schema(X_all)
    current_months = [m for m in sorted(df["batch_month"].unique()) if m > cut]

    written: list[Path] = []
    for m in current_months:
        cur_mask = df["batch_month"] == m
        written.append(data_drift_report(
            X_all[ref_mask], X_all[cur_mask], schema, out_dir / f"drift_{m}.html",
        ))
        written.append(regression_report(
            y_all[ref_mask], pred_all[ref_mask], y_all[cur_mask], pred_all[cur_mask],
            out_dir / f"performance_{m}.html",
        ))
    return written


def main() -> None:
    """Train a stand-in baseline and generate the monthly monitoring reports.

    Placeholder model until Phase 2's train.py registers a real one — swap the
    training block for loading the registered model when that exists.
    """
    from lightgbm import LGBMRegressor

    raw = add_batch_month(pd.read_csv(DATA_CSV))
    train_df, _ = temporal_split(raw, cutoff=DEFAULT_CUTOFF)

    params = fit_transformer(train_df)
    model = LGBMRegressor(random_state=0)
    model.fit(transform(train_df, params), train_df[TARGET])

    paths = generate_reports(raw, model, params)
    print(f"wrote {len(paths)} reports to {REPORTS_DIR}:")
    for p in paths:
        print(" ", p.name)


if __name__ == "__main__":
    main()
