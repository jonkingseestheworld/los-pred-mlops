"""Data-cleaning audit: what the analysis pipeline actually changed, with live counts.

Nothing here re-implements the cleaning. It calls the same `fit_transformer` /
`transform` the model was trained with (`los_pred.features`), then measures what
they did — so this page can't drift away from the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from los_pred.features import TARGET, fit_transformer, transform
from services.data import load_dataset

# Columns the app adds on top of the raw CSV (not part of the raw data).
APP_COLUMNS = ["batch_month", "split", "comorbidity_count"]
# The three variables the pipeline corrects, and the rule per variable.
CORRECTED = {
    "glucose": "glucose_median",          # impossible values (<= 0) -> train median
    "neutrophils": "neutrophils_cap",     # upper winsorise at train 99th percentile
    "bloodureanitro": "bloodureanitro_cap",
}


@dataclass
class CleaningReport:
    n_rows: int
    n_duplicates: int
    n_missing: int
    n_removed: int                 # rows dropped by cleaning (the pipeline caps, never drops)
    leakage_match_pct: float       # % of rows where discharged - visit == LOS
    n_stub_rows: int               # rows relabelled into the previous monthly batch
    n_glucose_fixed: int
    n_neutrophils_capped: int
    n_bun_capped: int
    respiration_iqr: float
    respiration_median: float
    secondary_dx_corr: float       # correlation of the dropped count with LOS
    params: dict                   # everything learned from the training split
    effect: pd.DataFrame           # raw vs cleaned min/max + rows changed, per corrected variable
    split: pd.DataFrame            # train/test months, rows, share
    features: list[str]            # columns of the final model matrix


def build_cleaning_report(df: pd.DataFrame) -> CleaningReport:
    """Audit the cleaning applied to `df` (as returned by `load_dataset`)."""
    raw = df.drop(columns=[c for c in APP_COLUMNS if c in df.columns])
    train = df[df["split"] == "train"]
    params = fit_transformer(train)                    # learned on TRAIN only
    cleaned = transform(df.drop(columns=["split"]), params)

    rows = []
    n_changed = {}
    for var in CORRECTED:
        changed = int((raw[var] != cleaned[var]).sum())
        n_changed[var] = changed
        rows.append({
            "Variable": var,
            "Raw min": raw[var].min(), "Raw max": raw[var].max(),
            "Cleaned min": cleaned[var].min(), "Cleaned max": cleaned[var].max(),
            "Rows changed": changed,
        })

    split = (
        df.groupby("split")
        .agg(first=("batch_month", "min"), last=("batch_month", "max"), rows=("split", "size"))
        .reindex(["train", "test"])
        .reset_index()
    )
    split["share"] = split["rows"] / len(df)

    resp = raw["respiration"]
    discharge_gap = (pd.to_datetime(raw["discharged"]) - raw["vdate"]).dt.days
    return CleaningReport(
        n_rows=len(raw),
        n_duplicates=int(raw.duplicated().sum()),
        n_missing=int(raw.isna().sum().sum()),
        n_removed=len(raw) - len(cleaned),
        leakage_match_pct=float((discharge_gap == raw[TARGET]).mean() * 100),
        n_stub_rows=int((raw["vdate"].dt.to_period("M").astype(str) != df["batch_month"]).sum()),
        n_glucose_fixed=n_changed["glucose"],
        n_neutrophils_capped=n_changed["neutrophils"],
        n_bun_capped=n_changed["bloodureanitro"],
        respiration_iqr=float(resp.quantile(0.75) - resp.quantile(0.25)),
        respiration_median=float(resp.median()),
        secondary_dx_corr=float(raw["secondarydiagnosisnonicd9"].corr(raw[TARGET])),
        params={k: v for k, v in params.items()},
        effect=pd.DataFrame(rows),
        split=split,
        features=list(cleaned.columns),
    )


@lru_cache(maxsize=1)
def cleaning_report() -> CleaningReport:
    """The audit for the real dataset, computed once per process."""
    return build_cleaning_report(load_dataset())
