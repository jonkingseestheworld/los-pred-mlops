"""Feature engineering for the length-of-stay model.

Lifted from `notebooks/02_feature_engineering.ipynb`. This is the single source
of truth for turning raw admission rows into the model feature matrix — the same
`transform()` is used at training time and at serving time (Phase 3 API), which
is what prevents training-serving skew.

Design:
- `add_batch_month` / `temporal_split` — build the monthly batch label (with the
  2013-01-01 stub merged into Dec) and split by time, not randomly.
- `fit_transformer(train)` — learn the *stateful* params from TRAIN only
  (facid vocabulary, glucose median, the two winsorize caps).
- `transform(frame, params)` — apply everything and return the feature matrix X
  (no target). Unknown facilities route to `fac_other`.
"""

from __future__ import annotations

import pandas as pd

# The 11 comorbidity flags (already 0/1 in the raw data).
FLAGS = [
    "dialysisrenalendstage", "asthma", "irondef", "pneum", "substancedependence",
    "psychologicaldisordermajor", "depress", "psychother", "fibrosisandother",
    "malnutrition", "hemo",
]

TARGET = "lengthofstay"

# Columns removed from the feature matrix: identifiers, leakage (dates),
# replaced-by-encoding (gender/facid), near-constant (respiration), a ~0-signal
# count (secondarydiagnosisnonicd9), the batch label, and the target itself.
DROP = [
    "eid", "vdate", "discharged", "gender", "facid",
    "respiration", "secondarydiagnosisnonicd9", "batch_month", TARGET,
]


# --- temporal batching / split -------------------------------------------
def add_batch_month(
    df: pd.DataFrame,
    stub: str = "2013-01",
    merge_into: str = "2012-12",
) -> pd.DataFrame:
    """Return a copy with a `batch_month` period column. The single-day `stub`
    month is merged into `merge_into` (keeps the raw `vdate` intact)."""
    month = pd.to_datetime(df["vdate"]).dt.to_period("M")
    merged = month.where(month < stub, pd.Period(merge_into, freq="M"))
    return df.assign(batch_month=merged)


def temporal_split(
    df: pd.DataFrame,
    cutoff: str = "2012-10",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: batch_month <= cutoff -> train, later -> test.
    Requires `add_batch_month` to have run first."""
    cut = pd.Period(cutoff, freq="M")
    is_train = df["batch_month"] <= cut
    return df[is_train].copy(), df[~is_train].copy()


# --- fit / transform ------------------------------------------------------
def _cap_99th(s: pd.Series) -> float:
    """Upper cap at the 99th percentile (learned on train)."""
    return s.quantile(0.99)


def fit_transformer(train: pd.DataFrame) -> dict:
    """Learn all stateful transform params from TRAIN only."""
    return {
        "facid_categories": sorted(train["facid"].unique()),
        "glucose_median": train.loc[train["glucose"] > 0, "glucose"].median(),
        "neutrophils_cap": _cap_99th(train["neutrophils"]),
        "bloodureanitro_cap": _cap_99th(train["bloodureanitro"]),
    }


def transform(frame: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Apply the fitted transform. Pure: returns the feature matrix X (no target)."""
    out = frame.copy()

    # --- stateless encodings / engineered features ---
    out["isMale"] = (out["gender"] == "M").astype(int)
    out["rcount"] = pd.to_numeric(out["rcount"].replace("5+", "5"))
    out["comorbidity_count"] = out[FLAGS].sum(axis=1)

    # --- stateful: DQ fixes (params learned on train) ---
    out.loc[out["glucose"] <= 0, "glucose"] = p["glucose_median"]
    out["neutrophils"] = out["neutrophils"].clip(upper=p["neutrophils_cap"])
    out["bloodureanitro"] = out["bloodureanitro"].clip(upper=p["bloodureanitro_cap"])

    # --- stateful: facid one-hot over the TRAIN vocabulary + 'other' (OOV) ---
    cats = list(p["facid_categories"]) + ["other"]
    known = out["facid"].where(out["facid"].isin(p["facid_categories"]), "other")
    dummies = pd.get_dummies(pd.Categorical(known, categories=cats), prefix="fac", dtype=int)
    dummies.index = out.index
    out = pd.concat([out, dummies], axis=1)

    return out.drop(columns=[c for c in DROP if c in out.columns])
