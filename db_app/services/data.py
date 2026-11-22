"""Dataset loading."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from los_pred.features import FLAGS, add_batch_month
from settings import DATA_CSV, SPLIT_CUTOFF


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Raw admissions + `batch_month` (str), `split` (train/test), `comorbidity_count`.

    Cached for the life of the process: the CSV is read once, not per session.
    """
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found — download the LOS dataset into data/ "
            "(the Description tab links to the source)."
        )
    df = add_batch_month(pd.read_csv(DATA_CSV))
    cut = pd.Period(SPLIT_CUTOFF, freq="M")
    df["split"] = np.where(df["batch_month"] <= cut, "train", "test")
    df["batch_month"] = df["batch_month"].astype(str)
    df["vdate"] = pd.to_datetime(df["vdate"])
    df["comorbidity_count"] = df[FLAGS].sum(axis=1)
    return df
