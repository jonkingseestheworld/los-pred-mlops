"""The prediction path shared by every serving surface (FastAPI, Azure ML endpoint).

raw admission records -> features.transform(params) -> model -> invert log1p
(if the model used it) -> floor-clip at 1 -> predicted days.

Kept dependency-light on purpose (numpy + pandas + `features` only — no FastAPI,
no MLflow) so any runtime can import it. This is the single source of truth for
the skew-sensitive post-processing; mirrors `train.evaluate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from los_pred.features import transform


def predict_days_batch(records: list[dict], model, params: dict, log_target: bool) -> np.ndarray:
    """Predict length of stay (days) for a list of raw admission records."""
    X = transform(pd.DataFrame(records), params)
    pred = np.asarray(model.predict(X), dtype=float)
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 1, None)


def predict_days(record: dict, model, params: dict, log_target: bool) -> float:
    """Predict length of stay (days) for one raw admission record."""
    return float(predict_days_batch([record], model, params, log_target)[0])
