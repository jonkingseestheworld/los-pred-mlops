"""Phase 3 — Model serving API (hospital length-of-stay).

Exposes the registered LOS model as a real-time HTTP service: POST a raw
admission record, get back a predicted length of stay in days.

Design (why it looks like this)
-------------------------------
- **FastAPI + Pydantic.** Pydantic (`Admission`) is the input contract — it
  validates and types the incoming JSON, rejecting malformed requests with a
  clear 422 *before* our code runs. FastAPI turns `predict()` into an HTTP
  endpoint and auto-documents it at /docs.
- **Load the live model by alias, via pyfunc.** `models:/los-pred-los@champion`
  follows whatever version is currently promoted (no hard-coded version). The
  flavor-agnostic `mlflow.pyfunc` interface means this code doesn't care whether
  @champion is LightGBM, XGBoost, etc.
- **No training-serving skew.** We rebuild features with the SAME
  `features.transform` used in training, fed the SAME fitted params that were
  persisted alongside the model (`transform_params.json` on the model's run).
  Then invert log1p (if the model used it) and floor-clip at 1 — identical to
  `train.py`'s `evaluate`.

Loaded once at startup (FastAPI lifespan): the pyfunc model, the fitted transform
params, and the `log_target` flag (read from the model's run).

Deps (install hands-on):  uv add fastapi uvicorn
Run:                      uv run uvicorn los_pred.app:app --reload --port 8000
Docs (Swagger UI):        http://127.0.0.1:8000/docs
Example:
    curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \\
      -d '{"rcount":"1","gender":"M","facid":"B","hematocrit":12.0,"neutrophils":9.0,
           "sodium":138.0,"glucose":100.0,"bloodureanitro":12.0,"creatinine":1.0,
           "bmi":29.0,"pulse":70,"dialysisrenalendstage":0,"asthma":0,"irondef":1,
           "pneum":0,"substancedependence":0,"psychologicaldisordermajor":0,
           "depress":0,"psychother":0,"fibrosisandother":0,"malnutrition":0,"hemo":0}'
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from los_pred.features import transform
from los_pred.train import REGISTERED_MODEL, TRACKING_URI

ALIAS = "champion"
# Deployment: point at a self-contained bundle (from los_pred.export_model) to
# serve without an MLflow tracking DB. Unset -> resolve the live @champion from
# the registry (local dev). Set by the Docker image.
LOCAL_MODEL_DIR = os.getenv("LOS_MODEL_DIR")


# --- input contract -------------------------------------------------------
class Admission(BaseModel):
    """One raw admission record — exactly the fields `features.transform` reads.

    (Identifiers, dates, and dropped columns like `respiration` are NOT required;
    they play no part in the feature matrix.)
    """

    rcount: str = Field(description="readmission count; '0'..'5' or '5+'")
    gender: str = Field(description="'M' or 'F'")
    facid: str = Field(description="facility id; unknown values route to 'other'")
    hematocrit: float = Field(ge=0, description="percentage; must be non-negative")
    neutrophils: float = Field(ge=0, description="count; must be non-negative")
    sodium: float = Field(ge=0, description="mEq/L; must be non-negative")
    glucose: float = Field(ge=0, description="mg/dL; must be non-negative")
    bloodureanitro: float = Field(ge=0, description="mg/dL; must be non-negative")
    creatinine: float = Field(ge=0, description="mg/dL; must be non-negative")
    bmi: float = Field(ge=0, description="kg/m²; must be non-negative")
    pulse: int = Field(ge=0, description="beats/min; must be non-negative")
    # 11 comorbidity flags (0/1)
    dialysisrenalendstage: int = Field(ge=0, le=1)
    asthma: int = Field(ge=0, le=1)
    irondef: int = Field(ge=0, le=1)
    pneum: int = Field(ge=0, le=1)
    substancedependence: int = Field(ge=0, le=1)
    psychologicaldisordermajor: int = Field(ge=0, le=1)
    depress: int = Field(ge=0, le=1)
    psychother: int = Field(ge=0, le=1)
    fibrosisandother: int = Field(ge=0, le=1)
    malnutrition: int = Field(ge=0, le=1)
    hemo: int = Field(ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "rcount": "1", "gender": "M", "facid": "B", "hematocrit": 12.0,
                "neutrophils": 9.0, "sodium": 138.0, "glucose": 100.0,
                "bloodureanitro": 12.0, "creatinine": 1.0, "bmi": 29.0, "pulse": 70,
                "dialysisrenalendstage": 0, "asthma": 0, "irondef": 1, "pneum": 0,
                "substancedependence": 0, "psychologicaldisordermajor": 0, "depress": 0,
                "psychother": 0, "fibrosisandother": 0, "malnutrition": 0, "hemo": 0,
            }
        }
    }



# --- prediction core (pure, unit-testable without HTTP) -------------------
def predict_days(record: dict, model, params: dict, log_target: bool) -> float:
    """raw record -> transform(params) -> model -> invert log1p -> clip >=1 -> days."""
    X = transform(pd.DataFrame([record]), params)
    pred = np.asarray(model.predict(X), dtype=float)
    if log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 1, None)
    return float(pred[0])


# --- artifact loading -----------------------------------------------------
def load_artifacts(alias: str = ALIAS) -> dict:
    """Load model (pyfunc) + fitted transform params + log_target.

    Model, params, and log_target always come from the SAME model version, so
    they can't drift apart — this is what prevents training-serving skew.

    Two modes:
    - **Baked bundle** (deployment) — if LOS_MODEL_DIR is set, load a
      self-contained folder produced by `los_pred.export_model`. No MLflow
      tracking DB needed (the local sqlite registry stores absolute artifact
      paths that don't exist inside a container).
    - **Registry** (local dev) — resolve the live `@alias` version from MLflow.
    """
    if LOCAL_MODEL_DIR:
        d = Path(LOCAL_MODEL_DIR)
        meta = json.loads((d / "metadata.json").read_text())
        params = json.loads((d / "transform_params.json").read_text())
        model = mlflow.pyfunc.load_model(str(d / "model"))
        return {"model": model, "params": params,
                "log_target": bool(meta["log_target"]), "version": meta["version"]}

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    run = client.get_run(mv.run_id)
    params = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/transform_params.json")
    log_target = run.data.params.get("log_target", "False") == "True"
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{alias}")
    return {"model": model, "params": params, "log_target": log_target,
            "version": mv.version}


# --- app ------------------------------------------------------------------
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(load_artifacts())            # load once on startup
    STATE["predictions_count"] = 0            # reset the volume counter each lifecycle
    print(f"loaded {REGISTERED_MODEL}@{ALIAS} v{STATE['version']} "
          f"(log_target={STATE['log_target']})")
    yield
    STATE.clear()


app = FastAPI(title="LOS prediction API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness/readiness check — reports which model version is serving."""
    return {"status": "ok", "model": REGISTERED_MODEL,
            "alias": ALIAS, "version": STATE.get("version")}


@app.get("/metrics")
def metrics() -> dict:
    """Prediction volume and model info."""
    return {"predictions_count": STATE.get("predictions_count", 0),
            "model": REGISTERED_MODEL, "version": STATE.get("version")}


@app.post("/predict")
def predict(admission: Admission) -> dict:
    """Predict length of stay (days) for one admission record."""
    days = predict_days(admission.model_dump(), STATE["model"],
                        STATE["params"], STATE["log_target"])
    STATE["predictions_count"] += 1
    return {"predicted_los_days": round(days, 2)}
