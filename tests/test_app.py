"""Tests for los_pred.app — the FastAPI serving layer.

Two levels, both hermetic (no MLflow registry / CSV needed):
  - predict_days(): the pure prediction core — real features.transform + the
    invert-log1p and floor-clip logic, driven by a stub model so the maths is
    exact.
  - HTTP endpoints: /health and /predict via FastAPI's TestClient, with
    `load_artifacts` monkeypatched to inject the stub model — so we test the
    request/validation/response wiring without loading the real model.

The TestClient tests need httpx (Starlette's test transport); they skip cleanly
if it's absent. Install it as a dev dep: uv add --dev httpx
"""

import numpy as np
import pytest

import los_pred.app as appmod
from los_pred.app import predict_days

# Fitted transform params (stub) — facid "B" is in-vocabulary.
STUB_PARAMS = {
    "facid_categories": ["A", "B"],
    "glucose_median": 100.0,
    "neutrophils_cap": 20.0,
    "bloodureanitro_cap": 30.0,
}

# One valid raw admission record — every field the Admission contract requires.
VALID_RECORD = {
    "rcount": "1", "gender": "M", "facid": "B", "hematocrit": 12.0,
    "neutrophils": 9.0, "sodium": 138.0, "glucose": 100.0, "bloodureanitro": 12.0,
    "creatinine": 1.0, "bmi": 29.0, "pulse": 70,
    "dialysisrenalendstage": 0, "asthma": 0, "irondef": 1, "pneum": 0,
    "substancedependence": 0, "psychologicaldisordermajor": 0, "depress": 0,
    "psychother": 0, "fibrosisandother": 0, "malnutrition": 0, "hemo": 0,
}


class StubModel:
    """Ignores X, predicts a fixed value (in whatever space the test wants)."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


# --- predict_days(): the prediction core ---------------------------------
def test_predict_days_returns_float():
    model = StubModel(np.log1p(3.0))
    out = predict_days(VALID_RECORD, model, STUB_PARAMS, log_target=True)
    assert isinstance(out, float)


def test_predict_days_inverts_log_target():
    """log-space prediction of log1p(5) must come back as ~5 days."""
    model = StubModel(np.log1p(5.0))
    out = predict_days(VALID_RECORD, model, STUB_PARAMS, log_target=True)
    assert out == pytest.approx(5.0, abs=1e-9)


def test_predict_days_floor_clips_at_1():
    """A raw prediction of 0 must be clipped up to the floor of 1 day."""
    model = StubModel(0.0)
    out = predict_days(VALID_RECORD, model, STUB_PARAMS, log_target=False)
    assert out == 1.0


# --- HTTP endpoints -------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    """TestClient with load_artifacts patched to inject the stub model."""
    pytest.importorskip("httpx", reason="TestClient needs httpx (uv add --dev httpx)")
    from fastapi.testclient import TestClient

    stub_state = {
        "model": StubModel(np.log1p(3.0)),  # -> 3.0 days after expm1
        "params": STUB_PARAMS,
        "log_target": True,
        "version": "test",
    }
    monkeypatch.setattr(appmod, "load_artifacts", lambda alias="champion": stub_state)
    with TestClient(appmod.app) as c:       # runs lifespan -> STATE = stub_state
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "test"


def test_predict_returns_prediction(client):
    resp = client.post("/predict", json=VALID_RECORD)
    assert resp.status_code == 200
    days = resp.json()["predicted_los_days"]
    assert days == pytest.approx(3.0)   # stub predicts log1p(3) -> expm1 -> 3.0
    assert days >= 1.0                  # floor always respected


def test_predict_rejects_invalid_input(client):
    """Pydantic must 422 a bad request (flag out of 0/1 range) before predicting."""
    bad = {**VALID_RECORD, "asthma": 5}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_tracks_volume(client):
    """Each /predict call increments the volume counter."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions_count"] == 0
    assert body["version"] == "test"

    # Make a prediction
    client.post("/predict", json=VALID_RECORD)
    resp = client.get("/metrics")
    assert resp.json()["predictions_count"] == 1

    # Make another
    client.post("/predict", json=VALID_RECORD)
    resp = client.get("/metrics")
    assert resp.json()["predictions_count"] == 2
