"""Azure ML managed online endpoint — scoring script (hospital length-of-stay).

Azure ML's inference server imports this file and calls:
  - init()          once, when the deployment container starts
  - run(raw_data)   for every request (raw_data = the request body as a string)

It serves the SAME model bundle as the Docker/FastAPI app (`serving_model/`:
model + transform_params.json + metadata.json), through the SAME prediction path
(`los_pred.predict`), so a raw admission record gets identical features and
post-processing (invert log1p, floor-clip at 1) on every serving surface.

Request body — one record, a list, or {"data": [...]}:
    {"data": [{"rcount": "1", "gender": "M", "facid": "B", ...}]}
Response:
    {"predicted_los_days": [4.09]}

Packaging: `azure/package_code.sh` builds a FLAT `azure/code/` = this file +
features.py + predict.py. Upload those three files as the deployment's code
(scoring script + optional scripts in the studio; `code:` folder in the CLI).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types
from pathlib import Path

import mlflow.pyfunc

# The Azure ML studio uploader places scripts SIDE BY SIDE in one flat folder
# (score.py, predict.py, features.py) — there is no `los_pred/` directory. The
# project modules import each other as `los_pred.*`, so register a `los_pred`
# package whose search path is this folder; `los_pred.predict` then resolves to
# the sibling predict.py (and its `los_pred.features` import to features.py).
# That keeps one source of truth in src/ with no import rewriting.
if "los_pred" not in sys.modules:
    _pkg = types.ModuleType("los_pred")
    _pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["los_pred"] = _pkg

from los_pred.predict import predict_days_batch  # noqa: E402  (after the shim)

_STATE: dict = {}


def _find_bundle_dir() -> Path:
    """Locate the registered bundle under AZUREML_MODEL_DIR.

    Azure mounts the registered model at $AZUREML_MODEL_DIR; depending on how it
    was registered, the bundle may sit in a subfolder (e.g. .../serving_model/),
    so search for its metadata.json rather than hard-coding the layout.
    """
    root = Path(os.environ["AZUREML_MODEL_DIR"])
    try:
        return next(root.rglob("metadata.json")).parent
    except StopIteration:
        raise FileNotFoundError(f"no model bundle (metadata.json) found under {root}")


def init() -> None:
    """Load the model, fitted transform params, and log_target once at startup."""
    bundle = _find_bundle_dir()
    meta = json.loads((bundle / "metadata.json").read_text())
    _STATE["params"] = json.loads((bundle / "transform_params.json").read_text())
    _STATE["log_target"] = bool(meta["log_target"])
    _STATE["version"] = meta["version"]
    _STATE["model"] = mlflow.pyfunc.load_model(str(bundle / "model"))
    logging.info("loaded LOS model v%s (log_target=%s) from %s",
                 _STATE["version"], _STATE["log_target"], bundle)


def _parse_records(raw_data: str) -> list[dict]:
    """Request body -> list of raw admission records."""
    payload = json.loads(raw_data)
    records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return [records] if isinstance(records, dict) else list(records)


def run(raw_data: str) -> dict:
    """Score one request: raw admission record(s) -> predicted LOS days."""
    days = predict_days_batch(
        _parse_records(raw_data), _STATE["model"], _STATE["params"], _STATE["log_target"],
    )
    return {"predicted_los_days": [round(float(d), 2) for d in days]}
