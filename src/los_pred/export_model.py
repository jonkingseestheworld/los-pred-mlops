"""Export the live @champion model into a self-contained `serving_model/` bundle.

Why: a deployment image should ship an immutable model, not depend on a live
MLflow registry at runtime. Our local sqlite registry also stores *absolute*
artifact paths (e.g. /Users/.../mlruns/...) that don't exist inside a container,
so the registry can't be used from the image anyway. This copies everything the
FastAPI app needs into one folder with relative paths:

    serving_model/
      model/                  <- the MLflow model (MLmodel + binaries)
      transform_params.json   <- the fitted transform state
      metadata.json           <- {"log_target": ..., "version": ...}

The app loads this folder when LOS_MODEL_DIR is set (see los_pred.app).
Re-run after promoting a new @champion.

Run:  uv run python -m los_pred.export_model
"""

from __future__ import annotations

import json
import shutil

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

from los_pred.train import PROJECT_ROOT, REGISTERED_MODEL, TRACKING_URI

ALIAS = "champion"
OUT = PROJECT_ROOT / "serving_model"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    run = client.get_run(mv.run_id)
    log_target = run.data.params.get("log_target", "False") == "True"
    # MLflow stores all parameters as strings (even booleans). 
    # So you can't do log_target == True — it would compare a string to a boolean, which is always False. 
    # You have to compare string-to-string: "True" == "True".
    
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # model artifacts (MLmodel + binaries) -> serving_model/model/
    download_artifacts(artifact_uri=f"models:/{REGISTERED_MODEL}@{ALIAS}",
                       dst_path=str(OUT / "model"))
    # fitted transform params (a run artifact) -> serving_model/transform_params.json
    download_artifacts(artifact_uri=f"runs:/{mv.run_id}/transform_params.json",
                       dst_path=str(OUT))
    # metadata the app needs at load time (log_target for the back-transform)
    (OUT / "metadata.json").write_text(
        json.dumps({"log_target": log_target, "version": mv.version}, indent=2))

    print(f"exported {REGISTERED_MODEL}@{ALIAS} v{mv.version} -> {OUT}")


if __name__ == "__main__":
    main()
