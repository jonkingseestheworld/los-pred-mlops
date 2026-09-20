# los-pred-mlops
Project covering the important aspects of MLOPS workflow in the context of healthcare service utilisation

## Setup

**1. Install Python 3.12**
- **macOS:** `brew install python@3.12`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

Verify: `python3.12 --version`

**2. Install uv (package manager)**
[uv](https://docs.astral.sh/uv/) is a fast Python package installer. One-liner:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Verify: `uv --version`

**3. Clone and set up the project**
```bash
git clone <repo-url>
cd los-pred-mlops
uv sync              # Creates virtual env and installs dependencies
```

**4. Using in VSCode**
- Open the project folder in VSCode
- VSCode should auto-detect the virtual environment (created by `uv sync`)
- Select Python interpreter: <kbd>Cmd/Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> → "Python: Select Interpreter" → Choose the uv venv

**Running commands:**
```bash
uv run jupyter notebook          # Start Jupyter
uv run python script.py          # Run a Python script
uv add package_name              # Add a package (updates pyproject.toml)
```

## Getting Started

**1. Data**
Download the [Kaggle LOS dataset](https://www.kaggle.com/datasets/ashimsharanr/hospital-length-of-stay-elos-prediction) and place `LengthOfStay.csv` in the `data/` folder.

> **Note — data versioning (not set up here).** `data/` is gitignored, so neither the
> raw CSV nor the processed `data/processed/*.parquet` splits are tracked in git. For this
> local learning project that's fine: the raw CSV is the single source of truth, and the
> parquet splits are a **reproducible output** of `02_feature_engineering.ipynb` — re-run it
> to regenerate them. In a real/production or team setting, version the data with
> [**DVC**](https://dvc.org/): git tracks a small `.dvc` pointer while the actual bytes live
> in a private remote (local dir, Azure Blob, S3, …), never on GitHub — keeping large/sensitive
> data reproducible and out of git history. `uv add --dev dvc` when that day comes.

**2. Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Covers data overview, distributions, correlations, and baseline predictions.

**3. Training & experiment tracking (Phase 2)**
```bash
uv add mlflow lightgbm scikit-learn xgboost
```

Train the model and run the target-treatment experiment (l2 / log1p / poisson).
The best variant is registered in the local MLflow Model Registry as `los-pred-los`:
```bash
uv run python -m los_pred.train
```

Compare algorithms (LightGBM vs XGBoost) on a fixed target treatment. Logged to a
separate `los-pred-algo` experiment; prints a winner but does not register:
```bash
uv run python -m los_pred.compare_algos                 # default: log1p
uv run python -m los_pred.compare_algos --treatment l2
```

Inspect runs, metrics, and registered models in the MLflow UI:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Practice the promotion lifecycle (Candidate → Staging → Production, mapped to
MLflow 3 aliases `@candidate` → `@staging` → `@champion`):
```bash
uv run python -m los_pred.registry --assign champion --version 2
uv run python -m los_pred.registry --show
```
Downstream code then loads the **live model by alias**, not a version number, so a
promotion needs no code change (see `docs/model_registry_note.md`):
```python
import mlflow.lightgbm
model = mlflow.lightgbm.load_model("models:/los-pred-los@champion")
```

Residual analysis of the registered model (bias, LOS-17 ceiling effect,
heteroscedasticity, and documented limitations):
```bash
jupyter notebook notebooks/04_posthoc_analysis.ipynb
```

> **Note — MLflow artifacts are local-only.** `mlflow.db`, `mlruns/`, and `mlartifacts/`
> are gitignored. The tracking DB and model artifacts are reproducible by re-running
> `train.py`, so they're not committed to git.

**4. Serving (Phase 3)**
```bash
uv add fastapi uvicorn
```

Serve the live `@champion` model as a real-time API. It loads the model (via the
flavor-agnostic `mlflow.pyfunc` interface) plus the fitted transform params
persisted on its run, so raw admission records are featurised with the exact same
`features.transform` used in training (no training-serving skew):
```bash
uv run uvicorn los_pred.app:app --reload --port 8000
```

- **Interactive docs / try it out:** http://127.0.0.1:8000/docs
- **Endpoints:** `POST /predict` (raw admission JSON → predicted LOS days),
  `GET /health` (serving model version), `GET /metrics` (prediction volume).

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
  -d '{"rcount":"1","gender":"M","facid":"B","hematocrit":12.0,"neutrophils":9.0,"sodium":138.0,"glucose":100.0,"bloodureanitro":12.0,"creatinine":1.0,"bmi":29.0,"pulse":70,"dialysisrenalendstage":0,"asthma":0,"irondef":1,"pneum":0,"substancedependence":0,"psychologicaldisordermajor":0,"depress":0,"psychother":0,"fibrosisandother":0,"malnutrition":0,"hemo":0}'
# -> {"predicted_los_days": 4.09}
```

**5. Tests**
```bash
uv add --dev httpx        # for the FastAPI TestClient (serving tests)
uv run pytest -q          # feature + training + serving pipelines
```
