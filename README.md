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

**3. Training (Phase 2)**
```bash
uv add mlflow scikit-learn xgboost
uv run python -m los_pred.train
```
Runs controlled experiments on target transformations and logs results to MLflow.
