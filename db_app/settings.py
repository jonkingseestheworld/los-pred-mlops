"""Central configuration for the dashboard: paths, branding and constants.

Every other file imports from here, so changing a path or a name is a one-file
edit (the Shiny equivalent of a `global.R` / `config.yml`).
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# --- paths -----------------------------------------------------------------
# Raw Microsoft LOS data (gitignored — the Description tab says where to get it).
DATA_CSV = Path(os.getenv("LOS_DATA_CSV", PROJECT_ROOT / "data" / "LengthOfStay.csv"))

# Pre-computed Cohort Summary tables: aggregates only, committed with the app and
# rebuilt by `db_app/build_summary.py`.
SUMMARY_DIR = APP_DIR / "data"

# Temporal split used at training time (train <= cutoff month, test = later).
SPLIT_CUTOFF = "2012-10"

# --- branding (placeholder names; swap freely) -----------------------------
APP_TITLE = "Evaluating Healthcare Resource Usage"
ORG_ABBR = "Demo app"
ORG_NAME = ""
TEAM_NAME = "Exploring & Predicting Hospital Length-of-Stay"
LOADING_MESSAGE = "The app is loading — please wait . . ."
# Served from db_app/img/ (see `static_assets` in app.py); used in the navbar and loading page.
LOGO_SRC = "img/los.png"

# --- source ----------------------------------------------------------------
KAGGLE_URL = "https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft"
MICROSOFT_DOC_URL = "https://microsoft.github.io/r-server-hospital-length-of-stay/input_data.html"
