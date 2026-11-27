"""Save / load the pre-computed summary tables, with a version stamp.

Written by `db_app/build_summary.py`, read by the Cohort Summary tab. The files hold
aggregates only (no patient rows), so they can ship with the app without the raw data.

`summary_meta.json` is the stamp: a fingerprint of the source data plus enough context
(row count, date range, split cutoff, git commit) to say which data the tables describe.
At start-up the app compares the stamp with the current CSV and warns if they differ.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from services.summary import SummaryTables
from settings import APP_DIR, DATA_CSV, SPLIT_CUTOFF, SUMMARY_DIR

CONTINUOUS_FILE = "summary_continuous.csv"
CATEGORICAL_FILE = "summary_categorical.csv"
LOS_FILE = "summary_los.csv"
META_FILE = "summary_meta.json"
SCHEMA_VERSION = 3          # bump when the columns / layout of the saved tables change

_TEXT_COLUMNS = {"By": str, "Group": str, "Variable": str, "Level": str}   # keep "0", "5+" as text


# --- version stamp -----------------------------------------------------------
def data_fingerprint(path: Path) -> str:
    """SHA-256 of the source file (read in chunks, so any file size is fine)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=APP_DIR,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_meta(df: pd.DataFrame, source: Path = DATA_CSV) -> dict:
    """The stamp for tables built from `df` (loaded from `source`)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "data_sha256": data_fingerprint(source),
        "n_rows": int(len(df)),
        "first_visit": f"{df['vdate'].min():%Y-%m-%d}",
        "last_visit": f"{df['vdate'].max():%Y-%m-%d}",
        "split_cutoff": SPLIT_CUTOFF,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }


# --- save / load ---------------------------------------------------------------
def save_summary_tables(tables: SummaryTables, meta: dict, out_dir: Path = SUMMARY_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.continuous.to_csv(out_dir / CONTINUOUS_FILE, index=False)
    tables.categorical.to_csv(out_dir / CATEGORICAL_FILE, index=False)
    tables.los.to_csv(out_dir / LOS_FILE, index=False)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    load_summary_tables.cache_clear()
    summary_is_stale.cache_clear()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build the summary tables first: "
            "`uv run --group dashboard python db_app/build_summary.py`"
        )
    return path


@lru_cache(maxsize=4)
def load_summary_tables(out_dir: Path = SUMMARY_DIR) -> SummaryTables:
    """Read once per process: a few hundred rows, however large the source data."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(_require(out_dir / name), dtype=_TEXT_COLUMNS)

    return SummaryTables(read(CONTINUOUS_FILE), read(CATEGORICAL_FILE), read(LOS_FILE))


def load_summary_meta(out_dir: Path = SUMMARY_DIR) -> dict:
    return json.loads(_require(out_dir / META_FILE).read_text())


@lru_cache(maxsize=4)
def summary_is_stale(out_dir: Path = SUMMARY_DIR, source: Path = DATA_CSV) -> bool | None:
    """True if the source data changed since the tables were built; None if it can't be checked
    (e.g. the raw CSV is not shipped with the app)."""
    if not source.exists():
        return None
    return data_fingerprint(source) != load_summary_meta(out_dir)["data_sha256"]
