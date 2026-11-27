"""Pre-compute the Cohort Summary tables the dashboard displays.

Run from the repo root whenever the data (or the variable list) changes:

    uv run --group dashboard python db_app/build_summary.py

Reads the raw CSV once and writes small aggregate tables, plus a version stamp
(`summary_meta.json`), to `db_app/data/`. The Cohort Summary tab then never needs
the raw rows.
"""

from __future__ import annotations

from services.data import load_dataset
from services.summary import build_summary_tables
from services.summary_store import build_meta, save_summary_tables
from settings import DATA_CSV, SUMMARY_DIR


def main() -> None:
    df = load_dataset()
    tables = build_summary_tables(df)
    meta = build_meta(df, DATA_CSV)
    save_summary_tables(tables, meta)
    print(
        f"{len(df):,} rows -> {len(tables.continuous)} continuous + "
        f"{len(tables.categorical)} categorical + {len(tables.los)} length-of-stay count rows "
        f"in {SUMMARY_DIR}\n"
        f"data fingerprint {meta['data_sha256'][:12]}, commit {meta['git_commit']}"
    )


if __name__ == "__main__":
    main()
