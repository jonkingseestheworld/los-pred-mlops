# Dashboard — Shiny for Python

A standalone demo dashboard for the hospital length-of-stay data. Tabs:

| Tab | Module | What it does |
|---|---|---|
| Description | `modules/description/` | Shell with two sub-tabs: **Background** (`background.py` — dataset source, how to use the app, data dictionary) and **Data processing** (`data_cleaning.py` — short educational walk-through of the dataset, the time-based train/test split and the processing steps; static text) |
| Cohort Summary | `modules/group_summary.py` | Mean / SD / median / Q1 / Q3 for continuous variables and n (%) for categorical ones, optionally split by facility, gender, readmissions or train/test, plus a length-of-stay histogram for the whole population (admissions per day, independent of "Group by", with a "View as table" fallback). Reads **pre-computed tables** (see below), not the raw rows |
| Forecaster | `modules/forecaster.py` | Placeholder (prediction + explanation to come) |

A full-screen loading page (`components/loading.py`, `www/loading.*`) shows until
Shiny has connected and rendered its first outputs.

## Run

From the repo root (needs `data/LengthOfStay.csv`):

```bash
uv run --group dashboard shiny run --reload db_app/app.py
```

## Pre-computed summary tables

The Cohort Summary tab does not summarise patient rows on every request. The tables for
every variable under every grouping (including the admissions-per-day counts the histogram is drawn from) are built once and saved in `data/` (aggregates only,
committed with the app), so the tab's cost does not grow with the number of patients and
the app does not need the raw data for it.

```bash
uv run --group dashboard python db_app/build_summary.py   # rerun whenever the data changes
```

`data/summary_meta.json` is the **version stamp**: SHA-256 of the source CSV, row count,
first/last visit date, split cutoff, build time, git commit and a schema version. The tab
shows it in the sidebar, and if the raw CSV is present and no longer matches the stamp it
shows a "tables may be out of date" warning. Only the pre-computed groupings can be shown
(medians and quartiles can't be combined after the fact); adding a grouping means rebuilding.

## Layout

```
db_app/
  app.py            assembles the page shell + one module per tab (no logic)
  settings.py       paths, branding, constants (the app's config)
  build_summary.py  one-off script: raw CSV -> data/ summary tables + version stamp
  data/             pre-computed Cohort Summary tables (3 CSVs) + summary_meta.json
  modules/          one Shiny module per tab: <tab>_ui(id) / <tab>_server(id, ...)
    description/    a tab with sub-tabs = a package: __init__.py (shell) + one module per sub-tab
  components/       reusable UI pieces (loading screen, placeholder card, HTML table)
  services/         pure Python, no Shiny: data loading, variable metadata, summaries,
                    summary tables (build / save / load + stamp), the SVG histogram (`charts.py`, no plotting library),
                    cleaning audit (runs the real `los_pred.features` pipeline and measures it)
  www/              CSS / JS
```

Rules of thumb: modules only wire reactive inputs to outputs; any computation
that is repeated or worth testing lives in `services/` (tested in
`tests/test_db_app_summary.py`, `tests/test_db_app_summary_store.py`, `tests/test_db_app_charts.py` and `tests/test_db_app_cleaning.py`). Add a variable in `services/variables.py` and it
appears in the data dictionary and the Cohort Summary tables (then rebuild the summary tables). The tab currently shows every variable; its variable pickers are commented out in `modules/group_summary.py`.

## Customising

- Names on the loading page, title and source links: `settings.py`.
- Minimum loading-screen time: `MIN_MS` in `www/loading.js`.
- Data path can be overridden with the `LOS_DATA_CSV` environment variable.
