# Dashboard — Shiny for Python

A standalone demo dashboard for the hospital length-of-stay data. Tabs:

| Tab | Module | What it does |
|---|---|---|
| Description | `modules/description/` | Shell with two sub-tabs: **Background** (`background.py` — dataset source, how to use the app, data dictionary) and **Data cleaning** (`data_cleaning.py` — every cleaning step applied before modelling, with live row counts, before/after ranges, the train/test split and learned parameters) |
| Group summary | `modules/group_summary.py` | Mean / SD / median / Q1 / Q3 for continuous variables and n (%) for categorical ones, optionally split by facility, gender, readmissions or train/test |
| Forecaster | `modules/forecaster.py` | Placeholder (prediction + explanation to come) |

A full-screen loading page (`components/loading.py`, `www/loading.*`) shows until
Shiny has connected and rendered its first outputs.

## Run

From the repo root (needs `data/LengthOfStay.csv`):

```bash
uv run --group dashboard shiny run --reload db_app/app.py
```

## Layout

```
db_app/
  app.py            assembles the page shell + one module per tab (no logic)
  settings.py       paths, branding, constants (the app's config)
  modules/          one Shiny module per tab: <tab>_ui(id) / <tab>_server(id, ...)
    description/    a tab with sub-tabs = a package: __init__.py (shell) + one module per sub-tab
  components/       reusable UI pieces (loading screen, placeholder card, HTML table)
  services/         pure Python, no Shiny: data loading, variable metadata, summaries,
                    cleaning audit (runs the real `los_pred.features` pipeline and measures it)
  www/              CSS / JS
```

Rules of thumb: modules only wire reactive inputs to outputs; any computation
that is repeated or worth testing lives in `services/` (tested in
`tests/test_db_app_summary.py` and `tests/test_db_app_cleaning.py`). Add a variable in `services/variables.py` and it
appears in the data dictionary and the Group summary pickers.

## Customising

- Names on the loading page, title and source links: `settings.py`.
- Minimum loading-screen time: `MIN_MS` in `www/loading.js`.
- Data path can be overridden with the `LOS_DATA_CSV` environment variable.
