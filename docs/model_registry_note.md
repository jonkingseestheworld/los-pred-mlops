# What does the Model Registry replace in a typical academic / consultancy workflow?

Short answer: it replaces the pile of **ad-hoc conventions** that a registry-free
project uses to answer four questions — *which model is this? is it any good?
which one is live? how was it made?* — and turns them into one queryable,
versioned, lineage-linked source of truth.

## The typical academic / consultancy workflow

Without a registry, model handoff is done by hand:

- **Naming conventions in filenames** — `model_final.pkl`, `model_final_v2.pkl`,
  `model_FINAL_use_this_one.pkl`. The version lives in the filename and in
  people's memory.
- **Artifacts scattered across storage** — a pickle emailed to a colleague, or
  dropped in SharePoint / Dropbox / a shared drive, detached from the code that
  produced it.
- **Metrics in a slide deck or spreadsheet** — the results table lives in a
  PowerPoint or an Excel sheet, separate from the model file it describes; the
  link between "this number" and "this artifact" is manual and easily broken.
- **"Which one is live?" is tribal knowledge** — someone says *"use the one from
  the March run"*; promotion means renaming a file or sending an email.
- **Reproducibility is best-effort** — the exact data, parameters, and code
  behind a saved model are usually not recorded alongside it, so re-creating a
  result months later is hard or impossible.

## What the Model Registry replaces it with

| Ad-hoc practice | Model Registry equivalent |
|---|---|
| `model_final_v2.pkl` naming | **Versioned model name** (`los-pred-los` v1, v2, v3 …) — versions assigned automatically, never by filename. |
| Emailed / shared-drive pickle | **Central store** with each version's artifact in one place, addressable as `models:/los-pred-los/2`. |
| Metrics in a deck / spreadsheet | **Metrics + params linked to the version** via its source run — the number and the artifact are the same object. |
| "Use the March model" (tribal knowledge) | **Aliases** (`@champion`, `@staging`, `@candidate`) — a movable pointer says which version plays which role; serving loads `models:/los-pred-los@champion` and never hard-codes a version. |
| Rename-the-file promotion | **Explicit, auditable transition** — reassigning an alias (+ status tags) is a recorded action, not a silent file swap. |
| Best-effort reproducibility | **Lineage** — every version links back to the run that made it (code, params, data reference), so results are reconstructable. |

## Why it matters here (MLOps framing)

The registry is what lets *downstream code depend on a role, not a file*. In this
project, `04_posthoc_analysis.ipynb`, `monitor.py`, and (in Phase 3) the serving
API all ask for `los-pred-los@champion` — so when a better model is trained and
promoted, everything picks it up with no code change. That decoupling — plus the
audit trail of *who promoted what, when* — is exactly what the informal
consultancy workflow lacks, and it's the foundation the serving and deployment
phases build on.

> **MLflow 3.x note:** the older *stages* (Staging / Production) were removed;
> the modern equivalent is **aliases + tags**, which this project uses. See
> `src/los_pred/registry.py` to practice the Candidate → Staging → Production
> (`@candidate` → `@staging` → `@champion`) transition.
