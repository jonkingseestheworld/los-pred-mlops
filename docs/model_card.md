# Model Card — Hospital Length-of-Stay Predictor (`los-pred-los`)

> **Status:** learning / portfolio project on a **synthetic** dataset. **Not for
> real clinical use.** See _Ethical & clinical considerations_ below.

## Overview

| | |
|---|---|
| **Task** | Regression — predict hospital **length of stay (LOS)** in days |
| **Model** | LightGBM gradient-boosted trees (`LGBMRegressor`) |
| **Target treatment** | Trained on `log1p(LOS)`; predictions inverted with `expm1` and floor-clipped at 1 day |
| **Registered as** | `los-pred-los` in the MLflow Model Registry; live version tagged `@champion` (v4) |
| **Output range** | ~1–17 days (dataset caps LOS at 17) |
| **Owner** | JoKL · single-maintainer learning project |

## Intended use

- **Primary:** aggregate / capacity planning — e.g. forecasting total bed-days
  across a ward, where predictions are summed over many patients.
- **Reasonable:** point estimates for **routine, typical admissions** (LOS ≈ 1–9
  days), which cover ~95%+ of cases and are predicted accurately.

### Out of scope / misuse

- **Individual long-stay / high-risk triage without clinical oversight.** The
  model systematically **under-predicts long stays** (see _Limitations_), so
  using it to flag or rank the longest-stay patients will under-flag exactly the
  cases of interest.
- **Any hard threshold on the raw prediction** (e.g. "alert if predicted LOS > 10")
  — long-stayers get pulled below the threshold, concentrating false negatives in
  the highest-severity patients.
- **Real patient care.** Trained on a synthetic dataset; not validated on real
  clinical data, and not a medical device.

## Training data

- **Source:** Microsoft synthetic *LengthOfStay* dataset (via Kaggle).
- **Split:** **temporal**, not random — train ≤ 2012-10, test = later months
  (Nov/Dec 2012); the single 2013-01-01 stub is merged into Dec. This simulates
  predicting future admissions from past data.
- **Preprocessing (fit on TRAIN only, to avoid training–serving skew):**
  impossible `glucose ≤ 0` imputed to the train median; `neutrophils` and
  `bloodureanitro` winsorized at the train 99th percentile; `facid` one-hot over
  the train vocabulary with an explicit `other` bucket for unseen facilities;
  near-constant `respiration` and leakage/ID columns dropped.
- **Engineered features:** `comorbidity_count` (sum of 11 binary comorbidity
  flags), `isMale`, numeric `rcount` (`"5+" → 5`).

## Performance

Evaluated on the held-out temporal test set (n = 16,953), scored in **real days**:

| Metric | Value |
|---|---|
| MAE | **0.355 days** |
| RMSE | ~0.52 days |
| Mean residual (bias) | **+0.027 days** (≈ unbiased in aggregate) |
| Naive baseline (predict-the-mean) MAE | 1.914 days |

The model beats the naive baseline by ~81%. (Note: the synthetic data is
near-deterministic, so absolute errors are unusually low; expect larger errors on
real data.)

## Key drivers (explainability)

From SHAP (`TreeExplainer`) — see `notebooks/05_explainability.ipynb`:

- **Top drivers:** `rcount` (readmission count) and `comorbidity_count`, both
  pushing predictions **up** — consistent with clinical intuition that sicker
  patients (more prior admissions, more comorbidities) stay longer. Lab values
  contribute smaller, consistent amounts.
- SHAP values are computed in **log-space** (the model trains on `log1p(LOS)`);
  direction and ranking are interpretable, magnitudes are log-days.

## Limitations

- **Under-prediction of long stays (the headline limitation).** Mean residual is
  ~0 for short stays but rises monotonically with LOS — up to **+2.4 days** at
  LOS 16 (positive = under-prediction). The global near-zero bias **hides** this,
  because long stays are rare (LOS ≥ 10 is only ~1.6% of the test set).
- **Root cause is data, not the model.** SHAP shows the model *does* recognise
  long-stayers: `rcount`'s mean SHAP is ~0 across all patients but **+0.42** for
  the long-stay subset (`comorbidity_count` ~0 → +0.17). The features fire in the
  right direction — the model simply cannot **calibrate the magnitude** because
  the training tail is scarce and hard-capped at LOS 17.
- **Heteroscedasticity:** residual spread grows with the prediction (std ~0.36
  for short stays → ~1.16 for predicted LOS > 10).
- **Synthetic data:** patterns are cleaner than real hospital data; real-world
  performance and the long tail would differ (likely worse in the tail).

## How to read a prediction (guidance)

- Trust point estimates for **routine, short stays**.
- For patients with **many readmissions / multiple comorbidities**, treat the
  prediction as a **conservative lower bound** and apply clinical judgment — the
  true stay may be longer than predicted.
- Prefer the model for **aggregate** forecasting over **individual** high-stakes
  decisions.

## Improvement levers (not yet applied)

- Report **uncertainty** (prediction intervals), especially for long stays.
- Reframe the decision as **classification** (e.g. `P(LOS > 7 days)`) to sidestep
  the ceiling.
- **Quantile / cost-sensitive** objective if under-prediction is the costly error.
- Acquire more (uncapped) long-stay examples.

## Ethical & clinical considerations

This is a **learning artifact on synthetic data**, not a validated clinical tool.
Any real deployment would require validation on real, representative data,
fairness analysis across patient subgroups, clinical safety review, and ongoing
monitoring. The documented long-stay bias is safety-relevant: it fails in the
direction of under-estimating resource needs for the sickest patients.

## References (in this repo)

- `notebooks/04_posthoc_analysis.ipynb` — residual analysis, ceiling effect
- `notebooks/05_explainability.ipynb` — SHAP global/local/long-stay analysis
- `src/los_pred/train.py` — training + registration
- `docs/model_registry_note.md` — registry / promotion notes
