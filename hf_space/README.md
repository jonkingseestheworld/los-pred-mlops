---
title: LOS Prediction API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: FastAPI service predicting hospital length of stay (days)
---

# Hospital length-of-stay prediction API

A FastAPI service that serves a LightGBM regression model predicting how many
days a patient will stay in hospital, from one admission record.

This is a demo for an MLOps learning project. The model is trained on the public
Kaggle hospital length-of-stay dataset (synthetic data), not on real patients,
and must not be used for clinical decisions.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check; reports the model version being served |
| GET | `/metrics` | Prediction count since the last restart |
| POST | `/predict` | Predict length of stay for one admission record |
| GET | `/docs` | Interactive API docs (Swagger UI) |

## Example

```bash
curl -X POST https://<space-url>/predict \
  -H 'content-type: application/json' \
  -d '{"rcount":"1","gender":"M","facid":"B","hematocrit":12.0,"neutrophils":9.0,
       "sodium":138.0,"glucose":100.0,"bloodureanitro":12.0,"creatinine":1.0,
       "bmi":29.0,"pulse":70,"dialysisrenalendstage":0,"asthma":0,"irondef":1,
       "pneum":0,"substancedependence":0,"psychologicaldisordermajor":0,
       "depress":0,"psychother":0,"fibrosisandother":0,"malnutrition":0,"hemo":0}'
# -> {"predicted_los_days": 4.09}
```

On the free tier the Space sleeps when idle, so the first request after a pause
can take a minute while it wakes up.
