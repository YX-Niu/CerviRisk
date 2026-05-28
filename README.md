# CerviRisk

An end-to-end ML system for cervical cancer screening risk prediction. Simulates longitudinal HPV/cytology registry data, builds leakage-aware historical features, trains CIN2+ risk models across three time horizons (1/3/5 years), and serves predictions via a Streamlit triage dashboard and FastAPI endpoint.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic data
python src/ingestion/simulator.py

# 2. Train models
python src/training_pipeline.py

# 3. Run monthly prediction pipeline (auto-opens triage dashboard)
python src/prediction_pipeline.py
```

## Pipeline

```
simulator.py          →  training_pipeline.py              →  prediction_pipeline.py
  (synthetic data)         preprocess → train                   ingest batch
                           cluster → explain                     batch predict
                                                                 drift monitor
                                                                 → dashboard
```

All configuration lives in `src/config.py`: cohort size, split years, XGBoost parameters, clustering settings, drift thresholds.

## Training Pipeline

```bash
python src/training_pipeline.py [--skip-clustering] [--skip-explain]
```

Steps: feature engineering → XGBoost training (1/3/5-year horizons) → cluster profiling → SHAP explanations.

## Prediction Pipeline

```bash
python src/prediction_pipeline.py [--batch-date YYYY-MM-DD] [--inject-demo-drift]
```

Ingests the monthly screening batch, runs batch risk prediction, checks for data drift against the training baseline, then launches the Streamlit triage dashboard at `http://localhost:8501`. Press Ctrl+C to stop.

## FastAPI Endpoint

```bash
uvicorn src.serving.app:app --reload
```

Single-record prediction at `POST /predict`. See `src/serving/test_request.py` for an example request.

## Hyperparameter Tuning

Tuning is separate from the main training pipeline:

```bash
python src/modeling/tune.py --target outcome_cin2_3yr --n-iter 20
```

Copy the best parameters from `reports/best_xgb_params.json` into `src/config.py` and retrain.

## Repository Layout

```
src/
  config.py               Central configuration
  training_pipeline.py    Orchestrates preprocess → train → cluster → explain
  prediction_pipeline.py  Monthly batch predict → drift monitor → dashboard
  ingestion/              Data simulation and batch ingestion
  features/               Feature engineering and prediction feature builder
  modeling/               Training, tuning, clustering, SHAP explanations
  serving/                FastAPI single-record prediction endpoint
  dashboard/              Streamlit triage dashboard (app.py)
  monitoring/             Data drift detection (KS + chi-square)

data/                     Generated data (gitignored)
models/                   Trained model artifacts (gitignored)
reports/                  Metrics, plots, drift reports (gitignored)
```

## Scalability

The simulator supports large cohorts via `--n-women` and `--chunk-size`. A Polars preprocessing path and year-partitioned Parquet output are available for datasets that don't fit in memory.

---

> All records are synthetic and for demonstration only. Not for clinical use.
