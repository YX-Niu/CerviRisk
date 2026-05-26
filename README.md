# CerviRisk

An end-to-end ML system for cervical cancer screening risk prediction. Simulates longitudinal HPV/cytology registry data, builds leakage-aware historical features, trains CIN2+ risk models across three time horizons (1/3/5 years), and serves predictions via FastAPI with a Streamlit clinician dashboard.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic data
python src/ingestion/simulator.py

# 2. Train models
python src/training_pipeline.py

# 3. Launch dashboards
streamlit run src/dashboard/app.py              # clinical triage
streamlit run src/dashboard/app_presentation.py # pipeline demo

# 4. Serve predictions
uvicorn src.serving.app:app --reload
```

## Pipeline

```
simulator.py  →  preprocess.py  →  train.py  →  app.py (FastAPI)
  (data)           (features)      (models)      (serving)
```

All configuration lives in `src/config.py`: cohort size, split years, XGBoost parameters, clustering settings, drift thresholds.

## Models

Three models are benchmarked on the primary 3-year CIN2+ target; the best is saved as `models/best_cin2_3yr.pkl`. XGBoost models are also saved for all three horizons and used by the prediction API.

To run a hyperparameter search:

```bash
python src/modeling/tune.py --target outcome_cin2_3yr --n-iter 20
```

Copy the best parameters from `reports/best_xgb_params.json` into `src/config.py` and retrain.

## Repository Layout

```
src/
  config.py               Central configuration
  training_pipeline.py    Orchestrates preprocess → train → cluster → explain
  ingestion/              Data simulation and batch ingestion
  features/               Feature engineering (Pandas + Polars)
  modeling/               Training, tuning, clustering, SHAP explanations
  serving/                FastAPI prediction endpoint
  dashboard/              Streamlit dashboards
  monitoring/             Data drift detection (KS + chi-square)

data/                     Generated data (gitignored)
models/                   Trained model artifacts (gitignored)
reports/                  Metrics, plots, drift reports (gitignored)
```

## Scalability

The simulator supports large cohorts via `--n-women` and `--chunk-size`. A Polars preprocessing path and year-partitioned Parquet output are available for datasets that don't fit in memory.

## Note

All records are synthetic and for demonstration only. Not for clinical use.
