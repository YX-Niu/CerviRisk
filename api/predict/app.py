from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data/processed"
MODEL_DIR = ROOT / "models"


class ScreeningRecord(BaseModel):
    age: float
    birth_year: int | None = None
    visit_index: int = 0
    region: str = "central"
    hpv_vaccinated: bool = False
    smoking_status: str = "never"
    immunosuppressed: bool = False
    parity: int = 0
    hpv_test_result: str = "negative"
    hpv_genotype: str = "negative"
    hrhpv_positive: bool = False
    cytology_result: str = "NILM"
    colposcopy_performed: bool = False
    histology_result: str = "none"
    cin2plus_detected: bool = False
    treatment_performed: bool = False
    persistent_hrhpv: bool = False
    n_previous_screens: int = 0
    time_since_last_screen: float = 999.0
    ever_had_abnormal_cyto: bool = False
    ever_had_hrHPV: bool = Field(False, alias="ever_had_hrHPV")
    ever_had_cin1plus: bool = False
    ever_had_cin2plus: bool = False

    class Config:
        populate_by_name = True


app = FastAPI(title="CerviRisk API", version="0.1.0")

metadata: dict[str, Any] | None = None
models: dict[str, Any] = {}


def load_artifacts() -> None:
    global metadata, models
    if metadata is None:
        metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    if not models:
        models = {
            "1yr": joblib.load(MODEL_DIR / "xgb_cin2_1yr.pkl"),
            "3yr": joblib.load(MODEL_DIR / "xgb_cin2_3yr.pkl"),
            "5yr": joblib.load(MODEL_DIR / "xgb_cin2_5yr.pkl"),
        }


def make_feature_row(record: ScreeningRecord) -> pd.DataFrame:
    load_artifacts()
    assert metadata is not None
    data = record.model_dump(by_alias=True)
    if data["birth_year"] is None:
        data["birth_year"] = 2026 - int(round(float(data["age"])))

    row = {col: float(data.get(col, 0.0)) for col in metadata["numeric_columns"]}
    for col, levels in metadata["category_levels"].items():
        value = str(data.get(col, ""))
        for level in levels:
            row[f"{col}_{level}"] = 1.0 if value == level else 0.0

    frame = pd.DataFrame([row])
    return frame.reindex(columns=metadata["feature_columns"], fill_value=0.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(record: ScreeningRecord) -> dict[str, Any]:
    load_artifacts()
    features = make_feature_row(record)
    risks = {window: float(model.predict_proba(features)[:, 1][0]) for window, model in models.items()}
    return {"risk_probabilities": risks, "model": "xgboost", "target": "CIN2+"}
