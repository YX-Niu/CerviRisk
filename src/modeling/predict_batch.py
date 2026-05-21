from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from src.features.inference import build_inference_features, load_metadata


MODEL_FILES = {
    "1yr": "xgb_cin2_1yr.pkl",
    "3yr": "xgb_cin2_3yr.pkl",
    "5yr": "xgb_cin2_5yr.pkl",
}

CLINICAL_COLUMNS = [
    "age",
    "region",
    "visit_index",
    "hpv_test_result",
    "hpv_genotype",
    "hrhpv_positive",
    "cytology_result",
    "histology_result",
    "persistent_hrhpv",
    "n_previous_screens",
    "time_since_last_screen",
    "ever_had_abnormal_cyto",
    "ever_had_hrHPV",
]


def add_triage_fields(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    high_grade_cytology = predictions.get("cytology_result", "").isin(["ASC-H", "HSIL", "AGC"])
    hpv16_or_18 = predictions.get("hpv_genotype", "").isin(["hpv16", "hpv18", "multiple_hr"])
    hrhpv = predictions.get("hrhpv_positive", False).astype(bool)
    risk_3yr = predictions["risk_cin2_3yr"]
    risk_5yr = predictions["risk_cin2_5yr"]

    urgent = (risk_3yr >= 0.60) | (risk_5yr >= 0.75) | (high_grade_cytology & hpv16_or_18)
    high = (risk_3yr >= 0.40) | (hpv16_or_18 & hrhpv) | (high_grade_cytology & hrhpv)
    medium = (risk_3yr >= 0.20) | hrhpv

    predictions["priority"] = "routine"
    predictions.loc[medium, "priority"] = "medium"
    predictions.loc[high, "priority"] = "high"
    predictions.loc[urgent, "priority"] = "urgent"

    predictions["recommended_action"] = "Routine screening interval"
    predictions.loc[medium, "recommended_action"] = "Repeat HPV/cytology follow-up"
    predictions.loc[high, "recommended_action"] = "Clinician review within monthly triage"
    predictions.loc[urgent, "recommended_action"] = "Colposcopy or specialist review"

    reasons = []
    for row in predictions.itertuples(index=False):
        row_reasons = []
        if getattr(row, "risk_cin2_3yr") >= 0.60:
            row_reasons.append("3-year risk >= 60%")
        elif getattr(row, "risk_cin2_3yr") >= 0.40:
            row_reasons.append("3-year risk >= 40%")
        elif getattr(row, "risk_cin2_3yr") >= 0.20:
            row_reasons.append("3-year risk >= 20%")
        if getattr(row, "risk_cin2_5yr") >= 0.75:
            row_reasons.append("5-year risk >= 75%")
        if getattr(row, "hpv_genotype", "") in {"hpv16", "hpv18", "multiple_hr"}:
            row_reasons.append("high-risk HPV genotype")
        if getattr(row, "cytology_result", "") in {"ASC-H", "HSIL", "AGC"}:
            row_reasons.append("high-grade cytology")
        reasons.append("; ".join(row_reasons) if row_reasons else "low model risk and routine findings")
    predictions["review_reason"] = reasons
    return predictions


def predict_batch(input_path: Path, output_path: Path, metadata_path: Path = PROCESSED_DIR / "metadata.json") -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    records = pd.read_parquet(input_path)
    features = build_inference_features(records, metadata)

    output_columns = ["person_id", "screening_date"] + [col for col in CLINICAL_COLUMNS if col in records.columns]
    predictions = records[output_columns].copy()
    for window, filename in MODEL_FILES.items():
        model = joblib.load(MODEL_DIR / filename)
        predictions[f"risk_cin2_{window}"] = model.predict_proba(features)[:, 1]
    predictions = add_triage_fields(predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    predictions.describe(include="all").to_csv(REPORT_DIR / "prediction_summary.csv")
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch CIN2+ risk predictions for a monthly screening batch.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/predictions/latest_predictions.parquet"))
    parser.add_argument("--metadata", type=Path, default=PROCESSED_DIR / "metadata.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_batch(args.input, args.output, args.metadata)
    print(f"Saved batch predictions: {args.output}")
    print("rows:", len(predictions))
    print(predictions[["risk_cin2_1yr", "risk_cin2_3yr", "risk_cin2_5yr"]].describe().to_string())


if __name__ == "__main__":
    main()
