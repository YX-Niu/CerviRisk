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


def predict_batch(input_path: Path, output_path: Path, metadata_path: Path = PROCESSED_DIR / "metadata.json") -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    records = pd.read_parquet(input_path)
    features = build_inference_features(records, metadata)

    predictions = records[["person_id", "screening_date"]].copy()
    for window, filename in MODEL_FILES.items():
        model = joblib.load(MODEL_DIR / filename)
        predictions[f"risk_cin2_{window}"] = model.predict_proba(features)[:, 1]

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
