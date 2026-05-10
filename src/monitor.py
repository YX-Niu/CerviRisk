from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_simulator import SimulatorConfig, generate_screening_data


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
RAW_PATH = Path("data/raw/screening_records.parquet")


def categorical_drift(reference: pd.Series, current: pd.Series) -> dict:
    ref_counts = reference.astype(str).value_counts()
    cur_counts = current.astype(str).value_counts()
    levels = sorted(set(ref_counts.index).union(cur_counts.index))
    table = np.array([[ref_counts.get(level, 0) for level in levels], [cur_counts.get(level, 0) for level in levels]])
    _, p_value, _, _ = chi2_contingency(table + 1)
    return {"test": "chi_square", "p_value": float(p_value), "drift_detected": bool(p_value < 0.01)}


def numeric_drift(reference: pd.Series, current: pd.Series) -> dict:
    stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    return {"test": "ks", "statistic": float(stat), "p_value": float(p_value), "drift_detected": bool(p_value < 0.01)}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(RAW_PATH)
    train_dates = pd.to_datetime(reference["screening_date"]).dt.year.between(2010, 2018)
    reference = reference.loc[train_dates]

    new_batch = generate_screening_data(SimulatorConfig(n_women=900, start_year=2024, end_year=2025, seed=2026))
    new_batch.loc[new_batch.sample(frac=0.18, random_state=7).index, "hpv_genotype"] = "other_hr"

    report = {
        "age": numeric_drift(reference["age"], new_batch["age"]),
        "hpv_genotype": categorical_drift(reference["hpv_genotype"], new_batch["hpv_genotype"]),
        "cytology_result": categorical_drift(reference["cytology_result"], new_batch["cytology_result"]),
    }
    report["overall_drift_detected"] = any(item["drift_detected"] for item in report.values())

    output_path = REPORT_DIR / "drift_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["overall_drift_detected"]:
        print("WARNING: drift threshold exceeded for at least one monitored feature.")


if __name__ == "__main__":
    main()
