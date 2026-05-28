from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import INCOMING_DIR, MONITORING, RAW_PATH, REPORT_DIR, SPLITS
from src.ingestion.simulator import SimulatorConfig, generate_screening_data


def categorical_drift(reference: pd.Series, current: pd.Series) -> dict:
    ref_counts = reference.astype(str).value_counts()
    cur_counts = current.astype(str).value_counts()
    levels = sorted(set(ref_counts.index).union(cur_counts.index))
    table = np.array([[ref_counts.get(level, 0) for level in levels], [cur_counts.get(level, 0) for level in levels]])
    _, p_value, _, _ = chi2_contingency(table + 1)
    return {"test": "chi_square", "p_value": float(p_value), "drift_detected": bool(p_value < MONITORING.p_value_threshold)}


def numeric_drift(reference: pd.Series, current: pd.Series) -> dict:
    stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    return {"test": "ks", "statistic": float(stat), "p_value": float(p_value), "drift_detected": bool(p_value < MONITORING.p_value_threshold)}


def latest_incoming_batch() -> Path | None:
    if not INCOMING_DIR.exists():
        return None
    batches = sorted(INCOMING_DIR.glob("batch_date=*/screening_records.parquet"))
    return batches[-1] if batches else None


def load_current_batch(path: Path | None, inject_demo_drift: bool) -> pd.DataFrame:
    if path is None:
        batch = generate_screening_data(
            SimulatorConfig(
                n_women=MONITORING.fallback_n_women,
                start_year=MONITORING.fallback_start_year,
                end_year=MONITORING.fallback_end_year,
                seed=MONITORING.fallback_seed,
            )
        )
    else:
        batch = pd.read_parquet(path)
    if inject_demo_drift:
        batch = batch.copy()
        batch.loc[batch.sample(frac=MONITORING.demo_drift_fraction, random_state=7).index, "hpv_genotype"] = "other_hr"
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a new CerviRisk batch against the training reference distribution.")
    parser.add_argument("--current-batch", type=Path, default=None)
    parser.add_argument("--inject-demo-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(RAW_PATH)
    train_dates = pd.to_datetime(reference["screening_date"]).dt.year.between(SPLITS.train_start_year, SPLITS.train_end_year)
    reference = reference.loc[train_dates]

    current_batch_path = args.current_batch or latest_incoming_batch()
    new_batch = load_current_batch(current_batch_path, inject_demo_drift=args.inject_demo_drift)

    report = {
        "current_batch_path": str(current_batch_path) if current_batch_path else "simulated_in_memory",
        "age": numeric_drift(reference["age"], new_batch["age"]),
        "hpv_genotype": categorical_drift(reference["hpv_genotype"], new_batch["hpv_genotype"]),
        "cytology_result": categorical_drift(reference["cytology_result"], new_batch["cytology_result"]),
    }
    report["overall_drift_detected"] = any(
        item["drift_detected"] for item in report.values() if isinstance(item, dict)
    )

    output_path = REPORT_DIR / "drift_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    print("\n--- Drift Monitoring Summary ---")
    for feat in ["age", "hpv_genotype", "cytology_result"]:
        status = "DRIFT" if report[feat]["drift_detected"] else "✅ OK"
        p_val = report[feat]["p_value"]
        print(f"{feat:15} | {status} | p-value: {p_val:.4f}")

    if report["overall_drift_detected"]:
        print("WARNING: drift threshold exceeded for at least one monitored feature.")
        print("\nWARNING: Significant data drift detected! Model results may be unreliable.")


if __name__ == "__main__":
    main()
