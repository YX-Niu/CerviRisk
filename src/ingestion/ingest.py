from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import INCOMING_DIR, INGESTION
from src.ingestion.simulator import SimulatorConfig, generate_screening_data


def simulate_monthly_batch(batch_date: date, n_records: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg = SimulatorConfig(n_women=n_records, start_year=batch_date.year, end_year=batch_date.year, seed=seed)
    batch = generate_screening_data(cfg).groupby("person_id", as_index=False).tail(1).copy()

    days_in_month = pd.Period(batch_date, freq="M").days_in_month
    random_days = rng.integers(1, days_in_month + 1, size=len(batch))
    batch["screening_date"] = [
        pd.Timestamp(year=batch_date.year, month=batch_date.month, day=int(day))
        for day in random_days
    ]
    batch["ingestion_batch_date"] = pd.Timestamp(batch_date)
    return batch.sort_values(["screening_date", "person_id"]).reset_index(drop=True)


def write_manifest(batch_path: Path, batch: pd.DataFrame, cadence: str) -> None:
    manifest_path = INCOMING_DIR / "manifest.jsonl"
    entry = {
        "batch_date": str(batch["ingestion_batch_date"].iloc[0].date()),
        "cadence": cadence,
        "path": str(batch_path),
        "rows": int(len(batch)),
        "min_screening_date": str(batch["screening_date"].min().date()),
        "max_screening_date": str(batch["screening_date"].max().date()),
    }
    existing = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    manifest_path.write_text(existing + json.dumps(entry) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a monthly incoming CerviRisk screening batch.")
    parser.add_argument("--batch-date", type=date.fromisoformat, default=INGESTION.batch_date)
    parser.add_argument("--n-records", type=int, default=INGESTION.n_records)
    parser.add_argument("--seed", type=int, default=INGESTION.seed)
    parser.add_argument("--output-dir", type=Path, default=INCOMING_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = simulate_monthly_batch(args.batch_date, args.n_records, args.seed)
    batch_dir = args.output_dir / f"batch_date={args.batch_date.isoformat()}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / "screening_records.parquet"
    batch.to_parquet(batch_path, index=False)
    write_manifest(batch_path, batch, cadence="monthly")

    print(f"Saved monthly ingestion batch: {batch_path}")
    print("rows:", len(batch))
    print("date range:", batch["screening_date"].min().date(), "to", batch["screening_date"].max().date())


if __name__ == "__main__":
    main()
