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
from src.db import DB_PATH, insert_screening_records, query_batch_by_date
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


def load_batch_from_db(batch_date: date, db_path: Path) -> pd.DataFrame | None:
    df = query_batch_by_date(batch_date.isoformat(), db_path=db_path)
    return df if not df.empty else None


def write_manifest(batch_path: Path, batch: pd.DataFrame, cadence: str) -> None:
    manifest_path = INCOMING_DIR / "manifest.jsonl"
    raw_date = batch["ingestion_batch_date"].iloc[0]
    batch_date_str = str(pd.Timestamp(raw_date).date()) if raw_date is not None else "unknown"
    entry = {
        "batch_date": batch_date_str,
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
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load the monthly batch from the SQLite database instead of re-simulating.",
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_db:
        batch = load_batch_from_db(args.batch_date, db_path=args.db_path)
        if batch is None:
            print(f"No records found in DB for batch_date={args.batch_date}. Falling back to simulation.")
            batch = simulate_monthly_batch(args.batch_date, args.n_records, args.seed)
            insert_screening_records(batch, db_path=args.db_path)
            print(f"Inserted simulated batch into {args.db_path}")
        else:
            print(f"Loaded {len(batch):,} rows from DB for batch_date={args.batch_date}")
    else:
        batch = simulate_monthly_batch(args.batch_date, args.n_records, args.seed)
        insert_screening_records(batch, db_path=args.db_path)
        print(f"Simulated and wrote {len(batch):,} rows to DB")

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
