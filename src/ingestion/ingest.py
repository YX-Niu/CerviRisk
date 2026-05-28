from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import INCOMING_DIR, INGESTION, RAW_PATH
from src.db import DB_PATH, insert_screening_records, query_batch_by_date


def load_batch_from_raw(batch_date: date, raw_path: Path = RAW_PATH) -> pd.DataFrame | None:
    """Slice the month's records from the historical raw dataset — same patients as training."""
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    raw["screening_date"] = pd.to_datetime(raw["screening_date"])
    mask = (raw["screening_date"].dt.year == batch_date.year) & \
           (raw["screening_date"].dt.month == batch_date.month)
    batch = raw.loc[mask].copy()
    if batch.empty:
        return None
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
    parser = argparse.ArgumentParser(description="Extract a monthly CerviRisk screening batch from raw data.")
    parser.add_argument("--batch-date", type=date.fromisoformat, default=INGESTION.batch_date)
    parser.add_argument("--output-dir", type=Path, default=INCOMING_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load the monthly batch from the SQLite database instead.",
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_db:
        batch = load_batch_from_db(args.batch_date, db_path=args.db_path)
        if batch is None:
            print(f"No records in DB for {args.batch_date}, falling back to raw data.")
            batch = load_batch_from_raw(args.batch_date, args.raw_path)
        else:
            print(f"Loaded {len(batch):,} rows from DB for {args.batch_date}")
    else:
        batch = load_batch_from_raw(args.batch_date, args.raw_path)

    if batch is None or batch.empty:
        print(f"No records found for {args.batch_date} in {args.raw_path}. Nothing written.")
        return

    batch_dir = args.output_dir / f"batch_date={args.batch_date.isoformat()}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / "screening_records.parquet"
    batch.to_parquet(batch_path, index=False)
    write_manifest(batch_path, batch, cadence="monthly")

    print(f"Saved monthly batch: {batch_path}")
    print("rows:", len(batch))
    print("date range:", batch["screening_date"].min().date(), "to", batch["screening_date"].max().date())


if __name__ == "__main__":
    main()
