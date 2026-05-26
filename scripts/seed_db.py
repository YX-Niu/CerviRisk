"""Seed the CerviRisk SQLite database from the existing raw Parquet file.

Usage:
    python scripts/seed_db.py                    # uses data/raw/screening_records.parquet
    python scripts/seed_db.py --raw data/raw/screening_records.parquet --db data/cervirisk.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db import DB_PATH, db_stats, init_db, insert_screening_records
from src.config import RAW_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the CerviRisk SQLite DB from raw Parquet.")
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw.exists():
        print(f"Raw file not found: {args.raw}")
        print("Run `python src/ingestion/simulator.py` first to generate the data.")
        sys.exit(1)

    print(f"Reading {args.raw} ...")
    df = pd.read_parquet(args.raw)
    print(f"  {len(df):,} rows, {df['person_id'].nunique():,} patients")

    init_db(args.db)
    n = insert_screening_records(df, db_path=args.db)
    print(f"Inserted {n:,} visit rows into {args.db}")

    stats = db_stats(args.db)
    print(f"\nDB summary:")
    print(f"  patients : {stats['patients']:,}")
    print(f"  visits   : {stats['visits']:,}")
    print(f"  date range: {stats['earliest_visit']} → {stats['latest_visit']}")


if __name__ == "__main__":
    main()
