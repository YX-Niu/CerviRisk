from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, RAW_PATH

def scan_input(path: Path) -> pl.LazyFrame:
    if path.is_dir():
        return pl.scan_parquet(str(path / "**/*.parquet"), hive_partitioning=True)
    return pl.scan_parquet(path)


def build_polars_features(input_path: Path) -> pl.LazyFrame:
    abnormal_cyto = pl.col("cytology_result") != "NILM"
    cin1plus = pl.col("histology_result").is_in(["CIN1", "CIN2", "CIN3", "cancer"])
    cin2plus = pl.col("histology_result").is_in(["CIN2", "CIN3", "cancer"])
    hrhpv = pl.col("hrhpv_positive").cast(pl.Int8)

    return (
        scan_input(input_path)
        .with_columns(pl.col("screening_date").cast(pl.Datetime))
        .sort(["person_id", "screening_date"])
        .with_columns(
            pl.cum_count("person_id").over("person_id").sub(1).alias("n_previous_screens"),
            (
                pl.col("screening_date").diff().over("person_id").dt.total_days() / 365.25
            )
            .fill_null(999.0)
            .alias("time_since_last_screen"),
            abnormal_cyto.cast(pl.Int8).cum_max().shift(1).over("person_id").fill_null(0).alias("ever_had_abnormal_cyto"),
            hrhpv.cum_max().shift(1).over("person_id").fill_null(0).alias("ever_had_hrHPV"),
            cin1plus.cast(pl.Int8).cum_max().shift(1).over("person_id").fill_null(0).alias("ever_had_cin1plus"),
            cin2plus.cast(pl.Int8).cum_max().shift(1).over("person_id").fill_null(0).alias("ever_had_cin2plus"),
            pl.col("screening_date").dt.year().alias("screening_year"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polars feature-engineering path for large CerviRisk datasets.")
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "features_polars.parquet")
    parser.add_argument("--partition-by-year", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features = build_polars_features(args.input).collect(engine="streaming")
    if args.partition_by_year:
        features.write_parquet(args.output, partition_by="screening_year")
    else:
        features.write_parquet(args.output)
    print(f"Saved {args.output}")
    print("shape:", features.shape)
    print("missing values:", features.null_count().sum_horizontal().item())


if __name__ == "__main__":
    main()
