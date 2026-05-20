from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class SimulationDefaults:
    n_women: int = 22000
    start_year: int = 2010
    end_year: int = 2024
    seed: int = 42


@dataclass(frozen=True)
class IngestionDefaults:
    batch_date: date = date(2024, 1, 1)
    n_records: int = 750
    seed: int = 202401


@dataclass(frozen=True)
class SplitDefaults:
    train_start_year: int = 2010
    train_end_year: int = 2018
    validation_start_year: int = 2019
    validation_end_year: int = 2020
    test_start_year: int = 2021
    test_end_year: int = 2022
    future_holdout_start_year: int = 2023


@dataclass(frozen=True)
class ModelingDefaults:
    primary_target: str = "outcome_cin2_3yr"
    random_seed: int = 42
    xgb_n_estimators: int = 260
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.045
    xgb_subsample: float = 0.86
    xgb_colsample_bytree: float = 0.86
    xgb_n_jobs: int = 4


@dataclass(frozen=True)
class ClusteringDefaults:
    n_clusters: int = 4
    eligibility_cutoff: str = "2021-12-31"
    random_seed: int = 42


@dataclass(frozen=True)
class MonitoringDefaults:
    fallback_n_women: int = 900
    fallback_start_year: int = 2024
    fallback_end_year: int = 2025
    fallback_seed: int = 2026
    p_value_threshold: float = 0.01
    demo_drift_fraction: float = 0.18


RAW_PATH = Path("data/raw/screening_records.parquet")
INCOMING_DIR = Path("data/incoming")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

SIMULATION = SimulationDefaults()
INGESTION = IngestionDefaults()
SPLITS = SplitDefaults()
MODELING = ModelingDefaults()
CLUSTERING = ClusteringDefaults()
MONITORING = MonitoringDefaults()
