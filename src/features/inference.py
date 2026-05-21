from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR


def load_metadata(metadata_path: Path = PROCESSED_DIR / "metadata.json") -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def add_inference_defaults(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    if "n_previous_screens" not in records.columns and "visit_index" in records.columns:
        records["n_previous_screens"] = records["visit_index"]
    if "time_since_last_screen" not in records.columns:
        records["time_since_last_screen"] = 999.0
    if "ever_had_abnormal_cyto" not in records.columns:
        records["ever_had_abnormal_cyto"] = records.get("cytology_result", pd.Series(index=records.index, dtype=str)).ne("NILM")
    if "ever_had_hrHPV" not in records.columns:
        records["ever_had_hrHPV"] = records.get("hrhpv_positive", False)
    if "ever_had_cin1plus" not in records.columns:
        records["ever_had_cin1plus"] = records.get("histology_result", pd.Series(index=records.index, dtype=str)).isin(["CIN1", "CIN2", "CIN3", "cancer"])
    if "ever_had_cin2plus" not in records.columns:
        records["ever_had_cin2plus"] = records.get("histology_result", pd.Series(index=records.index, dtype=str)).isin(["CIN2", "CIN3", "cancer"])
    return records


def build_inference_features(records: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    records = add_inference_defaults(records)

    rows = pd.DataFrame(index=records.index)
    for col in metadata["numeric_columns"]:
        rows[col] = pd.to_numeric(records[col], errors="coerce").fillna(0.0) if col in records.columns else 0.0

    for col, levels in metadata["category_levels"].items():
        values = records[col].astype(str) if col in records.columns else pd.Series("", index=records.index)
        for level in levels:
            rows[f"{col}_{level}"] = values.eq(level).astype(float)

    return rows.reindex(columns=metadata["feature_columns"], fill_value=0.0)
