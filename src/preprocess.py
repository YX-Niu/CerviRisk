from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/screening_records.parquet")
PROCESSED_DIR = Path("data/processed")

TARGETS = ["outcome_cin2_1yr", "outcome_cin2_3yr", "outcome_cin2_5yr"]
ID_COLUMNS = ["person_id", "screening_date"]
CATEGORICAL_COLUMNS = [
    "region",
    "smoking_status",
    "hpv_test_result",
    "hpv_genotype",
    "cytology_result",
    "histology_result",
]
NUMERIC_COLUMNS = [
    "age",
    "birth_year",
    "visit_index",
    "hpv_vaccinated",
    "immunosuppressed",
    "parity",
    "hrhpv_positive",
    "colposcopy_performed",
    "cin2plus_detected",
    "treatment_performed",
    "persistent_hrhpv",
    "n_previous_screens",
    "time_since_last_screen",
    "ever_had_abnormal_cyto",
    "ever_had_hrHPV",
    "ever_had_cin1plus",
    "ever_had_cin2plus",
]


def add_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["person_id", "screening_date"]).copy()
    df["screening_date"] = pd.to_datetime(df["screening_date"])
    cin2_dates = (
        df.loc[df["cin2plus_detected"], ["person_id", "screening_date"]]
        .groupby("person_id")["screening_date"]
        .apply(list)
        .to_dict()
    )

    for years in [1, 3, 5]:
        labels = []
        for row in df[["person_id", "screening_date"]].itertuples(index=False):
            future_limit = row.screening_date + pd.DateOffset(years=years)
            has_future_cin2 = any(row.screening_date < dt <= future_limit for dt in cin2_dates.get(row.person_id, []))
            labels.append(int(has_future_cin2))
        df[f"outcome_cin2_{years}yr"] = labels
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["person_id", "screening_date"]).copy()
    grouped = df.groupby("person_id", group_keys=False)

    df["n_previous_screens"] = grouped.cumcount()
    previous_dates = grouped["screening_date"].shift(1)
    df["time_since_last_screen"] = (df["screening_date"] - previous_dates).dt.days.div(365.25)
    df["time_since_last_screen"] = df["time_since_last_screen"].fillna(999.0)

    abnormal_cyto = df["cytology_result"].ne("NILM")
    cin1plus = df["histology_result"].isin(["CIN1", "CIN2", "CIN3", "cancer"])
    cin2plus = df["histology_result"].isin(["CIN2", "CIN3", "cancer"])
    df["ever_had_abnormal_cyto"] = grouped.apply(lambda g: abnormal_cyto.loc[g.index].shift(fill_value=False).cummax()).astype(int).to_numpy()
    df["ever_had_hrHPV"] = grouped.apply(lambda g: df.loc[g.index, "hrhpv_positive"].shift(fill_value=False).cummax()).astype(int).to_numpy()
    df["ever_had_cin1plus"] = grouped.apply(lambda g: cin1plus.loc[g.index].shift(fill_value=False).cummax()).astype(int).to_numpy()
    df["ever_had_cin2plus"] = grouped.apply(lambda g: cin2plus.loc[g.index].shift(fill_value=False).cummax()).astype(int).to_numpy()
    return df


def build_feature_frame(df: pd.DataFrame, category_levels: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(float)

    if category_levels is None:
        category_levels = {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL_COLUMNS}

    parts = [df[ID_COLUMNS + TARGETS + NUMERIC_COLUMNS]]
    for col in CATEGORICAL_COLUMNS:
        cat = pd.Categorical(df[col].astype(str), categories=category_levels[col])
        dummies = pd.get_dummies(cat, prefix=col, dtype=float)
        parts.append(dummies.set_index(df.index))

    features = pd.concat(parts, axis=1)
    return features, category_levels


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    raw = raw.copy()
    raw["screening_date"] = pd.to_datetime(raw["screening_date"])
    raw = add_outcome_labels(raw)
    raw = add_history_features(raw)
    return build_feature_frame(raw)


def split_by_time(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(features["screening_date"])
    return {
        "train": features.loc[dates.dt.year.between(2010, 2018)].copy(),
        "validation": features.loc[dates.dt.year.between(2019, 2020)].copy(),
        "test": features.loc[dates.dt.year.between(2021, 2022)].copy(),
        "future_holdout": features.loc[dates.dt.year >= 2023].copy(),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(RAW_PATH)
    features, category_levels = build_dataset(raw)
    feature_columns = [c for c in features.columns if c not in ID_COLUMNS + TARGETS]

    features.to_parquet(PROCESSED_DIR / "features_all.parquet", index=False)
    for name, split in split_by_time(features).items():
        split.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    metadata = {
        "feature_columns": feature_columns,
        "target_columns": TARGETS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "category_levels": category_levels,
        "split_policy": {
            "train": "2010-2018",
            "validation": "2019-2020",
            "test": "2021-2022",
            "future_holdout": "2023+",
        },
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Processed rows:", features.shape)
    for target in TARGETS:
        print(target, "rate:", float(np.round(features[target].mean(), 4)))
    print("missing values:", int(features.isna().sum().sum()))


if __name__ == "__main__":
    main()
