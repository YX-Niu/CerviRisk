from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODEL_DIR, MODELING, PROCESSED_DIR, REPORT_DIR

PRIMARY_TARGET = MODELING.primary_target


def _metrics(y_true: pd.Series, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, prob) if y_true.nunique() == 2 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {"auc": auc, "sensitivity": sensitivity, "specificity": specificity}


def _xgb_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=MODELING.xgb_n_estimators,
        max_depth=MODELING.xgb_max_depth,
        learning_rate=MODELING.xgb_learning_rate,
        subsample=MODELING.xgb_subsample,
        colsample_bytree=MODELING.xgb_colsample_bytree,
        eval_metric="logloss",
        random_state=MODELING.random_seed,
        n_jobs=MODELING.xgb_n_jobs,
        scale_pos_weight=scale_pos_weight,
    )


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet").sort_values("screening_date")


def train_for_target(target: str, feature_columns: list[str]) -> tuple[object, pd.DataFrame]:
    train = load_split("train")
    validation = load_split("validation")
    train_val = pd.concat([train, validation], ignore_index=True).sort_values("screening_date")

    x_train = train[feature_columns]
    y_train = train[target]
    x_val = validation[feature_columns]
    y_val = validation[target]
    scale_pos_weight = max(float((y_train == 0).sum() / max((y_train == 1).sum(), 1)), 1.0)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=18,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=4,
        ),
        "xgboost": _xgb_model(scale_pos_weight),
    }

    rows = []
    tscv = TimeSeriesSplit(n_splits=4)
    for model_name, model in models.items():
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val), start=1):
            fold_model = clone(model)
            x_fold_train = train_val.iloc[train_idx][feature_columns]
            y_fold_train = train_val.iloc[train_idx][target]
            x_fold_val = train_val.iloc[val_idx][feature_columns]
            y_fold_val = train_val.iloc[val_idx][target]
            if y_fold_train.nunique() < 2 or y_fold_val.nunique() < 2:
                continue
            fold_model.fit(x_fold_train, y_fold_train)
            prob = fold_model.predict_proba(x_fold_val)[:, 1]
            rows.append({"target": target, "model": model_name, "fold": fold, **_metrics(y_fold_val, prob)})

        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        rows.append({"target": target, "model": model_name, "fold": "validation", **_metrics(y_val, val_prob)})
        models[model_name] = model

    results = pd.DataFrame(rows)
    best_name = (
        results.loc[results["fold"].eq("validation")]
        .sort_values("auc", ascending=False)
        .iloc[0]["model"]
    )
    return models[best_name], results


def save_xgb_target_model(target: str, feature_columns: list[str]) -> XGBClassifier:
    train = pd.concat([load_split("train"), load_split("validation")], ignore_index=True)
    x = train[feature_columns]
    y = train[target]
    scale_pos_weight = max(float((y == 0).sum() / max((y == 1).sum(), 1)), 1.0)
    model = _xgb_model(scale_pos_weight)
    model.fit(x, y)
    return model


def save_dask_xgb_target_model(target: str, feature_columns: list[str]) -> XGBClassifier:
    try:
        import dask.dataframe as dd
        from dask.distributed import Client, LocalCluster
        from xgboost.dask import DaskXGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Dask-XGBoost mode requires dask[dataframe], distributed, and xgboost with dask support."
        ) from exc

    train = pd.concat([load_split("train"), load_split("validation")], ignore_index=True)
    y = train[target]
    scale_pos_weight = max(float((y == 0).sum() / max((y == 1).sum(), 1)), 1.0)

    scheduler_address = os.getenv("DASK_SCHEDULER_ADDRESS")
    if scheduler_address:
        cluster = None
        client = Client(scheduler_address)
    else:
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=False,
            dashboard_address=None,
            worker_dashboard_address=None,
        )
        client = Client(cluster)
    try:
        x_dd = dd.from_pandas(train[feature_columns], npartitions=4)
        y_dd = dd.from_pandas(y, npartitions=4)
        dask_model = DaskXGBClassifier(
            n_estimators=MODELING.xgb_n_estimators,
            max_depth=MODELING.xgb_max_depth,
            learning_rate=MODELING.xgb_learning_rate,
            subsample=MODELING.xgb_subsample,
            colsample_bytree=MODELING.xgb_colsample_bytree,
            eval_metric="logloss",
            random_state=MODELING.random_seed,
            scale_pos_weight=scale_pos_weight,
        )
        dask_model.client = client
        dask_model.fit(x_dd, y_dd)

        local_model = _xgb_model(scale_pos_weight)
        bootstrap_x = train[feature_columns].head(2)
        local_model.fit(bootstrap_x, np.array([0, 1]))
        with tempfile.NamedTemporaryFile(suffix=".json") as model_file:
            dask_model.get_booster().save_model(model_file.name)
            local_model.load_model(model_file.name)
        return local_model
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]

    all_results = []
    best_model, primary_results = train_for_target(PRIMARY_TARGET, feature_columns)
    all_results.append(primary_results)
    joblib.dump(best_model, MODEL_DIR / "best_cin2_3yr.pkl")

    use_dask_xgb = os.getenv("CERVIRISK_USE_DASK_XGB", "0") == "1"
    if use_dask_xgb:
        print("Dask-XGBoost mode enabled via CERVIRISK_USE_DASK_XGB=1")

    for target in metadata["target_columns"]:
        xgb = save_dask_xgb_target_model(target, feature_columns) if use_dask_xgb else save_xgb_target_model(target, feature_columns)
        suffix = target.replace("outcome_", "")
        joblib.dump(xgb, MODEL_DIR / f"xgb_{suffix}.pkl")
        if target == PRIMARY_TARGET:
            joblib.dump(xgb, MODEL_DIR / "xgb_cin2_3yr.pkl")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

    xgb_primary = joblib.load(MODEL_DIR / "xgb_cin2_3yr.pkl")
    importances = pd.DataFrame(
        {"feature": feature_columns, "importance": xgb_primary.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    test = load_split("test")
    test_prob = xgb_primary.predict_proba(test[feature_columns])[:, 1]
    print("Primary target:", PRIMARY_TARGET)
    print(results.to_string(index=False))
    print("Test metrics for saved XGBoost:", _metrics(test[PRIMARY_TARGET], test_prob))
    print("Top feature importances:")
    print(importances.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
