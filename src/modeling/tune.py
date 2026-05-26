from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import MODELING, PROCESSED_DIR, REPORT_DIR


PARAM_GRID = {
    "n_estimators": [160, 220, 260, 340, 420],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.025, 0.04, 0.06, 0.08],
    "subsample": [0.75, 0.86, 0.95],
    "colsample_bytree": [0.75, 0.86, 0.95],
    "min_child_weight": [1, 3, 6],
    "reg_lambda": [1.0, 2.0, 5.0],
}


def load_training_frame() -> tuple[pd.DataFrame, list[str]]:
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    frame = pd.concat([train, validation], ignore_index=True).sort_values("screening_date")
    return frame, metadata["feature_columns"]


def evaluate_params(frame: pd.DataFrame, feature_columns: list[str], target: str, params: dict) -> dict:
    y = frame[target]
    scale_pos_weight = max(float((y == 0).sum() / max((y == 1).sum(), 1)), 1.0)
    splitter = TimeSeriesSplit(n_splits=4)
    fold_aucs = []

    for train_idx, val_idx in splitter.split(frame):
        x_train = frame.iloc[train_idx][feature_columns]
        y_train = frame.iloc[train_idx][target]
        x_val = frame.iloc[val_idx][feature_columns]
        y_val = frame.iloc[val_idx][target]
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=MODELING.random_seed,
            n_jobs=MODELING.xgb_n_jobs,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_val)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_val, prob)))

    return {
        **params,
        "target": target,
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with time-series validation.")
    parser.add_argument("--target", default=MODELING.primary_target)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=MODELING.random_seed)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "tuning_results.csv")
    parser.add_argument("--best-params-output", type=Path, default=REPORT_DIR / "best_xgb_params.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame, feature_columns = load_training_frame()
    sampled_params = list(ParameterSampler(PARAM_GRID, n_iter=args.n_iter, random_state=args.seed))
    rows = [evaluate_params(frame, feature_columns, args.target, params) for params in sampled_params]

    results = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    best_params = {
        key: results.iloc[0][key].item() if hasattr(results.iloc[0][key], "item") else results.iloc[0][key]
        for key in PARAM_GRID
    }
    args.best_params_output.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print(f"Saved tuning results: {args.output}")
    print(f"Saved best parameters: {args.best_params_output}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
