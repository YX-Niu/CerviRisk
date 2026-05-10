from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/xgb_cin2_3yr.pkl")
REPORT_DIR = Path("reports")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet").sort_values("screening_date")
    model = joblib.load(MODEL_PATH)

    sample = test.sample(min(600, len(test)), random_state=42)
    x = sample[feature_columns]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    shap.summary_plot(shap_values, x, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary_bar.png", dpi=180)
    plt.close()

    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    test = test.assign(risk_probability=probabilities)
    high_risk_idx = int(np.argmax(probabilities))
    low_risk_idx = int(np.argmin(probabilities))
    false_negative_pool = test[(test["outcome_cin2_3yr"] == 1) & (test["risk_probability"] < 0.5)]
    false_negative_idx = int(false_negative_pool.index[0]) if not false_negative_pool.empty else high_risk_idx

    cases = test.iloc[[high_risk_idx, low_risk_idx]].copy()
    if false_negative_idx in test.index:
        cases = pd.concat([cases, test.loc[[false_negative_idx]]], ignore_index=True)
    cases["case_type"] = ["high_risk", "low_risk", "possible_false_negative"][: len(cases)]
    cases[["case_type", "person_id", "screening_date", "risk_probability", "outcome_cin2_3yr"]].to_csv(
        REPORT_DIR / "shap_case_examples.csv", index=False
    )

    case_x = cases[feature_columns]
    case_shap = explainer.shap_values(case_x)
    for i, case_type in enumerate(cases["case_type"]):
        force = shap.force_plot(
            explainer.expected_value,
            case_shap[i, :],
            case_x.iloc[i, :],
            matplotlib=False,
        )
        shap.save_html(str(REPORT_DIR / f"shap_force_{case_type}.html"), force)

    print(f"Saved {REPORT_DIR / 'shap_summary_bar.png'}")
    print(cases[["case_type", "person_id", "risk_probability", "outcome_cin2_3yr"]].to_string(index=False))


if __name__ == "__main__":
    main()
