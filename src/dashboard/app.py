from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PREDICTION_ROOT = Path("data/predictions")
RAW_RECORDS_PATH = Path("data/raw/screening_records.parquet")
DRIFT_REPORT = Path("reports/drift_report.json")
PRIORITY_ORDER = ["urgent", "high", "medium", "routine"]
HISTORY_COLUMNS = [
    "screening_date",
    "age",
    "hpv_test_result",
    "hpv_genotype",
    "hrhpv_positive",
    "cytology_result",
    "histology_result",
    "persistent_hrhpv",
    "cin2plus_detected",
    "treatment_performed",
]


def latest_prediction_file() -> Path | None:
    files = sorted(PREDICTION_ROOT.glob("batch_date=*/predictions.parquet"))
    return files[-1] if files else None


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "screening_date" in df.columns:
        df["screening_date"] = pd.to_datetime(df["screening_date"]).dt.date
    return df


def load_drift_report() -> dict:
    if not DRIFT_REPORT.exists():
        return {}
    return json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))


@st.cache_data
def load_history(path: str = str(RAW_RECORDS_PATH)) -> pd.DataFrame:
    raw_path = Path(path)
    if not raw_path.exists():
        return pd.DataFrame()
    columns = ["person_id"] + HISTORY_COLUMNS
    df = pd.read_parquet(raw_path, columns=columns)
    df["screening_date"] = pd.to_datetime(df["screening_date"])
    return df.sort_values(["person_id", "screening_date"])


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def risk_column_config() -> dict:
    return {
        "risk_cin2_1yr": st.column_config.ProgressColumn(
            "1-year CIN2+ risk",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "risk_cin2_3yr": st.column_config.ProgressColumn(
            "3-year CIN2+ risk",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "risk_cin2_5yr": st.column_config.ProgressColumn(
            "5-year CIN2+ risk",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    }


def to_display_percent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["risk_cin2_1yr", "risk_cin2_3yr", "risk_cin2_5yr"]:
        if col in df:
            df[col] = df[col] * 100
    return df


def priority_rank(series: pd.Series) -> pd.Series:
    ranks = {name: index for index, name in enumerate(PRIORITY_ORDER)}
    return series.map(ranks).fillna(len(PRIORITY_ORDER))


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        priorities = st.multiselect("Priority", PRIORITY_ORDER, default=["urgent", "high"])
        min_risk = st.slider("Minimum 3-year risk", 0.0, 1.0, 0.20, 0.05)

        genotype_options = sorted(df["hpv_genotype"].dropna().astype(str).unique()) if "hpv_genotype" in df else []
        selected_genotypes = st.multiselect("HPV genotype", genotype_options)

        cytology_options = sorted(df["cytology_result"].dropna().astype(str).unique()) if "cytology_result" in df else []
        selected_cytology = st.multiselect("Cytology", cytology_options)

    filtered = df[df["priority"].isin(priorities)] if "priority" in df else df
    filtered = filtered[filtered["risk_cin2_3yr"] >= min_risk]
    if selected_genotypes and "hpv_genotype" in filtered:
        filtered = filtered[filtered["hpv_genotype"].astype(str).isin(selected_genotypes)]
    if selected_cytology and "cytology_result" in filtered:
        filtered = filtered[filtered["cytology_result"].astype(str).isin(selected_cytology)]
    return filtered


def show_drift_status(report: dict) -> None:
    if not report:
        st.info("No drift report found yet.")
        return

    status = "Detected" if report.get("overall_drift_detected") else "Not detected"
    st.metric("Data Drift", status)
    drift_rows = []
    for feature in ["age", "hpv_genotype", "cytology_result"]:
        if feature in report:
            item = report[feature]
            drift_rows.append(
                {
                    "feature": feature,
                    "test": item.get("test"),
                    "p_value": item.get("p_value"),
                    "drift_detected": item.get("drift_detected"),
                }
            )
    if drift_rows:
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)


def show_patient_detail(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No patients match the selected filters.")
        return

    df = df.sort_values(["priority_rank", "risk_cin2_3yr"], ascending=[True, False])
    labels = df["person_id"].astype(str) + " | " + df["priority"].astype(str) + " | 3yr " + df["risk_cin2_3yr"].map(format_percent)
    selected_label = st.selectbox("Patient review", labels.tolist())
    row = df.iloc[labels.tolist().index(selected_label)]

    st.subheader("Model Risk Reference")
    left, middle, right = st.columns(3)
    left.metric("1-year risk", format_percent(float(row["risk_cin2_1yr"])))
    middle.metric("3-year risk", format_percent(float(row["risk_cin2_3yr"])))
    right.metric("5-year risk", format_percent(float(row["risk_cin2_5yr"])))
    st.caption("Risk values are model estimates for CIN2+ and should support, not replace, clinical judgement.")

    st.subheader("Clinical Context")
    detail_columns = [
        "person_id",
        "screening_date",
        "age",
        "hpv_test_result",
        "hpv_genotype",
        "cytology_result",
        "histology_result",
        "persistent_hrhpv",
        "n_previous_screens",
        "time_since_last_screen",
        "priority",
        "recommended_action",
        "review_reason",
    ]
    available = [col for col in detail_columns if col in row.index]
    st.table(pd.DataFrame({"field": available, "value": [row[col] for col in available]}))

    st.subheader("Screening History")
    history = load_history()
    if history.empty:
        st.info("No historical raw records found.")
        return

    patient_history = history[history["person_id"].eq(row["person_id"])].copy()
    if patient_history.empty:
        st.info("No previous screening history found for this patient.")
        return

    patient_history["screening_date"] = patient_history["screening_date"].dt.date
    previous = patient_history[pd.to_datetime(patient_history["screening_date"]) < pd.to_datetime(row["screening_date"])]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Previous Screens", f"{len(previous):,}")
    h2.metric("Prior hrHPV+", f"{int(previous['hrhpv_positive'].sum()):,}")
    h3.metric("Prior Abnormal Cytology", f"{int(previous['cytology_result'].ne('NILM').sum()):,}")
    h4.metric("Prior CIN2+", f"{int(previous['cin2plus_detected'].sum()):,}")

    display_history = patient_history.sort_values("screening_date", ascending=False).head(8)
    st.dataframe(display_history[HISTORY_COLUMNS], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="CerviRisk Monthly Triage", layout="wide")
    st.title("CerviRisk Monthly Triage")

    default_path = latest_prediction_file()
    if default_path is None:
        st.error("No prediction file found. Run `python src/prediction_pipeline.py --batch-date YYYY-MM-DD` first.")
        return

    with st.sidebar:
        prediction_files = sorted(PREDICTION_ROOT.glob("batch_date=*/predictions.parquet"), reverse=True)
        selected_path = st.selectbox("Prediction batch", prediction_files, format_func=lambda p: str(p))

    df = load_predictions(str(selected_path))
    if "priority" not in df:
        st.error("Prediction file does not include triage fields. Rerun `src/prediction_pipeline.py` with the latest code.")
        return

    df["priority_rank"] = priority_rank(df["priority"])
    filtered = apply_filters(df)

    total = len(df)
    urgent = int(df["priority"].eq("urgent").sum())
    high_or_urgent = int(df["priority"].isin(["urgent", "high"]).sum())
    mean_risk = float(df["risk_cin2_3yr"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly Records", f"{total:,}")
    c2.metric("Urgent Reviews", f"{urgent:,}")
    c3.metric("High or Urgent", f"{high_or_urgent:,}")
    c4.metric("Mean 3-year Risk", format_percent(mean_risk))

    tab_queue, tab_patient, tab_monitor = st.tabs(["Review Queue", "Patient Detail", "Monitoring"])

    with tab_queue:
        queue_columns = [
            "priority",
            "person_id",
            "screening_date",
            "age",
            "hpv_genotype",
            "cytology_result",
            "risk_cin2_1yr",
            "risk_cin2_3yr",
            "risk_cin2_5yr",
            "recommended_action",
            "review_reason",
        ]
        available = [col for col in queue_columns if col in filtered.columns]
        queue = filtered.sort_values(["priority_rank", "risk_cin2_3yr"], ascending=[True, False])[available]
        st.dataframe(
            to_display_percent(queue),
            use_container_width=True,
            hide_index=True,
            column_config=risk_column_config(),
        )

    with tab_patient:
        show_patient_detail(filtered)

    with tab_monitor:
        show_drift_status(load_drift_report())


if __name__ == "__main__":
    main()
