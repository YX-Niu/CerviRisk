from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PREDICTION_ROOT = Path("data/predictions")
RAW_RECORDS_PATH = Path("data/raw/screening_records.parquet")
DRIFT_REPORT = Path("reports/drift_report.json")
PRIORITY_ORDER = ["urgent", "high", "medium", "routine"]
PRIORITY_COLORS = {"urgent": "#d62728", "high": "#ff7f0e", "medium": "#1f77b4", "routine": "#2ca02c"}
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
HPV_COLORS = {
    "negative": "#2ca02c",
    "low_risk": "#aec7e8",
    "other_hr": "#ffbb78",
    "hpv18": "#ff7f0e",
    "hpv16": "#d62728",
    "multiple_hr": "#9467bd",
}


def latest_prediction_file() -> Path | None:
    files = sorted(PREDICTION_ROOT.glob("batch_date=*/predictions.parquet"))
    return files[-1] if files else None


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "screening_date" in df.columns:
        df["screening_date"] = pd.to_datetime(df["screening_date"]).dt.date
    return df


@st.cache_data
def load_drift_report() -> dict:
    if not DRIFT_REPORT.exists():
        return {}
    return json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))


@st.cache_data
def load_raw() -> pd.DataFrame:
    if not RAW_RECORDS_PATH.exists():
        return pd.DataFrame()
    columns = ["person_id"] + HISTORY_COLUMNS
    df = pd.read_parquet(RAW_RECORDS_PATH, columns=columns)
    df["screening_date"] = pd.to_datetime(df["screening_date"])
    return df.sort_values(["person_id", "screening_date"])


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def risk_column_config() -> dict:
    return {
        "risk_cin2_1yr": st.column_config.ProgressColumn(
            "1-year CIN2+ risk", format="%.1f%%", min_value=0, max_value=100,
        ),
        "risk_cin2_3yr": st.column_config.ProgressColumn(
            "3-year CIN2+ risk", format="%.1f%%", min_value=0, max_value=100,
        ),
        "risk_cin2_5yr": st.column_config.ProgressColumn(
            "5-year CIN2+ risk", format="%.1f%%", min_value=0, max_value=100,
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


# ══════════════════════════ TAB 1 — REVIEW QUEUE ═════════════════════════════

def show_review_queue(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    # Summary charts
    c_pie, c_hist, c_scatter = st.columns(3)

    with c_pie:
        counts = df["priority"].value_counts().reindex(PRIORITY_ORDER).fillna(0)
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.52,
            marker_colors=[PRIORITY_COLORS[p] for p in counts.index],
            textinfo="label+value",
        ))
        fig.update_layout(
            title="Priority Breakdown", showlegend=False,
            height=240, margin=dict(t=40, b=10, l=5, r=5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_hist:
        fig = px.histogram(
            df, x="risk_cin2_3yr", color="priority",
            color_discrete_map=PRIORITY_COLORS,
            category_orders={"priority": PRIORITY_ORDER},
            nbins=24, barmode="stack",
            labels={"risk_cin2_3yr": "3-year risk"},
            title="3-Year Risk Distribution",
        )
        fig.update_layout(
            showlegend=False, height=240, margin=dict(t=40, b=10),
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_scatter:
        fig = px.scatter(
            df, x="age", y="risk_cin2_3yr", color="priority",
            color_discrete_map=PRIORITY_COLORS,
            category_orders={"priority": PRIORITY_ORDER},
            opacity=0.55,
            labels={"age": "Age", "risk_cin2_3yr": "3-yr risk"},
            title="Age vs Risk",
        )
        fig.update_layout(
            showlegend=False, height=240, margin=dict(t=40, b=10),
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Filtered queue table
    st.markdown("---")
    queue_columns = [
        "priority", "person_id", "screening_date", "age",
        "hpv_genotype", "cytology_result",
        "risk_cin2_1yr", "risk_cin2_3yr", "risk_cin2_5yr",
        "recommended_action", "review_reason",
    ]
    available = [col for col in queue_columns if col in filtered.columns]
    queue = filtered.sort_values(
        ["priority_rank", "risk_cin2_3yr"], ascending=[True, False]
    )[available]
    st.dataframe(
        to_display_percent(queue),
        use_container_width=True,
        hide_index=True,
        column_config=risk_column_config(),
    )


# ══════════════════════════ TAB 2 — PATIENT DETAIL ═══════════════════════════

def show_patient_detail(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No patients match the selected filters.")
        return

    df = df.sort_values(["priority_rank", "risk_cin2_3yr"], ascending=[True, False])
    labels = (
        df["person_id"].astype(str)
        + " | " + df["priority"].astype(str)
        + " | 3yr " + df["risk_cin2_3yr"].map(format_percent)
    )
    selected_label = st.selectbox("Patient review", labels.tolist())
    row = df.iloc[labels.tolist().index(selected_label)]

    # Risk bar chart + clinical context side by side
    c_bar, c_tbl = st.columns(2)

    with c_bar:
        rv = {
            "1-year": float(row["risk_cin2_1yr"]),
            "3-year": float(row["risk_cin2_3yr"]),
            "5-year": float(row["risk_cin2_5yr"]),
        }
        fig = go.Figure(go.Bar(
            x=list(rv.keys()), y=list(rv.values()),
            marker_color=["#aec7e8", "#ff7f0e", "#d62728"],
            text=[f"{v:.1%}" for v in rv.values()],
            textposition="outside",
        ))
        fig.update_layout(
            title="CIN2+ Risk by Prediction Window",
            yaxis=dict(tickformat=".0%", range=[0, min(1.0, max(rv.values()) * 1.45)]),
            height=260, margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Model estimates only — should support, not replace, clinical judgement.")

    with c_tbl:
        st.subheader("Clinical Context")
        detail_columns = [
            "person_id", "screening_date", "age",
            "hpv_test_result", "hpv_genotype", "cytology_result",
            "histology_result", "persistent_hrhpv",
            "n_previous_screens", "time_since_last_screen",
            "priority", "recommended_action", "review_reason",
        ]
        available = [col for col in detail_columns if col in row.index]
        st.table(pd.DataFrame({"field": available, "value": [row[col] for col in available]}))

    # Screening history
    st.subheader("Screening History")
    history = load_raw()
    if history.empty:
        st.info("No historical raw records found.")
        return

    patient_history = history[history["person_id"].eq(row["person_id"])].copy()
    if patient_history.empty:
        st.info("No previous screening history found for this patient.")
        return

    patient_history["screening_date"] = patient_history["screening_date"].dt.date
    previous = patient_history[
        pd.to_datetime(patient_history["screening_date"]) < pd.to_datetime(row["screening_date"])
    ]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Previous Screens", f"{len(previous):,}")
    h2.metric("Prior hrHPV+", f"{int(previous['hrhpv_positive'].sum()):,}")
    h3.metric("Prior Abnormal Cytology", f"{int(previous['cytology_result'].ne('NILM').sum()):,}")
    h4.metric("Prior CIN2+", f"{int(previous['cin2plus_detected'].sum()):,}")

    # HPV genotype timeline
    tl = patient_history.sort_values("screening_date")
    fig = go.Figure(go.Scatter(
        x=pd.to_datetime(tl["screening_date"]),
        y=tl["hpv_genotype"],
        mode="markers+lines",
        marker=dict(
            color=[HPV_COLORS.get(g, "#ccc") for g in tl["hpv_genotype"]],
            size=14,
        ),
        line=dict(color="#ddd", width=1),
        text=tl["cytology_result"],
        hovertemplate="Date: %{x|%Y-%m-%d}<br>HPV: %{y}<br>Cytology: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="HPV Genotype Timeline",
        height=220, margin=dict(t=40, b=10),
        xaxis_title="Screening date",
        yaxis_title="HPV genotype",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    display_history = patient_history.sort_values("screening_date", ascending=False).head(8)
    st.dataframe(display_history[HISTORY_COLUMNS], use_container_width=True, hide_index=True)


# ══════════════════════════ TAB 3 — MONITORING ═══════════════════════════════

def show_drift_status(df: pd.DataFrame) -> None:
    report = load_drift_report()
    if not report:
        st.info("No drift report found yet. Run `python src/prediction_pipeline.py --batch-date YYYY-MM-DD`.")
        return

    status = report.get("overall_drift_detected", False)
    st.markdown(f"### Status: {'Drift Detected' if status else '✅ No Drift Detected'}")

    drift_rows = []
    for feature in ["age", "hpv_genotype", "cytology_result"]:
        if feature in report:
            item = report[feature]
            drift_rows.append({
                "feature": feature,
                "test": item.get("test"),
                "p_value": item.get("p_value"),
                "drift_detected": item.get("drift_detected"),
            })

    if not drift_rows:
        return

    drift_df = pd.DataFrame(drift_rows)
    c_bar, c_tbl = st.columns(2)

    with c_bar:
        fig = px.bar(
            drift_df, x="feature", y="p_value",
            color="drift_detected",
            color_discrete_map={True: "#d62728", False: "#2ca02c"},
            title="Feature Drift p-values (log scale)",
            text=drift_df["p_value"].map(lambda v: f"{v:.1e}"),
            labels={"feature": "Feature", "p_value": "p-value"},
        )
        fig.update_traces(textposition="outside")
        fig.add_hline(y=0.01, line_dash="dash", line_color="orange",
                      annotation_text="α = 0.01 threshold")
        fig.update_layout(
            height=300, margin=dict(t=40, b=20),
            yaxis_type="log", yaxis_title="p-value (log scale)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_tbl:
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
        for _, r in drift_df.iterrows():
            if r["drift_detected"]:
                st.error(f"**{r['feature']}** — p = {r['p_value']:.1e}: distribution shifted from training baseline.")
            else:
                st.success(f"**{r['feature']}** — p = {r['p_value']:.3f}: stable.")

    # Reference vs current distribution
    raw = load_raw()
    if raw.empty or df.empty:
        return

    st.markdown("---")
    st.subheader("Reference (Training) vs Current Batch — Distribution Comparison")
    ref = raw[pd.to_datetime(raw["screening_date"]).dt.year.between(2010, 2022)]

    c_age, c_geno = st.columns(2)
    with c_age:
        fig = go.Figure()
        for data, name, col in [
            (ref["age"], "Training reference", "#4e79a7"),
            (df["age"], "Current batch", "#d62728"),
        ]:
            fig.add_trace(go.Histogram(
                x=data, name=name, opacity=0.62, nbinsx=20,
                histnorm="probability", marker_color=col,
            ))
        fig.update_layout(
            barmode="overlay", height=270, margin=dict(t=30, b=20),
            xaxis_title="Age", yaxis_title="Proportion",
            title="Age Distribution",
            legend=dict(y=1.12, orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_geno:
        ref_geno = ref["hpv_genotype"].value_counts(normalize=True).rename("Reference")
        cur_geno = df["hpv_genotype"].value_counts(normalize=True).rename("Current")
        gdf = pd.concat([ref_geno, cur_geno], axis=1).fillna(0).reset_index()
        gdf.columns = ["hpv_genotype", "Reference", "Current"]
        fig = go.Figure()
        for col_name, color in [("Reference", "#4e79a7"), ("Current", "#d62728")]:
            fig.add_trace(go.Bar(
                name=col_name, x=gdf["hpv_genotype"],
                y=gdf[col_name], marker_color=color, opacity=0.82,
            ))
        fig.update_layout(
            barmode="group", height=270, margin=dict(t=30, b=20),
            xaxis_title="HPV Genotype", yaxis_title="Proportion",
            yaxis_tickformat=".0%", title="HPV Genotype Distribution",
            legend=dict(y=1.12, orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    cyto_order = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]
    ref_cyto = (ref["cytology_result"].value_counts(normalize=True)
                .reindex(cyto_order).fillna(0).rename("Reference"))
    cur_cyto = (df["cytology_result"].value_counts(normalize=True)
                .reindex(cyto_order).fillna(0).rename("Current"))
    cdf = pd.concat([ref_cyto, cur_cyto], axis=1).reset_index()
    cdf.columns = ["cytology", "Reference", "Current"]
    fig = go.Figure()
    for col_name, color in [("Reference", "#4e79a7"), ("Current", "#d62728")]:
        fig.add_trace(go.Bar(
            name=col_name, x=cdf["cytology"],
            y=cdf[col_name], marker_color=color, opacity=0.82,
        ))
    fig.update_layout(
        barmode="group", height=270, margin=dict(t=30, b=20),
        xaxis_title="Cytology Result", yaxis_title="Proportion",
        yaxis_tickformat=".1%", title="Cytology Distribution",
        legend=dict(y=1.12, orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════ MAIN ═════════════════════════════════════

def main() -> None:
    st.set_page_config(page_title="CerviRisk Monthly Triage", layout="wide")
    st.title("CerviRisk Monthly Triage")

    default_path = latest_prediction_file()
    if default_path is None:
        st.error("No prediction file found. Run `python src/prediction_pipeline.py --batch-date YYYY-MM-DD` first.")
        return

    with st.sidebar:
        prediction_files = sorted(
            PREDICTION_ROOT.glob("batch_date=*/predictions.parquet"), reverse=True
        )
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
        show_review_queue(df, filtered)

    with tab_patient:
        show_patient_detail(filtered)

    with tab_monitor:
        show_drift_status(df)


if __name__ == "__main__":
    main()
