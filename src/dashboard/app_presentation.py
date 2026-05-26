from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.db import DB_PATH, query_patient_history

PREDICTION_ROOT = Path("data/predictions")
INCOMING_DIR = Path("data/incoming")
DRIFT_REPORT = Path("reports/drift_report.json")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.csv")
MODEL_METRICS_PATH = Path("reports/model_metrics.csv")
TUNING_RESULTS_PATH = Path("reports/tuning_results.csv")
PROCESSED_DIR = Path("data/processed")

PRIORITY_ORDER = ["urgent", "high", "medium", "routine"]
PRIORITY_COLORS = {"urgent": "#d62728", "high": "#ff7f0e", "medium": "#1f77b4", "routine": "#2ca02c"}
HISTORY_COLUMNS = [
    "screening_date", "age", "hpv_test_result", "hpv_genotype",
    "hrhpv_positive", "cytology_result", "histology_result",
    "persistent_hrhpv", "cin2plus_detected", "treatment_performed",
]
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLOR = {
    "Logistic Regression": "#4e79a7",
    "Random Forest": "#d62728",
    "XGBoost": "#59a14f",
}


# ═══════════════════════════════ DATA LOADERS ════════════════════════════════

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
def load_raw() -> pd.DataFrame:
    files = sorted(INCOMING_DIR.glob("batch_date=*/screening_records.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH) if FEATURE_IMPORTANCE_PATH.exists() else None


@st.cache_data
def load_model_metrics() -> pd.DataFrame | None:
    return pd.read_csv(MODEL_METRICS_PATH) if MODEL_METRICS_PATH.exists() else None


@st.cache_data
def load_tuning() -> pd.DataFrame | None:
    return pd.read_csv(TUNING_RESULTS_PATH) if TUNING_RESULTS_PATH.exists() else None


@st.cache_data
def load_metadata() -> dict:
    p = PROCESSED_DIR / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_drift_report() -> dict:
    return json.loads(DRIFT_REPORT.read_text()) if DRIFT_REPORT.exists() else {}


@st.cache_data
def load_patient_history_from_db(person_id: str) -> pd.DataFrame:
    return query_patient_history(person_id)


# ═══════════════════════════════ UTILITIES ═══════════════════════════════════

def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def priority_rank(series: pd.Series) -> pd.Series:
    ranks = {n: i for i, n in enumerate(PRIORITY_ORDER)}
    return series.map(ranks).fillna(len(PRIORITY_ORDER))


def risk_column_config() -> dict:
    return {
        "risk_cin2_1yr": st.column_config.ProgressColumn("1-yr CIN2+ risk", format="%.1f%%", min_value=0, max_value=100),
        "risk_cin2_3yr": st.column_config.ProgressColumn("3-yr CIN2+ risk", format="%.1f%%", min_value=0, max_value=100),
        "risk_cin2_5yr": st.column_config.ProgressColumn("5-yr CIN2+ risk", format="%.1f%%", min_value=0, max_value=100),
    }


def pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["risk_cin2_1yr", "risk_cin2_3yr", "risk_cin2_5yr"]:
        if c in df:
            df[c] = df[c] * 100
    return df


# ═══════════════════════ TAB 1 — PIPELINE & CODE ════════════════════════════

def _pipeline_diagram() -> go.Figure:
    """
    Three-zone layout (left → center → right):

      Zone A  │  Zone B           │  Zone C
    ──────────┼───────────────────┼──────────────────────────────────────
    Simulator │                   │  ① preprocess → split → train → models
    Ingest    │  data/incoming/   │
              │  (unified store)  │  ② inference → predict → output → dash
              │                   │
              │                   │  ③ drift monitor
    """
    PROC = "#4e79a7"
    STOR = "#59a14f"
    SERV = "#e15759"
    MON  = "#f28e2b"
    W, H = 1.28, 0.64   # standard node size

    # ── Zone A: single data source (left, centered) ──────────────────────
    sim  = (1.10, 2.50)   # Simulator — generates all months at once

    # ── Zone B: screening data store (center, tall & prominent) ──────────
    inc  = (3.50, 2.50)   # data/incoming/ — one Parquet file per month
    IW, IH = 1.50, 2.20  # taller to show it holds many months

    # ── Zone C: ML processing (right) ────────────────────────────────────
    # Training branch  y=3.55
    prep = (5.60, 3.55)
    spl  = (7.00, 3.55)
    trn  = (8.30, 3.55)
    mdl  = (9.55, 3.55)
    # Inference branch  y=1.45  (ingest.py leads, fetches latest batch)
    ing  = (5.60, 1.45)
    inf  = (6.80, 1.45)
    prd  = (7.95, 1.45)
    out  = (9.10, 1.45)
    srv  = (10.25, 1.45)
    # Monitoring  y=0.35
    drft = (7.95, 0.35)

    fig = go.Figure()
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(visible=False, range=[-0.3, 11.10]),
        yaxis=dict(visible=False, range=[-0.15, 4.50]),
        plot_bgcolor="white",
        title=dict(
            text="CerviRisk — End-to-End System Architecture",
            font=dict(size=16, color="#222"),
            x=0.5, xanchor="center",
        ),
    )

    # ── Background zone boxes ─────────────────────────────────────────────
    # Zone A
    fig.add_shape(type="rect", x0=-0.20, x1=2.30, y0=1.55, y1=3.45,
                  fillcolor="rgba(78,121,167,0.05)",
                  line=dict(color="rgba(78,121,167,0.30)", width=1.2))
    fig.add_annotation(x=1.05, y=3.45, text="<b>Data Simulation</b>",
                       font=dict(size=12, color="#4e79a7"),
                       showarrow=False, yanchor="bottom")

    # Zone B
    fig.add_shape(type="rect", x0=2.60, x1=4.40, y0=1.20, y1=3.80,
                  fillcolor="rgba(89,161,79,0.09)",
                  line=dict(color="rgba(89,161,79,0.55)", width=1.8, dash="dot"))
    fig.add_annotation(x=3.50, y=3.80, text="<b>Screening Data Store</b>",
                       font=dict(size=12, color="#388e3c"),
                       showarrow=False, yanchor="bottom")

    # Zone C — Training
    fig.add_shape(type="rect", x0=4.80, x1=10.30, y0=2.85, y1=4.30,
                  fillcolor="rgba(78,121,167,0.05)",
                  line=dict(color="rgba(78,121,167,0.22)", width=1.2))
    fig.add_annotation(x=4.88, y=4.30, text="<b>① Training Pipeline</b>",
                       font=dict(size=11, color="#4e79a7"),
                       showarrow=False, yanchor="bottom", xanchor="left")

    # Zone C — Inference
    fig.add_shape(type="rect", x0=4.80, x1=10.90, y0=0.78, y1=2.20,
                  fillcolor="rgba(89,161,79,0.05)",
                  line=dict(color="rgba(89,161,79,0.22)", width=1.2))
    fig.add_annotation(x=4.88, y=2.20, text="<b>② Inference Pipeline</b>",
                       font=dict(size=11, color="#388e3c"),
                       showarrow=False, yanchor="bottom", xanchor="left")

    # Zone C — Monitoring
    fig.add_shape(type="rect", x0=5.85, x1=9.20, y0=-0.05, y1=0.76,
                  fillcolor="rgba(242,142,43,0.08)",
                  line=dict(color="rgba(242,142,43,0.40)", width=1.2))
    fig.add_annotation(x=5.92, y=0.76, text="<b>③ Drift Monitoring</b>",
                       font=dict(size=11, color="#e65100"),
                       showarrow=False, yanchor="bottom", xanchor="left")

    # ── Helper: draw a labelled node ─────────────────────────────────────
    def node(x, y, label, sub, color, nw=W, nh=H):
        fig.add_shape(type="rect",
                      x0=x-nw/2, x1=x+nw/2, y0=y-nh/2, y1=y+nh/2,
                      fillcolor=color, line=dict(color="white", width=2.2), opacity=0.93)
        fig.add_annotation(x=x, y=y+0.13, text=f"<b>{label}</b>",
                           font=dict(size=10, color="white"), showarrow=False)
        fig.add_annotation(x=x, y=y-0.16, text=sub,
                           font=dict(size=8, color="rgba(255,255,255,0.90)"), showarrow=False)

    # ── Helper: draw an arrow ─────────────────────────────────────────────
    def arr(x0, y0, x1, y1, color="#888", width=1.7):
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2,
            arrowsize=1.0, arrowwidth=width, arrowcolor=color,
        )

    # ── Helper: small inline label on an arrow ───────────────────────────
    def arr_label(x, y, text, color="#555"):
        fig.add_annotation(x=x, y=y, text=text,
                           font=dict(size=8.5, color=color),
                           showarrow=False, bgcolor="white",
                           borderpad=2)

    # ── Draw Zone A node ──────────────────────────────────────────────────
    node(*sim, "Simulator", "simulator.py --monthly", PROC)

    # ── Draw Zone B node (tall to show it holds many months) ──────────────
    node(*inc, "data/incoming/", "one Parquet file per month", STOR, nw=IW, nh=IH)

    # ── Draw Zone C nodes ─────────────────────────────────────────────────
    node(*prep, "Feature Eng.",    "preprocess.py",      PROC)
    node(*spl,  "Time Split",      "train / val / test",  PROC)
    node(*trn,  "Train Models",    "train.py",            PROC)
    node(*mdl,  "XGBoost .pkl",    "models/",             STOR)
    node(*ing,  "Load Batch",      "ingest.py",           PROC)
    node(*inf,  "Feature Infer.", "inference.py",        PROC)
    node(*prd,  "Batch Predict",  "predict_batch.py",    PROC)
    node(*out,  "Predictions",    "data/predictions/",   STOR)
    node(*srv,  "Dashboard + API","Streamlit + FastAPI", SERV)
    node(*drft, "Drift Monitor",  "drift.py",            MON)

    # ── Arrow: Simulator → Screening Data Store (horizontal) ─────────────
    arr(sim[0]+W/2, sim[1], inc[0]-IW/2, inc[1])
    arr_label(2.45, 2.78, "all months\n2010–2026", "#388e3c")

    # ── Arrows: Screening Data Store → Zone C ────────────────────────────
    # → Training: reads 2010–2022 range
    arr(inc[0]+IW/2, inc[1]+0.55, prep[0]-W/2, prep[1])
    arr_label(4.72, 3.30, "2010–2022", "#4e79a7")
    # → Inference: ingest.py fetches the latest month
    arr(inc[0]+IW/2, inc[1]-0.55, ing[0]-W/2, ing[1])
    arr_label(4.72, 1.72, "2026-01", "#388e3c")

    # ── Arrows: Training path ─────────────────────────────────────────────
    arr(prep[0]+W/2, prep[1], spl[0]-W/2, spl[1])
    arr(spl[0]+W/2,  spl[1],  trn[0]-W/2, trn[1])
    arr(trn[0]+W/2,  trn[1],  mdl[0]-W/2, mdl[1])

    # ── Arrows: Inference path ────────────────────────────────────────────
    arr(ing[0]+W/2, ing[1], inf[0]-W/2, inf[1])
    arr(inf[0]+W/2, inf[1], prd[0]-W/2, prd[1])
    arr(prd[0]+W/2, prd[1], out[0]-W/2, out[1])
    arr(out[0]+W/2, out[1], srv[0]-W/2, srv[1])

    # ── Arrow: models → predict_batch (cross-lane, models are reused) ─────
    arr(mdl[0], mdl[1]-H/2, prd[0], prd[1]+H/2, color="#9467bd", width=1.3)
    arr_label(9.55, 2.55, "scores\nbatch", "#9467bd")

    # ── Arrows: Drift monitoring ──────────────────────────────────────────
    arr(prd[0], prd[1]-H/2, drft[0], drft[1]+H/2, color=MON)
    arr(drft[0]+W/2, drft[1], srv[0], srv[1]-H/2, color=MON)
    arr_label(9.05, 0.88, "drift\nreport", "#e65100")

    return fig


def tab_pipeline() -> None:
    st.plotly_chart(_pipeline_diagram(), use_container_width=True)
    st.info(
        "**Key design principle:** `simulator.py --monthly` generates all monthly files at once into `data/incoming/`. "
        "Training, validation, test, and prediction all read from this same folder — the only difference is which "
        "date range you select.  Changing the split boundary requires no data regeneration."
    )
    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Code Structure & Layering")
        st.code(
            "src/\n"
            "├── config.py          # Central config — one place to change params\n"
            "├── db.py              # SQLite layer — patient + visit relational store\n"
            "├── ingestion/\n"
            "│   ├── simulator.py   # Generates ALL months → data/incoming/\n"
            "│   │                  #   --monthly  one Parquet file per calendar month\n"
            "│   └── ingest.py      # (optional) append real hospital data as new months\n"
            "├── features/\n"
            "│   ├── preprocess.py  # Training features + outcome labels\n"
            "│   │                  #   --from-monthly  reads date range from data/incoming/\n"
            "│   └── inference.py   # Inference feature builder (no label needed)\n"
            "├── modeling/\n"
            "│   ├── train.py       # Trains 3 models, selects best\n"
            "│   ├── tune.py        # Random-search hyperparameter tuning\n"
            "│   ├── predict_batch.py  # Scores a monthly parquet batch\n"
            "│   └── explain.py     # SHAP explanations\n"
            "├── serving/\n"
            "│   └── app.py         # FastAPI REST endpoint\n"
            "├── dashboard/\n"
            "│   ├── app.py                # Clinical triage dashboard\n"
            "│   └── app_presentation.py   # ML pipeline story (this file)\n"
            "└── monitoring/\n"
            "    └── drift.py       # Statistical drift detection",
            language="",
        )
        st.info("Each sub-package is independently runnable. `config.py` is the single source of truth for all parameters.")

    with c2:
        st.subheader("Storage Decisions — Why Each Format")
        storage = pd.DataFrame({
            "Stage": [
                "Patient + visit records",
                "All screening data (training & inference)",
                "Processed feature matrix",
                "Batch predictions",
                "Trained model files",
                "Drift reports / metadata",
            ],
            "Location": [
                "data/cervirisk.db  (SQLite)",
                "data/incoming/batch_date=YYYY-MM-DD/",
                "data/processed/  (train / val / test splits)",
                "data/predictions/batch_date=YYYY-MM-DD/",
                "models/  (.pkl via joblib)",
                "reports/  (JSON / CSV)",
            ],
            "Why this format": [
                "Relational: JOIN patients ↔ visits, filter by person_id",
                "One unified store — simulator writes history, ingest appends new months; training and inference both read here",
                "Columnar Parquet: directly consumed by sklearn/pandas",
                "Append-only partitions = built-in audit trail, re-load any past batch",
                "Compact sklearn object serialisation",
                "Human-readable, git-diffable, version-control friendly",
            ],
        })
        st.dataframe(storage, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Data Cadence — Monthly Batches")
        st.markdown("""
**Why monthly?**
- Lab results arrive in monthly reporting cycles from colposcopy units
- Clinical MDT triage decisions are made at monthly meetings
- Enough new records per month (~750) for statistically meaningful drift tests
- Model re-training (when triggered) runs quarterly on accumulated new data
        """)


# ════════════════════════ TAB 2 — DATA GENERATION ═══════════════════════════

def tab_data() -> None:
    st.subheader("Synthetic Data Simulator")

    params = pd.DataFrame({
        "Parameter": [
            "Patients simulated", "Time span", "Visits per patient (median)",
            "Random seed", "CIN2+ prevalence", "HPV vaccination rate",
            "Immunosuppressed fraction", "Smoking (current)",
        ],
        "Value": ["22,000", "2010 – 2026", "5", "42 (fully reproducible)",
                  "~1.2 %", "42 % (born ≥1988) / 12 % (older)",
                  "4.5 %", "17 %"],
        "Clinical basis": [
            "Representative regional screening cohort",
            "17-year longitudinal follow-up",
            "Annual / biennial screening interval",
            "Reproducibility for peer review",
            "Matches reported CIN2+ incidence in screened populations",
            "Birth-cohort HPV vaccine uptake gradient",
            "Known immunosuppression risk factor",
            "Tobacco as independent risk factor",
        ],
    })
    st.dataframe(params, use_container_width=True, hide_index=True)

    raw = load_raw()
    if raw.empty:
        st.warning("No monthly batches found — run `python src/ingestion/simulator.py --monthly` first.")
        return

    raw["year"] = pd.to_datetime(raw["screening_date"]).dt.year

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Records per Year — and Split Boundaries")
        yc = raw.groupby("year").size().reset_index(name="records")
        fig = px.bar(yc, x="year", y="records", color="records",
                     color_continuous_scale="Blues",
                     labels={"year": "Year", "records": "Screening Records"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=300, margin=dict(t=20, b=20))
        for x0, x1, label, col in [
            (2009.5, 2018.5, "Train", "rgba(78,121,167,0.18)"),
            (2018.5, 2020.5, "Val",   "rgba(89,161,79,0.18)"),
            (2020.5, 2022.5, "Test",  "rgba(242,142,43,0.18)"),
            (2022.5, 2026.5, "Holdout","rgba(214,39,40,0.12)"),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0,
                          annotation_text=f"<b>{label}</b>", annotation_position="top left",
                          annotation_font_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Visits per Patient")
        vp = raw.groupby("person_id").size().reset_index(name="visits")
        fig = px.histogram(vp, x="visits", nbins=8, color_discrete_sequence=["#4e79a7"],
                           labels={"visits": "Number of visits", "count": "Patients"})
        fig.update_layout(height=300, margin=dict(t=20, b=20), bargap=0.06)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    GENO_LABELS = {
        "negative": "HPV−", "low_risk": "Low-risk HPV", "other_hr": "Other hrHPV",
        "hpv18": "HPV18", "hpv16": "HPV16", "multiple_hr": "Multiple hrHPV",
    }

    with c3:
        st.subheader("HPV Genotype Distribution")
        geno = raw["hpv_genotype"].map(GENO_LABELS).value_counts()
        fig = go.Figure(go.Pie(
            labels=geno.index.tolist(), values=geno.values.tolist(),
            hole=0.48, textinfo="label+percent",
            marker_colors=px.colors.qualitative.Safe,
        ))
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("CIN2+ Detection Rate by HPV Type")
        cr = (raw.groupby("hpv_genotype")["cin2plus_detected"].mean()
              .rename("cin2plus_rate").reset_index())
        cr["label"] = cr["hpv_genotype"].map(GENO_LABELS)
        cr = cr.sort_values("cin2plus_rate")
        cr["pct"] = cr["cin2plus_rate"] * 100
        fig = px.bar(cr, x="pct", y="label", orientation="h",
                     color="pct", color_continuous_scale="Reds",
                     labels={"pct": "CIN2+ Detection Rate (%)", "label": ""})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("HPV16 and Multiple hrHPV carry the highest risk — a key signal the model captures.")

    st.markdown("---")
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Cytology Result Distribution")
        cyto_order = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]
        cy = raw["cytology_result"].value_counts().reindex(cyto_order).fillna(0).reset_index()
        cy.columns = ["cytology", "count"]
        fig = px.bar(cy, x="cytology", y="count", color="cytology",
                     color_discrete_sequence=["#4393c3","#74c476","#fd8d3c","#ff7f0e","#d62728","#9467bd"],
                     labels={"cytology": "Cytology Result", "count": "Records"})
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Age Distribution at First Visit")
        first = raw.sort_values("screening_date").groupby("person_id").first().reset_index()
        fig = px.histogram(first, x="age", nbins=22, color_discrete_sequence=["#59a14f"],
                           labels={"age": "Age at first visit", "count": "Patients"})
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total records",   f"{len(raw):,}")
    k2.metric("Unique patients", f"{raw['person_id'].nunique():,}")
    k3.metric("CIN2+ rate",      f"{raw['cin2plus_detected'].mean():.2%}")
    k4.metric("hrHPV+ rate",     f"{raw['hrhpv_positive'].mean():.2%}")


# ══════════════════════ TAB 3 — FEATURE ENGINEERING ═════════════════════════

def tab_features() -> None:
    # Time split Gantt
    st.subheader("Time-Based Train / Validation / Test Split")
    splits = [
        ("Training",      2010, 2018, 63042,  "#4e79a7"),
        ("Validation",    2019, 2020, 13990,  "#59a14f"),
        ("Test",          2021, 2022, 14756,  "#f28e2b"),
        ("Future Holdout",2023, 2026, 14769,  "#e15759"),
    ]
    fig = go.Figure()
    for name, start, end, rows, col in splits:
        fig.add_trace(go.Bar(
            name=name, x=[end - start], y=[name],
            base=start, orientation="h",
            marker_color=col, opacity=0.88,
            text=f"  {rows:,} rows",
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=f"{name}: {start}–{end}, {rows:,} rows<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay", height=200,
        xaxis=dict(title="Year", range=[2009, 2025.5], dtick=1),
        yaxis=dict(title=""),
        showlegend=True,
        legend=dict(orientation="h", y=1.2),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.warning("**Why time-based, not random?** Random CV would let the model train on 2022 records and predict 2015 outcomes — data leakage. Time-split enforces temporal order, matching real deployment where the model only ever sees past data.")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Feature Categories  (45 total)")
        feat_cats = pd.DataFrame({
            "Category": [
                "Current visit — numeric (10)",
                "Current visit — categorical, one-hot (29)",
                "Longitudinal history (6)",
                "Demographic (4)",
                "Three outcome targets (labels, not features)",
            ],
            "Key examples": [
                "age, visit_index, parity, colposcopy_performed",
                "hpv_genotype ×6, cytology ×6, histology ×6, region ×5, smoking ×3, hpv_test ×2",
                "n_previous_screens, time_since_last_screen, ever_had_hrHPV, ever_had_cin1plus",
                "hpv_vaccinated, immunosuppressed, birth_year",
                "outcome_cin2_1yr / 3yr / 5yr — forward-looking binary labels",
            ],
        })
        st.dataframe(feat_cats, use_container_width=True, hide_index=True)
        st.info("Longitudinal features capture patient history that a single-visit snapshot misses — this is the key engineering insight of the feature layer.")

    with c2:
        st.subheader("Class Imbalance — CIN2+ is Rare")
        raw = load_raw()
        if not raw.empty:
            pos = int(raw["cin2plus_detected"].sum())
            neg = len(raw) - pos
            fig = go.Figure(go.Pie(
                labels=["CIN2+ (positive)", "No CIN2+ (negative)"],
                values=[pos, neg], hole=0.55,
                marker_colors=["#d62728", "#4e79a7"],
                textinfo="label+percent",
            ))
            fig.update_layout(showlegend=False, height=260,
                              margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            ratio = neg / pos
            st.error(f"Only {pos/(pos+neg):.1%} of records are CIN2+ positive.  \n"
                     f"→ XGBoost `scale_pos_weight = {ratio:.0f}` compensates, preventing the model from simply predicting 'all negative' and claiming high accuracy.")

    # Outcome label construction diagram
    st.markdown("---")
    st.subheader("How Outcome Labels Are Constructed")

    fig = go.Figure()
    visits_x = [2012, 2015, 2018, 2021]
    cin2_x   = 2019.3

    for yrs, col, lbl in [
        (5, "rgba(148,103,189,0.12)", "5-yr window → label = 1"),
        (3, "rgba(214,39,40,0.14)",   "3-yr window → label = 1"),
        (1, "rgba(255,127,14,0.20)",  "1-yr window → label = 1"),
    ]:
        fig.add_vrect(x0=2018, x1=2018+yrs, fillcolor=col, line_width=0,
                      annotation_text=f"<b>+{yrs}yr</b>",
                      annotation_position="top right", annotation_font_size=10)

    fig.add_trace(go.Scatter(
        x=visits_x, y=[1]*4, mode="markers+text",
        marker=dict(size=18, color="#4e79a7", symbol="circle"),
        text=["Visit 1", "Visit 2", "<b>Target visit</b>", "Visit 4"],
        textposition="top center", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[cin2_x], y=[1], mode="markers+text",
        marker=dict(size=22, color="#d62728", symbol="x-thin", line=dict(width=3, color="#d62728")),
        text=["CIN2+ detected"], textposition="bottom center", showlegend=False,
    ))
    fig.add_annotation(
        x=2018, y=0.70,
        text="<b>At this visit:</b><br>1-yr label = <b>1</b> ✓ (CIN2+ within 2019)<br>"
             "3-yr label = <b>1</b> ✓ (CIN2+ within 2021)<br>"
             "5-yr label = <b>1</b> ✓ (CIN2+ within 2023)",
        font=dict(size=11), align="left",
        showarrow=True, arrowhead=2, ax=60, ay=-60,
        bordercolor="#ccc", borderwidth=1, bgcolor="white",
    )

    fig.update_layout(
        height=280,
        xaxis=dict(title="Year", range=[2009, 2025], dtick=1),
        yaxis=dict(visible=False, range=[0.3, 1.8]),
        plot_bgcolor="white", margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "For each visit we look forward 1 / 3 / 5 years within the same patient's record. "
        "Only information available *at* the visit enters the feature vector — no data leakage."
    )


# ═════════════════════════ TAB 4 — MODEL TRAINING ═══════════════════════════

def tab_training() -> None:
    metrics = load_model_metrics()

    # Cross-validation folds diagram
    st.subheader("Walk-Forward Cross-Validation  (TimeSeriesSplit, 4 folds)")
    fold_data = [
        (1, 2010, 2013, 2014, 2015),
        (2, 2010, 2015, 2016, 2017),
        (3, 2010, 2017, 2018, 2019),
        (4, 2010, 2019, 2020, 2021),
    ]
    fig = go.Figure()
    shown = {"Train": False, "Validate": False}
    for fold, ts, te, vs, ve in fold_data:
        for label, x0, x1, col in [("Train", ts, te, "#4e79a7"), ("Validate", vs, ve, "#f28e2b")]:
            fig.add_trace(go.Bar(
                name=label, x=[x1 - x0], y=[f"Fold {fold}"],
                base=x0, orientation="h",
                marker_color=col, opacity=0.85,
                showlegend=not shown[label],
                text=f" {x0}–{x1}",
                textposition="inside", insidetextanchor="start",
            ))
            shown[label] = True
    fig.update_layout(
        barmode="overlay", height=230,
        xaxis=dict(title="Year", range=[2009, 2022], dtick=1),
        legend=dict(orientation="h", y=1.15),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each fold's training window grows. The model never sees validation data — mimicking real deployment where only past records are available.")

    if metrics is None:
        st.warning("Model metrics not found.")
        return

    metrics["model_label"] = metrics["model"].map(MODEL_LABEL).fillna(metrics["model"])
    val = metrics[metrics["fold"] == "validation"].copy()

    st.markdown("---")
    st.subheader("Why Not Random Forest?  The Sensitivity Problem")

    c_sens, c_scatter = st.columns(2)

    with c_sens:
        fig = px.bar(
            val, x="model_label", y="sensitivity",
            color="model_label", color_discrete_map=MODEL_COLOR,
            labels={"model_label": "Model", "sensitivity": "Sensitivity (recall for CIN2+)"},
            text=val["sensitivity"].map(lambda v: f"{v:.3f}"),
        )
        fig.update_traces(textposition="outside")
        fig.add_hline(y=0.40, line_dash="dash", line_color="orange",
                      annotation_text="Minimum acceptable clinical threshold")
        fig.update_layout(
            showlegend=False, height=300, margin=dict(t=30, b=20),
            yaxis=dict(range=[0, 0.85], title="Sensitivity"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.error("**Random Forest: sensitivity ≈ 0.008** — predicts almost no positives.  \n"
                 "In cancer screening this means missing 99% of at-risk patients. Clinically useless, regardless of AUC.")

    with c_scatter:
        fig = go.Figure()
        for _, row in val.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["sensitivity"]], y=[row["specificity"]],
                mode="markers+text",
                marker=dict(size=20, color=MODEL_COLOR.get(row["model_label"], "#aaa")),
                text=[row["model_label"].replace("Logistic Regression","LR")
                      .replace("Random Forest","RF").replace("XGBoost","XGB")],
                textposition="top right", showlegend=False,
            ))
        fig.add_scatter(x=[1], y=[1], mode="markers",
                        marker=dict(size=14, color="#aaa", symbol="star"),
                        name="Ideal", showlegend=True)
        fig.add_shape(type="line", x0=0, y0=1, x1=1, y1=0,
                      line=dict(dash="dash", color="#ccc"))
        fig.update_layout(
            height=300, margin=dict(t=30, b=20),
            xaxis=dict(title="Sensitivity — catches CIN2+", range=[0, 1.05]),
            yaxis=dict(title="Specificity — avoids false alarms", range=[0.5, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("**XGBoost** occupies the best clinical tradeoff — high specificity (0.87) while retaining useful sensitivity (0.49).  \n"
                   "Logistic Regression is more sensitive but flags too many false alarms.")

    # AUC across folds
    st.markdown("---")
    st.subheader("Model Stability — AUC Across All CV Folds")
    cv = metrics[metrics["fold"].astype(str).str.match(r"^\d+$")].copy()
    cv["fold_num"] = cv["fold"].astype(int)
    fig = px.line(
        cv, x="fold_num", y="auc", color="model_label",
        color_discrete_map=MODEL_COLOR, markers=True,
        labels={"fold_num": "CV Fold", "auc": "AUC", "model_label": "Model"},
    )
    fig.update_layout(height=270, margin=dict(t=20, b=20), yaxis=dict(range=[0.5, 1.0]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Stable AUC across folds confirms the model generalises to unseen time periods, not just the training window.")

    # Hyperparameter tuning
    st.markdown("---")
    st.subheader("Hyperparameter Tuning — Random Search over Parameter Grid")
    c_grid, c_fi = st.columns(2)

    with c_grid:
        param_space = pd.DataFrame({
            "Parameter": ["n_estimators", "max_depth", "learning_rate",
                          "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"],
            "Search range": ["[160, 220, 260, 340, 420]", "[2, 3, 4]",
                             "[0.025, 0.04, 0.06, 0.08]", "[0.75, 0.86, 0.95]",
                             "[0.75, 0.86, 0.95]", "[1, 3, 6]", "[1.0, 2.0, 5.0]"],
            "Controls": ["Model size vs training time", "Tree depth — over/underfit",
                         "Step size — speed vs precision", "Row subsampling — overfit",
                         "Feature subsampling — overfit", "Min leaf size — regularisation", "L2 penalty"],
        })
        st.dataframe(param_space, use_container_width=True, hide_index=True)

        tuning = load_tuning()
        if tuning is not None and not tuning.empty:
            st.markdown("**Random search results (mean AUC across 4 folds):**")
            show_cols = [c for c in ["n_estimators","max_depth","learning_rate",
                                     "subsample","mean_auc","std_auc"] if c in tuning.columns]
            tshow = tuning[show_cols].copy().round(4)
            st.dataframe(tshow.sort_values("mean_auc", ascending=False),
                         use_container_width=True, hide_index=True)

    with c_fi:
        fi = load_feature_importance()
        if fi is not None:
            def clean(n: str) -> str:
                return (n.replace("hpv_genotype_", "HPV: ")
                         .replace("cytology_result_", "Cytology: ")
                         .replace("smoking_status_", "Smoking: ")
                         .replace("histology_result_", "Histology: ")
                         .replace("hpv_test_result_", "HPV test: ")
                         .replace("region_", "Region: ")
                         .replace("_", " "))
            top = fi.head(14).copy()
            top["label"] = top["feature"].apply(clean)
            top = top.sort_values("importance")
            fig = px.bar(top, x="importance", y="label", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         labels={"importance": "Gain (feature importance)", "label": ""})
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              height=380, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("hrHPV positivity and HPV test result dominate. History features (ever_had_hrHPV, ever_had_cin1plus) appear in the top 10 — validating the longitudinal feature engineering.")


# ═══════════════════════ TAB 5 — INFERENCE & SERVING ════════════════════════

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        priorities = st.multiselect("Priority", PRIORITY_ORDER, default=["urgent", "high"])
        min_risk = st.slider("Minimum 3-year risk", 0.0, 1.0, 0.20, 0.05)
        geno_opts = sorted(df["hpv_genotype"].dropna().astype(str).unique()) if "hpv_genotype" in df else []
        sel_geno = st.multiselect("HPV genotype", geno_opts)
        cyto_opts = sorted(df["cytology_result"].dropna().astype(str).unique()) if "cytology_result" in df else []
        sel_cyto = st.multiselect("Cytology", cyto_opts)

    out = df[df["priority"].isin(priorities)] if "priority" in df else df
    out = out[out["risk_cin2_3yr"] >= min_risk]
    if sel_geno and "hpv_genotype" in out:
        out = out[out["hpv_genotype"].astype(str).isin(sel_geno)]
    if sel_cyto and "cytology_result" in out:
        out = out[out["cytology_result"].astype(str).isin(sel_cyto)]
    return out


def tab_serving(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("Two Serving Modes")
    c_api, c_batch = st.columns(2)

    with c_api:
        st.markdown("#### Real-Time — FastAPI REST Endpoint")
        st.markdown("**`POST /predict`** — single record, returns risk in <50 ms:")
        st.code('{\n  "age": 45,\n  "hpv_genotype": "hpv16",\n  "hrhpv_positive": true,\n'
                '  "cytology_result": "HSIL",\n  "n_previous_screens": 3,\n'
                '  "time_since_last_screen": 2.1,\n  "ever_had_hrHPV": true\n}', language="json")
        st.markdown("**Response:**")
        st.code('{\n  "risk_probabilities": {\n    "1yr": 0.412,\n    "3yr": 0.781,\n'
                '    "5yr": 0.834\n  },\n  "model": "xgboost",\n  "target": "CIN2+"\n}', language="json")
        st.caption("`uvicorn src.serving.app:app --reload`  then `GET /health` to verify")

    with c_batch:
        st.markdown("#### Batch — Monthly Prediction Pipeline")
        st.code(
            "# Ingest this month's records from the database\n"
            "python src/ingestion/ingest.py \\\n"
            "    --from-db --batch-date 2026-01-01\n\n"
            "# Score the batch + run drift monitor\n"
            "python src/prediction_pipeline.py \\\n"
            "    --batch-date 2026-01-01",
            language="bash",
        )
        steps = ["DB Query", "Feature\nInference", "Score\n(XGBoost)", "Triage\nRules", "Parquet\nOutput", "Dashboard"]
        colors = ["#59a14f","#4e79a7","#4e79a7","#4e79a7","#59a14f","#e15759"]
        fig = go.Figure(go.Scatter(
            x=list(range(len(steps))), y=[1]*len(steps),
            mode="markers+text+lines",
            marker=dict(size=22, color=colors),
            text=steps, textposition="bottom center",
            line=dict(color="#ccc", width=2), showlegend=False,
        ))
        fig.update_layout(height=160, margin=dict(t=10, b=55, l=10, r=10),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Automated Triage Rules — How Predictions Become Actions")
    rules = pd.DataFrame({
        "Priority": ["🔴 Urgent", "🟠 High", "🟡 Medium", "🟢 Routine"],
        "Trigger condition": [
            "3yr risk ≥ 60%  OR  5yr risk ≥ 75%  OR  (HSIL/AGC + HPV16/18)",
            "3yr risk ≥ 40%  OR  (HPV16/18 + persistent hrHPV)  OR  (high-grade cytology + hrHPV)",
            "3yr risk ≥ 20%  OR  any hrHPV positive",
            "All other cases",
        ],
        "Recommended action": [
            "Colposcopy or immediate specialist review",
            "Clinician review within monthly triage meeting",
            "Repeat HPV / cytology at shortened interval",
            "Continue routine screening schedule",
        ],
    })
    st.dataframe(rules, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("This Month's Batch")
    total = len(df)
    urgent = int(df["priority"].eq("urgent").sum())
    hou = int(df["priority"].isin(["urgent","high"]).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records processed", f"{total:,}")
    k2.metric("Urgent", f"{urgent:,}", f"{urgent/total:.0%} of batch", delta_color="inverse")
    k3.metric("High or Urgent", f"{hou:,}", f"{hou/total:.0%} of batch", delta_color="inverse")
    k4.metric("Mean 3-yr risk", fmt_pct(float(df["risk_cin2_3yr"].mean())))

    c_pie, c_hist, c_scatter = st.columns(3)
    with c_pie:
        counts = df["priority"].value_counts().reindex(PRIORITY_ORDER).fillna(0)
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.52,
            marker_colors=[PRIORITY_COLORS[p] for p in counts.index],
            textinfo="label+value",
        ))
        fig.update_layout(showlegend=False, height=220, margin=dict(t=10,b=10,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)
    with c_hist:
        fig = px.histogram(df, x="risk_cin2_3yr", color="priority",
                           color_discrete_map=PRIORITY_COLORS,
                           category_orders={"priority": PRIORITY_ORDER},
                           nbins=24, barmode="stack",
                           labels={"risk_cin2_3yr": "3-yr risk"})
        fig.update_layout(showlegend=False, height=220, margin=dict(t=10,b=10),
                          xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with c_scatter:
        fig = px.scatter(df, x="age", y="risk_cin2_3yr", color="priority",
                         color_discrete_map=PRIORITY_COLORS,
                         category_orders={"priority": PRIORITY_ORDER},
                         opacity=0.55, labels={"age":"Age","risk_cin2_3yr":"3-yr risk"})
        fig.update_layout(showlegend=False, height=220, margin=dict(t=10,b=10),
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    q_cols = ["priority","person_id","screening_date","age","hpv_genotype","cytology_result",
              "risk_cin2_1yr","risk_cin2_3yr","risk_cin2_5yr","recommended_action","review_reason"]
    avail = [c for c in q_cols if c in filtered.columns]
    queue = filtered.sort_values(["priority_rank","risk_cin2_3yr"], ascending=[True,False])[avail]
    st.dataframe(pct_cols(queue), use_container_width=True, hide_index=True,
                 column_config=risk_column_config())

    st.markdown("---")
    st.subheader("Patient Drill-Down — Individual Risk Profile")
    _patient_detail(filtered)


def _patient_detail(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No patients match current filters.")
        return
    df = df.sort_values(["priority_rank","risk_cin2_3yr"], ascending=[True,False])
    labels = (df["person_id"].astype(str) + " | " + df["priority"].astype(str)
              + " | 3yr " + df["risk_cin2_3yr"].map(fmt_pct))
    sel = st.selectbox("Select patient", labels.tolist())
    row = df.iloc[labels.tolist().index(sel)]

    c_bar, c_tbl = st.columns(2)
    with c_bar:
        rv = {"1-year": float(row["risk_cin2_1yr"]),
              "3-year": float(row["risk_cin2_3yr"]),
              "5-year": float(row["risk_cin2_5yr"])}
        fig = go.Figure(go.Bar(
            x=list(rv.keys()), y=list(rv.values()),
            marker_color=["#aec7e8","#ff7f0e","#d62728"],
            text=[f"{v:.1%}" for v in rv.values()], textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(tickformat=".0%", range=[0, min(1.0, max(rv.values())*1.45)]),
            height=230, margin=dict(t=30,b=10,l=10,r=10), showlegend=False,
            title=dict(text="CIN2+ Risk by Prediction Window"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Model estimates only — should support, not replace, clinical judgement.")

    with c_tbl:
        dcols = ["person_id","screening_date","age","hpv_test_result","hpv_genotype",
                 "cytology_result","histology_result","persistent_hrhpv",
                 "priority","recommended_action"]
        avail = [c for c in dcols if c in row.index]
        st.table(pd.DataFrame({"Field": avail, "Value": [row[c] for c in avail]}))

    if DB_PATH.exists():
        ph = load_patient_history_from_db(str(row["person_id"]))
    else:
        raw = load_raw()
        ph = (raw[raw["person_id"].eq(row["person_id"])][["person_id"]+HISTORY_COLUMNS]
              .copy() if not raw.empty else pd.DataFrame())

    if ph.empty:
        return

    ph["screening_date"] = pd.to_datetime(ph["screening_date"])
    prev = ph[ph["screening_date"] < pd.to_datetime(row["screening_date"])]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Prior Screens",        f"{len(prev):,}")
    h2.metric("Prior hrHPV+",         f"{int(prev['hrhpv_positive'].sum()):,}")
    h3.metric("Prior Abnormal Cytol.", f"{int(prev['cytology_result'].ne('NILM').sum()):,}")
    h4.metric("Prior CIN2+",          f"{int(prev['cin2plus_detected'].sum()):,}")

    hpv_c = {"negative":"#2ca02c","low_risk":"#aec7e8","other_hr":"#ffbb78",
              "hpv18":"#ff7f0e","hpv16":"#d62728","multiple_hr":"#9467bd"}
    tl = ph.sort_values("screening_date")
    fig = go.Figure(go.Scatter(
        x=tl["screening_date"], y=tl["hpv_genotype"], mode="markers+lines",
        marker=dict(color=[hpv_c.get(g,"#ccc") for g in tl["hpv_genotype"]], size=14),
        line=dict(color="#ddd", width=1),
        text=tl["cytology_result"],
        hovertemplate="Date: %{x|%Y-%m-%d}<br>HPV: %{y}<br>Cytology: %{text}<extra></extra>",
    ))
    fig.update_layout(height=200, margin=dict(t=20,b=10),
                      xaxis_title="Screening date", yaxis_title="HPV genotype",
                      plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    disp = tl.sort_values("screening_date", ascending=False).head(8).copy()
    disp["screening_date"] = disp["screening_date"].dt.date
    cols_show = [c for c in HISTORY_COLUMNS if c in disp.columns]
    st.dataframe(disp[cols_show], use_container_width=True, hide_index=True)


# ═══════════════════════ TAB 6 — DRIFT MONITORING ═══════════════════════════

def tab_monitoring() -> None:
    st.subheader("Data Drift Detection Strategy")

    c_strat, c_resp = st.columns(2)
    with c_strat:
        strat = pd.DataFrame({
            "Feature type": ["Continuous — e.g. age", "Categorical — e.g. HPV genotype"],
            "Statistical test": ["Kolmogorov-Smirnov (KS)", "Chi-squared (χ²)"],
            "What it detects": [
                "Shift in age/risk distribution of incoming patients",
                "Change in HPV type mix or cytology result proportions",
            ],
            "Significance level": ["α = 0.01", "α = 0.01"],
        })
        st.dataframe(strat, use_container_width=True, hide_index=True)

    with c_resp:
        st.markdown("**Response Protocol**")
        st.markdown("""
| Drift signal | Immediate response | Long-term |
|---|---|---|
| 1 feature drifts | ⚠️ Dashboard warning | Monitor next 2 batches |
| 2+ features drift | 🔴 Alert clinical team | Schedule model review |
| Persistent (3+ months) | Mark predictions as lower-confidence | **Trigger retraining** on new data |
        """)

    report = load_drift_report()
    if not report:
        st.info("No drift report found. Run `python src/prediction_pipeline.py --batch-date 2026-01-01`.")
        return

    st.markdown("---")
    status = report.get("overall_drift_detected", False)
    st.markdown(f"### Current Status: {'🔴 Drift Detected' if status else '✅ No Drift Detected'}")

    drift_rows = []
    for feat in ["age", "hpv_genotype", "cytology_result"]:
        if feat in report:
            item = report[feat]
            drift_rows.append({
                "Feature": feat, "Test": item.get("test"),
                "p-value": item.get("p_value"),
                "Drift detected": item.get("drift_detected"),
            })

    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        c_bar, c_tbl = st.columns(2)

        with c_bar:
            fig = px.bar(drift_df, x="Feature", y="p-value",
                         color="Drift detected",
                         color_discrete_map={True: "#d62728", False: "#2ca02c"},
                         title="Feature Drift p-values (log scale)",
                         text=drift_df["p-value"].map(lambda v: f"{v:.1e}"))
            fig.update_traces(textposition="outside")
            fig.add_hline(y=0.01, line_dash="dash", line_color="orange",
                          annotation_text="α = 0.01 threshold")
            fig.update_layout(height=300, margin=dict(t=40,b=20),
                              yaxis_type="log", yaxis_title="p-value (log scale)")
            st.plotly_chart(fig, use_container_width=True)

        with c_tbl:
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
            for _, r in drift_df.iterrows():
                if r["Drift detected"]:
                    st.error(f"**{r['Feature']}** — p = {r['p-value']:.1e}: distribution shifted from training baseline.")
                else:
                    st.success(f"**{r['Feature']}** — p = {r['p-value']:.3f}: stable.")

    raw = load_raw()
    pred_path = latest_prediction_file()
    if pred_path is None or raw.empty:
        return

    st.markdown("---")
    st.subheader("Reference (Training) vs Current Batch — Distribution Comparison")
    batch = load_predictions(str(pred_path))
    ref = raw[pd.to_datetime(raw["screening_date"]).dt.year.between(2010, 2022)]

    c_age, c_geno = st.columns(2)
    with c_age:
        fig = go.Figure()
        for data, name, col in [(ref["age"], "Training reference", "#4e79a7"),
                                (batch["age"], "Current batch", "#d62728")]:
            fig.add_trace(go.Histogram(
                x=data, name=name, opacity=0.62, nbinsx=20,
                histnorm="probability", marker_color=col,
            ))
        fig.update_layout(barmode="overlay", height=270, margin=dict(t=30,b=20),
                          xaxis_title="Age", yaxis_title="Proportion",
                          title="Age Distribution", legend=dict(y=1.12, orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with c_geno:
        ref_geno = ref["hpv_genotype"].value_counts(normalize=True).rename("Reference")
        cur_geno = batch["hpv_genotype"].value_counts(normalize=True).rename("Current")
        gdf = pd.concat([ref_geno, cur_geno], axis=1).fillna(0).reset_index()
        gdf.columns = ["hpv_genotype", "Reference", "Current"]
        fig = go.Figure()
        for col_name, color in [("Reference","#4e79a7"), ("Current","#d62728")]:
            fig.add_trace(go.Bar(name=col_name, x=gdf["hpv_genotype"],
                                 y=gdf[col_name], marker_color=color, opacity=0.82))
        fig.update_layout(barmode="group", height=270, margin=dict(t=30,b=20),
                          xaxis_title="HPV Genotype", yaxis_title="Proportion",
                          yaxis_tickformat=".0%", title="HPV Genotype Distribution",
                          legend=dict(y=1.12, orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cytology Distribution — Reference vs Current")
    cyto_order = ["NILM","ASC-US","LSIL","ASC-H","HSIL","AGC"]
    ref_cyto = ref["cytology_result"].value_counts(normalize=True).reindex(cyto_order).fillna(0).rename("Reference")
    cur_cyto = batch["cytology_result"].value_counts(normalize=True).reindex(cyto_order).fillna(0).rename("Current")
    cdf = pd.concat([ref_cyto, cur_cyto], axis=1).reset_index()
    cdf.columns = ["cytology", "Reference", "Current"]
    fig = go.Figure()
    for col_name, color in [("Reference","#4e79a7"), ("Current","#d62728")]:
        fig.add_trace(go.Bar(name=col_name, x=cdf["cytology"],
                             y=cdf[col_name], marker_color=color, opacity=0.82))
    fig.update_layout(barmode="group", height=270, margin=dict(t=30,b=20),
                      xaxis_title="Cytology Result", yaxis_title="Proportion",
                      yaxis_tickformat=".1%",
                      legend=dict(y=1.12, orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════ MAIN ═════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="CerviRisk — ML Pipeline",
        layout="wide",
        page_icon="🏥",
    )
    st.title("CerviRisk — End-to-End ML Pipeline for Cervical Cancer Screening")

    if DB_PATH.exists():
        st.sidebar.success(f"✅ SQLite DB connected\n`{DB_PATH}`")
    else:
        st.sidebar.warning("SQLite DB not found.")

    pred_path = latest_prediction_file()
    df = pd.DataFrame()
    if pred_path:
        with st.sidebar:
            pred_files = sorted(PREDICTION_ROOT.glob("batch_date=*/predictions.parquet"), reverse=True)
            sel_path = st.selectbox("Prediction batch", pred_files, format_func=str)
        df = load_predictions(str(sel_path))

    filtered = pd.DataFrame()
    if not df.empty:
        df["priority_rank"] = priority_rank(df["priority"])
        filtered = _apply_filters(df)

    tabs = st.tabs([
        "🗺️  Pipeline & Code",
        "🧬  Data Generation",
        "🔧  Feature Engineering",
        "🤖  Model Training",
        "🚀  Inference & Serving",
        "📡  Drift Monitoring",
    ])

    with tabs[0]:
        tab_pipeline()
    with tabs[1]:
        tab_data()
    with tabs[2]:
        tab_features()
    with tabs[3]:
        tab_training()
    with tabs[4]:
        if df.empty:
            st.error("No prediction file found. Run `python src/prediction_pipeline.py --batch-date 2026-01-01` first.")
        else:
            tab_serving(df, filtered)
    with tabs[5]:
        tab_monitoring()


if __name__ == "__main__":
    main()
