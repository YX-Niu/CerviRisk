from __future__ import annotations

import matplotlib
import sys
from pathlib import Path

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.config import CLUSTERING, MODELING, PROCESSED_DIR, REPORT_DIR
except ModuleNotFoundError:
    from config import CLUSTERING, MODELING, PROCESSED_DIR, REPORT_DIR


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(PROCESSED_DIR / "features_all.parquet")
    features["screening_date"] = pd.to_datetime(features["screening_date"])
    eligible = features.loc[features["screening_date"] <= pd.Timestamp(CLUSTERING.eligibility_cutoff)].copy()
    latest = (
        eligible.sort_values(["person_id", "screening_date"])
        .groupby("person_id", as_index=False)
        .tail(1)
        .copy()
    )

    cluster_features = [
        "age",
        "n_previous_screens",
        "time_since_last_screen",
        "ever_had_abnormal_cyto",
        "ever_had_hrHPV",
        "persistent_hrhpv",
        "hrhpv_positive",
        "hpv_genotype_hpv16",
        "hpv_genotype_hpv18",
        "hpv_genotype_multiple_hr",
        "cytology_result_HSIL",
        "cytology_result_ASC-H",
        MODELING.primary_target,
    ]
    cluster_features = [c for c in cluster_features if c in latest.columns]
    x = latest[cluster_features].fillna(0)
    scaled = StandardScaler().fit_transform(x)

    model = KMeans(n_clusters=CLUSTERING.n_clusters, random_state=CLUSTERING.random_seed, n_init=20)
    latest["cluster"] = model.fit_predict(scaled)
    latest.to_parquet(PROCESSED_DIR / "latest_clustered.parquet", index=False)

    profile_cols = {
        "person_id": "count",
        "age": "mean",
        MODELING.primary_target: "mean",
        "hrhpv_positive": "mean",
        "persistent_hrhpv": "mean",
        "ever_had_abnormal_cyto": "mean",
        "hpv_genotype_hpv16": "mean",
        "hpv_genotype_hpv18": "mean",
        "cytology_result_HSIL": "mean",
    }
    profile_cols = {k: v for k, v in profile_cols.items() if k in latest.columns}
    profiles = latest.groupby("cluster").agg(profile_cols).rename(columns={"person_id": "n"})
    profiles.to_csv(REPORT_DIR / "cluster_profiles.csv")

    plot_profiles = profiles.drop(columns=["n"], errors="ignore")
    ax = plot_profiles.plot(kind="bar", figsize=(11, 6))
    ax.set_title("CerviRisk cluster profiles")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean / proportion")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "cluster_profiles.png", dpi=180)
    plt.close()

    print(profiles.round(3).to_string())
    print(f"Saved {REPORT_DIR / 'cluster_profiles.png'}")


if __name__ == "__main__":
    main()
