from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


PREDICTIONS_PATH = Path("data/predictions/batch_date=2024-01-01/predictions.parquet")
OUTPUT_PATH = Path("docs/dashboard_screenshot.png")


def draw_risk_bar(ax, x: float, y: float, width: float, value: float) -> None:
    bar_width = width * 0.58
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            bar_width,
            0.008,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor="#ffe8e8",
            edgecolor="#f4d0d0",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            bar_width * float(value),
            0.008,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor="#ff4b4b",
            edgecolor="none",
        )
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PREDICTIONS_PATH)
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "routine": 3}
    df = (
        df.assign(rank=df["priority"].map(priority_order))
        .sort_values(["rank", "risk_cin2_3yr"], ascending=[True, False])
        .head(8)
    )

    fig = plt.figure(figsize=(16, 9), facecolor="#f7f8fb")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 0.24, 1, facecolor="#eef0f5", edgecolor="none"))
    ax.text(0.025, 0.90, "Prediction batch", fontsize=12, color="#303241", weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.83),
            0.185,
            0.045,
            boxstyle="round,pad=0.01,rounding_size=0.008",
            facecolor="white",
            edgecolor="none",
        )
    )
    ax.text(0.035, 0.848, "data/predictions/batch_date=2024-01-01", fontsize=9.5, color="#303241")
    ax.text(0.025, 0.75, "Filters", fontsize=18, color="#303241", weight="bold")
    ax.text(0.025, 0.69, "Priority", fontsize=12, color="#303241")
    ax.text(
        0.035,
        0.64,
        " urgent ",
        fontsize=12,
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ff4b4b", "edgecolor": "none"},
    )
    ax.text(
        0.105,
        0.64,
        " high ",
        fontsize=12,
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ff4b4b", "edgecolor": "none"},
    )
    ax.text(0.025, 0.56, "Minimum 3-year risk", fontsize=12, color="#303241")
    ax.plot([0.03, 0.21], [0.525, 0.525], color="#dce1ea", lw=4, solid_capstyle="round")
    ax.plot([0.03, 0.085], [0.525, 0.525], color="#ff4b4b", lw=4, solid_capstyle="round")
    ax.scatter([0.085], [0.525], s=90, color="#ff4b4b")
    ax.text(0.074, 0.547, "0.20", fontsize=10, color="#ff4b4b")
    for label, y in [("HPV genotype", 0.44), ("Cytology", 0.31)]:
        ax.text(0.025, y, label, fontsize=12, color="#303241")
        ax.add_patch(
            FancyBboxPatch(
                (0.025, y - 0.06),
                0.185,
                0.045,
                boxstyle="round,pad=0.01,rounding_size=0.008",
                facecolor="white",
                edgecolor="none",
            )
        )
        ax.text(0.035, y - 0.043, "Choose options", fontsize=11, color="#8a8d99")

    ax.text(0.30, 0.83, "CerviRisk Monthly Triage", fontsize=34, color="#303241", weight="bold")
    metrics = [
        ("Monthly Records", "750"),
        ("Urgent Reviews", "107"),
        ("High or Urgent", "246"),
        ("Mean 3-year Risk", "34.5%"),
    ]
    for x, (label, value) in zip([0.30, 0.47, 0.64, 0.81], metrics):
        ax.text(x, 0.755, label, fontsize=12, color="#303241")
        ax.text(x, 0.69, value, fontsize=28, color="#303241")
    ax.text(0.30, 0.60, "Review Queue", fontsize=12, color="#ff4b4b")
    ax.plot([0.30, 0.365], [0.575, 0.575], color="#ff4b4b", lw=2)
    ax.text(0.38, 0.60, "Patient Detail", fontsize=12, color="#303241")
    ax.text(0.46, 0.60, "Monitoring", fontsize=12, color="#303241")

    cols = [
        "priority",
        "person_id",
        "screening_date",
        "age",
        "hpv_genotype",
        "cytology",
        "1-year CIN2+ risk",
        "3-year CIN2+ risk",
        "5-year CIN2+ risk",
    ]
    widths = [0.065, 0.075, 0.105, 0.045, 0.105, 0.085, 0.13, 0.13, 0.13]
    x0, y0, row_height = 0.30, 0.525, 0.052
    ax.add_patch(Rectangle((x0, y0), sum(widths), row_height, facecolor="#f2f3f6", edgecolor="#dddddf"))
    x = x0
    for col, width in zip(cols, widths):
        ax.text(x + 0.006, y0 + 0.020, col, fontsize=9.2, color="#8a8d99")
        x += width

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        y = y0 - index * row_height
        ax.add_patch(Rectangle((x0, y), sum(widths), row_height, facecolor="white", edgecolor="#e5e5e8"))
        values = [
            row["priority"],
            row["person_id"],
            str(pd.to_datetime(row["screening_date"]).date()),
            str(int(row["age"])),
            row["hpv_genotype"],
            row["cytology_result"],
        ]
        x = x0
        for value, width in zip(values, widths[:6]):
            ax.text(x + 0.006, y + 0.020, value, fontsize=9.5, color="#303241")
            x += width
        for risk, width in zip(
            [row["risk_cin2_1yr"], row["risk_cin2_3yr"], row["risk_cin2_5yr"]],
            widths[6:],
        ):
            draw_risk_bar(ax, x + 0.008, y + 0.023, width, risk)
            ax.text(x + width * 0.68, y + 0.016, f"{risk:.1%}", fontsize=9.2, color="#303241")
            x += width

    ax.text(
        0.30,
        0.065,
        "Clinician-facing dashboard preview. Synthetic/de-identified patient IDs are shown; no names, addresses, national IDs, or contact details are displayed.",
        fontsize=10,
        color="#666a75",
    )
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
