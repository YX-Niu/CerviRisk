from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete CerviRisk ML pipeline end to end.")
    parser.add_argument("--skip-explain", action="store_true", help="Skip SHAP plots for a faster smoke run.")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip K-Means cohort profiling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    steps = [
        ("monthly ingestion batch", [python, "src/ingest.py", "--batch-date", "2024-01-01"]),
        ("historical raw data simulation", [python, "src/data_simulator.py"]),
        ("preprocessing and feature engineering", [python, "src/preprocess.py"]),
        ("model training", [python, "src/train.py"]),
    ]
    if not args.skip_clustering:
        steps.append(("unsupervised cluster profiling", [python, "src/clustering.py"]))
    if not args.skip_explain:
        steps.append(("model explanations", [python, "src/explain.py"]))
    steps.append(
        (
            "data drift monitoring",
            [
                python,
                "src/monitor.py",
                "--current-batch",
                "data/incoming/batch_date=2024-01-01/screening_records.parquet",
                "--inject-demo-drift",
            ],
        )
    )

    for name, command in steps:
        run_step(name, command)

    print("\nPipeline complete. Artifacts are in data/, models/, and reports/.")


if __name__ == "__main__":
    main()
