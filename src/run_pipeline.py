from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from src.config import INCOMING_DIR, INGESTION, RAW_PATH, SIMULATION
except ModuleNotFoundError:
    from config import INCOMING_DIR, INGESTION, RAW_PATH, SIMULATION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = RAW_PATH
DEFAULT_BATCH_PATH = INCOMING_DIR / f"batch_date={INGESTION.batch_date.isoformat()}" / "screening_records.parquet"


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete CerviRisk ML pipeline end to end.")
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_PATH, help="Raw longitudinal screening records to preprocess.")
    parser.add_argument(
        "--bootstrap-synthetic",
        action="store_true",
        help="Generate demo raw data before preprocessing. Keep this off when raw data comes from a database/export.",
    )
    parser.add_argument("--bootstrap-n-women", type=int, default=SIMULATION.n_women, help="Number of synthetic women for demo bootstrap.")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip the simulated monthly incoming batch step.")
    parser.add_argument("--batch-date", default=INGESTION.batch_date.isoformat(), help="Batch date for simulated monthly ingestion.")
    parser.add_argument("--current-batch", type=Path, default=DEFAULT_BATCH_PATH, help="Batch used by drift monitoring.")
    parser.add_argument("--skip-explain", action="store_true", help="Skip SHAP plots for a faster smoke run.")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip K-Means cohort profiling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    raw_input = args.raw_input
    steps = []

    if args.bootstrap_synthetic:
        steps.append(
            (
                "demo raw data bootstrap",
                [
                    python,
                    "src/data/simulator.py",
                    "--n-women",
                    str(args.bootstrap_n_women),
                    "--output",
                    str(raw_input),
                ],
            )
        )
    elif not (ROOT / raw_input).exists():
        raise FileNotFoundError(
            f"{raw_input} does not exist. Provide --raw-input from your ingestion layer, "
            "or run with --bootstrap-synthetic for the self-contained demo."
        )

    steps.extend(
        [
            ("preprocessing and feature engineering", [python, "src/features/preprocess.py", "--raw-input", str(raw_input)]),
            ("model training", [python, "src/models/train.py"]),
        ]
    )
    if not args.skip_clustering:
        steps.append(("unsupervised cluster profiling", [python, "src/models/clustering.py"]))
    if not args.skip_explain:
        steps.append(("model explanations", [python, "src/models/explain.py"]))
    if not args.skip_ingestion:
        steps.append(("monthly ingestion batch for serving/monitoring", [python, "src/data/ingest.py", "--batch-date", args.batch_date]))
    steps.append(
        (
            "data drift monitoring",
            [
                python,
                "src/monitor/drift.py",
                "--current-batch",
                str(args.current_batch),
                "--inject-demo-drift",
            ],
        )
    )

    for name, command in steps:
        run_step(name, command)

    print("\nPipeline complete. Artifacts are in data/, models/, and reports/.")


if __name__ == "__main__":
    main()
