from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from src.config import INGESTION, RAW_PATH, SIMULATION
except ModuleNotFoundError:
    from config import INGESTION, RAW_PATH, SIMULATION


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CerviRisk training plus monthly prediction pipelines.")
    parser.add_argument("--raw-input", type=Path, default=RAW_PATH)
    parser.add_argument("--bootstrap-synthetic", action="store_true")
    parser.add_argument("--bootstrap-n-women", type=int, default=SIMULATION.n_women)
    parser.add_argument("--batch-date", default=INGESTION.batch_date.isoformat())
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--inject-demo-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    training_command = [
        python,
        "src/pipelines/training_pipeline.py",
        "--raw-input",
        str(args.raw_input),
    ]
    if args.bootstrap_synthetic:
        training_command.append("--bootstrap-synthetic")
        training_command.extend(["--bootstrap-n-women", str(args.bootstrap_n_women)])
    if args.skip_clustering:
        training_command.append("--skip-clustering")
    if args.skip_explain:
        training_command.append("--skip-explain")

    prediction_command = [
        python,
        "src/pipelines/prediction_pipeline.py",
        "--batch-date",
        args.batch_date,
    ]
    if args.inject_demo_drift:
        prediction_command.append("--inject-demo-drift")

    run_step("training pipeline", training_command)
    run_step("monthly prediction pipeline", prediction_command)
    print("\nFull CerviRisk pipeline complete.")


if __name__ == "__main__":
    main()
