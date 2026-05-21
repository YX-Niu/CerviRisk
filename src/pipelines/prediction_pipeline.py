from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import INCOMING_DIR, INGESTION


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_PATH = INCOMING_DIR / f"batch_date={INGESTION.batch_date.isoformat()}" / "screening_records.parquet"
DEFAULT_PREDICTION_PATH = Path("data/predictions") / f"batch_date={INGESTION.batch_date.isoformat()}" / "predictions.parquet"


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monthly CerviRisk prediction and monitoring pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Existing monthly batch parquet. If omitted, a synthetic monthly batch is generated.")
    parser.add_argument("--batch-date", default=INGESTION.batch_date.isoformat())
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--inject-demo-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    batch_path = args.input or DEFAULT_BATCH_PATH

    if args.input is None:
        batch_path = INCOMING_DIR / f"batch_date={args.batch_date}" / "screening_records.parquet"
        run_step("monthly ingestion batch", [python, "src/data/ingest.py", "--batch-date", args.batch_date])

    run_step(
        "batch risk prediction",
        [
            python,
            "src/models/predict_batch.py",
            "--input",
            str(batch_path),
            "--output",
            str(args.prediction_output),
        ],
    )

    command = [
        python,
        "src/monitor/drift.py",
        "--current-batch",
        str(batch_path),
    ]
    if args.inject_demo_drift:
        command.append("--inject-demo-drift")
    run_step("data drift monitoring", command)

    print("\nPrediction pipeline complete. Outputs are in data/predictions/ and reports/.")


if __name__ == "__main__":
    main()
