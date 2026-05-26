from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import RAW_PATH


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CerviRisk models from historical screening data.\n\n"
            "Expects data to already exist at --raw-input (default: data/raw/screening_records.parquet).\n"
            "Run `python src/ingestion/simulator.py` first if you need to generate synthetic data."
        )
    )
    parser.add_argument("--raw-input", type=Path, default=RAW_PATH)
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not (ROOT / args.raw_input).exists():
        print(f"Error: raw data not found at {args.raw_input}")
        print("Run first:  python src/ingestion/simulator.py")
        sys.exit(1)

    steps: list[tuple[str, list[str]]] = [
        ("preprocessing and feature engineering",
         [python, "src/features/preprocess.py", "--raw-input", str(args.raw_input)]),
        ("model training",
         [python, "src/modeling/train.py"]),
    ]
    if not args.skip_clustering:
        steps.append(("unsupervised cluster profiling", [python, "src/modeling/clustering.py"]))
    if not args.skip_explain:
        steps.append(("model explanations", [python, "src/modeling/explain.py"]))

    for name, command in steps:
        run_step(name, command)

    print("\nTraining pipeline complete. Artifacts are in data/processed/, models/, and reports/.")


if __name__ == "__main__":
    main()
