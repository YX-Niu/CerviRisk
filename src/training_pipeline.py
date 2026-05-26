from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import RAW_PATH, SIMULATION


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CerviRisk models from historical longitudinal screening data.")
    parser.add_argument("--raw-input", type=Path, default=RAW_PATH)
    parser.add_argument(
        "--bootstrap-synthetic",
        action="store_true",
        help="Generate synthetic historical raw data before training. Use --raw-input for database/export data.",
    )
    parser.add_argument("--bootstrap-n-women", type=int, default=SIMULATION.n_women)
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Simulate data as monthly batches into data/incoming/ and train from those files. "
             "This is the unified data flow: same format for training and monthly prediction.",
    )
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    raw_input = args.raw_input
    steps = []

    if args.monthly:
        # Unified flow: simulator writes monthly batches → preprocess reads them
        steps.append((
            "simulate monthly batches",
            [python, "src/ingestion/simulator.py", "--monthly",
             "--n-women", str(args.bootstrap_n_women)],
        ))
        steps.append((
            "preprocessing and feature engineering",
            [python, "src/features/preprocess.py", "--from-monthly"],
        ))
    elif args.bootstrap_synthetic:
        steps.append((
            "historical synthetic raw data bootstrap",
            [python, "src/ingestion/simulator.py",
             "--n-women", str(args.bootstrap_n_women), "--output", str(raw_input)],
        ))
        steps.append((
            "preprocessing and feature engineering",
            [python, "src/features/preprocess.py", "--raw-input", str(raw_input)],
        ))
    else:
        if not (ROOT / raw_input).exists():
            raise FileNotFoundError(
                f"{raw_input} does not exist. Options:\n"
                "  --monthly              simulate + train from monthly batches (recommended)\n"
                "  --bootstrap-synthetic  generate one flat raw file then train\n"
                "  --raw-input <path>     point to an existing Parquet export"
            )
        steps.append((
            "preprocessing and feature engineering",
            [python, "src/features/preprocess.py", "--raw-input", str(raw_input)],
        ))

    steps.extend([
        ("model training", [python, "src/modeling/train.py"]),
    ])
    if not args.skip_clustering:
        steps.append(("unsupervised cluster profiling", [python, "src/modeling/clustering.py"]))
    if not args.skip_explain:
        steps.append(("model explanations", [python, "src/modeling/explain.py"]))

    for name, command in steps:
        run_step(name, command)

    print("\nTraining pipeline complete. Artifacts are in data/processed/, models/, and reports/.")


if __name__ == "__main__":
    main()
