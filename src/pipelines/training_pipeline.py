from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import RAW_PATH, SIMULATION


ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    raw_input = args.raw_input
    steps = []

    if args.bootstrap_synthetic:
        steps.append(
            (
                "historical synthetic raw data bootstrap",
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

    for name, command in steps:
        run_step(name, command)

    print("\nTraining pipeline complete. Artifacts are in data/processed/, models/, and reports/.")


if __name__ == "__main__":
    main()
