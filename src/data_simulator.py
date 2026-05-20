from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/screening_records.parquet")


@dataclass(frozen=True)
class SimulatorConfig:
    n_women: int = 22000
    start_year: int = 2010
    end_year: int = 2024
    seed: int = 42


CYTO_RESULTS = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]
HPV_GENOTYPES = ["negative", "hpv16", "hpv18", "other_hr", "low_risk", "multiple_hr"]
HISTOLOGY = ["none", "normal", "CIN1", "CIN2", "CIN3", "cancer"]
REGIONS = ["north", "south", "east", "west", "central"]
SMOKING = ["never", "former", "current"]


def _choice(rng: np.random.Generator, values: list[str], probs: list[float]) -> str:
    return str(rng.choice(values, p=np.array(probs) / np.sum(probs)))


def _hpv_for_visit(
    rng: np.random.Generator,
    age: float,
    latent_risk: float,
    previous_hpv: str | None,
    vaccinated: bool,
    immunosuppressed: bool,
) -> str:
    age_factor = 1.20 if age < 30 else 0.85 if age > 50 else 1.0
    vaccine_factor = 0.62 if vaccinated else 1.0
    immune_factor = 1.45 if immunosuppressed else 1.0
    base_hr = np.clip(0.14 * latent_risk * age_factor * vaccine_factor * immune_factor, 0.03, 0.55)

    if previous_hpv in {"hpv16", "hpv18", "other_hr", "multiple_hr"}:
        persistence = {"hpv16": 0.62, "hpv18": 0.48, "other_hr": 0.36, "multiple_hr": 0.58}[previous_hpv]
        if rng.random() < persistence * immune_factor:
            return previous_hpv

    if rng.random() > base_hr:
        return _choice(rng, HPV_GENOTYPES, [0.88, 0.015, 0.01, 0.04, 0.05, 0.005])

    genotype_probs = [0.0, 0.22, 0.12, 0.47, 0.0, 0.19]
    if vaccinated:
        genotype_probs = [0.0, 0.12, 0.07, 0.58, 0.0, 0.23]
    return _choice(rng, HPV_GENOTYPES, genotype_probs)


def _cytology_for_visit(rng: np.random.Generator, hpv: str, latent_risk: float) -> str:
    if hpv == "negative":
        probs = [0.93, 0.035, 0.02, 0.006, 0.006, 0.003]
    elif hpv == "low_risk":
        probs = [0.80, 0.09, 0.09, 0.008, 0.007, 0.005]
    elif hpv == "hpv16":
        probs = [0.48, 0.18, 0.19, 0.055, 0.085, 0.01]
    elif hpv == "hpv18":
        probs = [0.56, 0.16, 0.14, 0.045, 0.055, 0.04]
    elif hpv == "multiple_hr":
        probs = [0.44, 0.19, 0.22, 0.06, 0.08, 0.01]
    else:
        probs = [0.62, 0.16, 0.15, 0.035, 0.03, 0.005]
    tilt = np.clip((latent_risk - 1.0) * 0.04, -0.03, 0.08)
    probs = np.array(probs, dtype=float)
    probs[0] -= tilt
    probs[3:5] += tilt / 2
    return _choice(rng, CYTO_RESULTS, probs.clip(0.001, None).tolist())


def _histology_for_visit(
    rng: np.random.Generator,
    hpv: str,
    cytology: str,
    persistent_hr: bool,
    latent_risk: float,
) -> str:
    risk = {
        "negative": 0.003,
        "low_risk": 0.006,
        "other_hr": 0.035,
        "hpv18": 0.055,
        "hpv16": 0.085,
        "multiple_hr": 0.095,
    }[hpv]
    risk *= {"NILM": 0.55, "ASC-US": 1.1, "LSIL": 1.6, "ASC-H": 3.0, "HSIL": 5.0, "AGC": 2.4}[cytology]
    risk *= 1.75 if persistent_hr else 1.0
    risk = float(np.clip(risk * latent_risk, 0.001, 0.70))

    if rng.random() > risk:
        return _choice(rng, HISTOLOGY, [0.78, 0.16, 0.055, 0.003, 0.0015, 0.0005])
    severity = rng.random()
    if severity < 0.48:
        return "CIN1"
    if severity < 0.74:
        return "CIN2"
    if severity < 0.96:
        return "CIN3"
    return "cancer"


def generate_person_records(person_id: int, rng: np.random.Generator, cfg: SimulatorConfig) -> list[dict]:
    birth_year = int(rng.integers(1945, 1994))
    latest_first_year = max(cfg.start_year, cfg.end_year - 3)
    first_year = min(max(cfg.start_year, birth_year + int(rng.integers(23, 35))), latest_first_year)
    if first_year > cfg.end_year:
        first_year = int(rng.integers(cfg.start_year, cfg.end_year - 1))

    dates = []
    year = first_year
    while year <= cfg.end_year:
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        dates.append(pd.Timestamp(year=year, month=month, day=day))
        interval = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.20, 0.48, 0.20, 0.07]))
        year += interval

    if len(dates) < 2:
        second_year = min(cfg.end_year, dates[0].year + 3)
        dates.append(pd.Timestamp(year=second_year, month=12, day=15))

    vaccinated = bool(rng.random() < (0.42 if birth_year >= 1988 else 0.12))
    smoking = _choice(rng, SMOKING, [0.64, 0.19, 0.17])
    immunosuppressed = bool(rng.random() < 0.045)
    parity = int(np.clip(rng.poisson(1.5), 0, 6))
    region = _choice(rng, REGIONS, [0.22, 0.20, 0.18, 0.20, 0.20])

    latent_risk = rng.lognormal(mean=0.0, sigma=0.35)
    if smoking == "current":
        latent_risk *= 1.28
    if immunosuppressed:
        latent_risk *= 1.65

    rows = []
    previous_hpv = None
    previous_hr = False
    for visit_index, date in enumerate(sorted(dates)):
        age = round((date.year + date.dayofyear / 365.25) - birth_year, 1)
        hpv = _hpv_for_visit(rng, age, latent_risk, previous_hpv, vaccinated, immunosuppressed)
        hrhpv_positive = hpv in {"hpv16", "hpv18", "other_hr", "multiple_hr"}
        cytology = _cytology_for_visit(rng, hpv, latent_risk)
        persistent_hr = bool(hrhpv_positive and previous_hr)
        colposcopy = cytology in {"ASC-H", "HSIL", "AGC"} or (hrhpv_positive and rng.random() < 0.18)
        histology = _histology_for_visit(rng, hpv, cytology, persistent_hr, latent_risk) if colposcopy or rng.random() < 0.08 else "none"
        cin2plus = histology in {"CIN2", "CIN3", "cancer"}
        treatment = bool(histology in {"CIN2", "CIN3", "cancer"} and rng.random() < 0.88)

        rows.append(
            {
                "person_id": f"P{person_id:06d}",
                "screening_date": date,
                "visit_index": visit_index,
                "birth_year": birth_year,
                "age": age,
                "region": region,
                "hpv_vaccinated": vaccinated,
                "smoking_status": smoking,
                "immunosuppressed": immunosuppressed,
                "parity": parity,
                "hpv_test_result": "positive" if hpv != "negative" else "negative",
                "hpv_genotype": hpv,
                "hrhpv_positive": hrhpv_positive,
                "cytology_result": cytology,
                "colposcopy_performed": colposcopy,
                "histology_result": histology,
                "cin2plus_detected": cin2plus,
                "treatment_performed": treatment,
                "persistent_hrhpv": persistent_hr,
            }
        )
        previous_hpv = hpv
        previous_hr = hrhpv_positive

    return rows


def generate_screening_data(cfg: SimulatorConfig = SimulatorConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for person_number in range(1, cfg.n_women + 1):
        rows.extend(generate_person_records(person_number, rng, cfg))

    df = pd.DataFrame(rows).sort_values(["screening_date", "person_id"]).reset_index(drop=True)
    return df


def generate_screening_chunks(cfg: SimulatorConfig, chunk_size: int) -> Iterator[pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for person_number in range(1, cfg.n_women + 1):
        rows.extend(generate_person_records(person_number, rng, cfg))
        if person_number % chunk_size == 0:
            yield pd.DataFrame(rows).sort_values(["screening_date", "person_id"]).reset_index(drop=True)
            rows = []
    if rows:
        yield pd.DataFrame(rows).sort_values(["screening_date", "person_id"]).reset_index(drop=True)


def write_screening_data(
    cfg: SimulatorConfig,
    output: Path,
    chunk_size: int | None = None,
    partition_by_year: bool = False,
) -> pd.DataFrame | None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size is None or chunk_size >= cfg.n_women:
        df = generate_screening_data(cfg)
        if partition_by_year:
            _write_year_partitioned(df, output, chunk_id=1)
        else:
            df.to_parquet(output, index=False)
        return df

    row_count = 0
    for i, chunk in enumerate(generate_screening_chunks(cfg, chunk_size), start=1):
        if partition_by_year:
            _write_year_partitioned(chunk, output, chunk_id=i)
        else:
            chunk_path = output.with_name(f"{output.stem}_part_{i:04d}{output.suffix}")
            chunk.to_parquet(chunk_path, index=False)
        row_count += len(chunk)
        print(f"wrote chunk {i}: rows={len(chunk):,}, cumulative_rows={row_count:,}")
    return None


def _write_year_partitioned(df: pd.DataFrame, output: Path, chunk_id: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    by_year = df.assign(screening_year=df["screening_date"].dt.year)
    for year, year_df in by_year.groupby("screening_year", sort=True):
        year_dir = output / f"screening_year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        year_df.drop(columns=["screening_year"]).to_parquet(
            year_dir / f"part-{chunk_id:05d}.parquet",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic longitudinal CerviRisk screening records.")
    parser.add_argument("--n-women", type=int, default=22000)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RAW_PATH)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--partition-by-year", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulatorConfig(
        n_women=args.n_women,
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed,
    )
    df = write_screening_data(
        cfg=cfg,
        output=args.output,
        chunk_size=args.chunk_size,
        partition_by_year=args.partition_by_year,
    )

    print(f"Saved {args.output}")
    if df is not None:
        print("shape:", df.shape)
        print(df.describe(include="all").transpose().head(30))
        print("missing values:")
        print(df.isna().sum().loc[lambda s: s > 0])
    else:
        print("chunked write complete")


if __name__ == "__main__":
    main()
