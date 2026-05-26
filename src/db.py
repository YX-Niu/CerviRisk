from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/cervirisk.db")

_CREATE_PATIENTS = """
CREATE TABLE IF NOT EXISTS patients (
    person_id       TEXT PRIMARY KEY,
    birth_year      INTEGER NOT NULL,
    region          TEXT NOT NULL,
    hpv_vaccinated  INTEGER NOT NULL,
    smoking_status  TEXT NOT NULL,
    immunosuppressed INTEGER NOT NULL,
    parity          INTEGER NOT NULL
);
"""

_CREATE_SCREENING_VISITS = """
CREATE TABLE IF NOT EXISTS screening_visits (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id           TEXT NOT NULL REFERENCES patients(person_id),
    screening_date      TEXT NOT NULL,
    visit_index         INTEGER NOT NULL,
    age                 REAL NOT NULL,
    hpv_test_result     TEXT NOT NULL,
    hpv_genotype        TEXT NOT NULL,
    hrhpv_positive      INTEGER NOT NULL,
    cytology_result     TEXT NOT NULL,
    colposcopy_performed INTEGER NOT NULL,
    histology_result    TEXT NOT NULL,
    cin2plus_detected   INTEGER NOT NULL,
    treatment_performed INTEGER NOT NULL,
    persistent_hrhpv    INTEGER NOT NULL,
    ingestion_batch_date TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_visits_person ON screening_visits(person_id);",
    "CREATE INDEX IF NOT EXISTS idx_visits_date ON screening_visits(screening_date);",
]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(_CREATE_PATIENTS)
        conn.execute(_CREATE_SCREENING_VISITS)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)


def insert_screening_records(df: pd.DataFrame, db_path: Path = DB_PATH, replace: bool = False) -> int:
    init_db(db_path)

    patient_cols = ["person_id", "birth_year", "region", "hpv_vaccinated", "smoking_status", "immunosuppressed", "parity"]
    patients = df[patient_cols].drop_duplicates("person_id")

    visit_cols = [
        "person_id", "screening_date", "visit_index", "age",
        "hpv_test_result", "hpv_genotype", "hrhpv_positive",
        "cytology_result", "colposcopy_performed", "histology_result",
        "cin2plus_detected", "treatment_performed", "persistent_hrhpv",
    ]
    visits = df[[c for c in visit_cols if c in df.columns]].copy()
    visits["screening_date"] = visits["screening_date"].astype(str)

    if "ingestion_batch_date" in df.columns:
        visits["ingestion_batch_date"] = df["ingestion_batch_date"].astype(str)
    else:
        visits["ingestion_batch_date"] = None

    conflict = "REPLACE" if replace else "IGNORE"

    patient_placeholders = ", ".join("?" * len(patient_cols))
    patient_sql = f"INSERT OR IGNORE INTO patients ({', '.join(patient_cols)}) VALUES ({patient_placeholders})"

    visit_cols_list = list(visits.columns)
    visit_placeholders = ", ".join("?" * len(visit_cols_list))
    visit_sql = f"INSERT INTO screening_visits ({', '.join(visit_cols_list)}) VALUES ({visit_placeholders})"

    with get_connection(db_path) as conn:
        conn.executemany(patient_sql, patients.itertuples(index=False, name=None))
        conn.executemany(visit_sql, visits.itertuples(index=False, name=None))

    return len(visits)


def query_all_records(db_path: Path = DB_PATH) -> pd.DataFrame:
    sql = """
        SELECT
            p.person_id, p.birth_year, p.region,
            p.hpv_vaccinated, p.smoking_status, p.immunosuppressed, p.parity,
            v.screening_date, v.visit_index, v.age,
            v.hpv_test_result, v.hpv_genotype, v.hrhpv_positive,
            v.cytology_result, v.colposcopy_performed, v.histology_result,
            v.cin2plus_detected, v.treatment_performed, v.persistent_hrhpv,
            v.ingestion_batch_date
        FROM patients p
        JOIN screening_visits v USING (person_id)
        ORDER BY v.screening_date, p.person_id
    """
    with get_connection(db_path) as conn:
        return pd.read_sql(sql, conn, parse_dates=["screening_date"])


def query_patient_history(person_id: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    sql = """
        SELECT
            v.screening_date, v.age,
            v.hpv_test_result, v.hpv_genotype, v.hrhpv_positive,
            v.cytology_result, v.histology_result,
            v.persistent_hrhpv, v.cin2plus_detected, v.treatment_performed
        FROM screening_visits v
        WHERE v.person_id = ?
        ORDER BY v.screening_date
    """
    with get_connection(db_path) as conn:
        return pd.read_sql(sql, conn, params=(person_id,), parse_dates=["screening_date"])


def query_batch_by_date(batch_date: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    sql = """
        SELECT
            p.person_id, p.birth_year, p.region,
            p.hpv_vaccinated, p.smoking_status, p.immunosuppressed, p.parity,
            v.screening_date, v.visit_index, v.age,
            v.hpv_test_result, v.hpv_genotype, v.hrhpv_positive,
            v.cytology_result, v.colposcopy_performed, v.histology_result,
            v.cin2plus_detected, v.treatment_performed, v.persistent_hrhpv,
            v.ingestion_batch_date
        FROM patients p
        JOIN screening_visits v USING (person_id)
        WHERE v.ingestion_batch_date = ?
        ORDER BY v.screening_date, p.person_id
    """
    with get_connection(db_path) as conn:
        return pd.read_sql(sql, conn, params=(batch_date,), parse_dates=["screening_date"])


def db_stats(db_path: Path = DB_PATH) -> dict:
    with get_connection(db_path) as conn:
        n_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        n_visits = conn.execute("SELECT COUNT(*) FROM screening_visits").fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(screening_date), MAX(screening_date) FROM screening_visits"
        ).fetchone()
    return {
        "patients": n_patients,
        "visits": n_visits,
        "earliest_visit": date_range[0],
        "latest_visit": date_range[1],
    }
