from __future__ import annotations

import requests


payload = {
    "age": 37,
    "region": "central",
    "hpv_vaccinated": False,
    "smoking_status": "current",
    "immunosuppressed": False,
    "parity": 1,
    "hpv_test_result": "positive",
    "hpv_genotype": "hpv16",
    "hrhpv_positive": True,
    "cytology_result": "ASC-H",
    "colposcopy_performed": True,
    "histology_result": "none",
    "persistent_hrhpv": True,
    "n_previous_screens": 3,
    "time_since_last_screen": 2.1,
    "ever_had_abnormal_cyto": True,
    "ever_had_hrHPV": True,
}


def main() -> None:
    response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
