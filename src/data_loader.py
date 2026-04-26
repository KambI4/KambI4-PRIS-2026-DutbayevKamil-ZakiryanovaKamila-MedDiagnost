from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models import Disease


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_diseases_from_json(path: Path | None = None) -> list[Disease]:
    path = path or DATA_DIR / "diseases.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return [
        Disease(
            name=item["name"],
            symptoms=item["symptoms"],
            medicines=item["medicines"],
            severity=item.get("severity", "medium"),
            urgency=item.get("urgency", "plan"),
            advice=item.get("advice", []),
            when_to_seek_help=item.get("when_to_seek_help", []),
        )
        for item in payload
    ]


def load_symptoms_catalog(path: Path | None = None) -> list[dict[str, str]]:
    path = path or DATA_DIR / "symptoms.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_medicines_catalog(path: Path | None = None) -> list[dict[str, str]]:
    path = path or DATA_DIR / "medicines.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
