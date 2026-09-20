"""Internal CSV parsing and sample-key matching for benchmark result adapters.

This helper computes no metric. Its scope is reading adapter CSV rows, coercing
selected fields to floats, and indexing rows by conservative path identifiers;
the calling adapter owns metric semantics and validation.

Basis: https://github.com/seruva19/ayase
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ayase.models import Sample


def _sample_keys(sample: Sample) -> tuple[str, ...]:
    """Return conservative result identifiers for a sample."""
    return (str(sample.path), sample.path.as_posix(), sample.path.name, sample.path.stem)


def _first_float(row: Dict[str, Any], fields: Iterable[str]) -> Optional[float]:
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _index_rows(rows: Iterable[Dict[str, Any]], id_fields: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for field in id_fields:
            raw = row.get(field)
            if raw is None:
                continue
            value = str(raw)
            path = Path(value)
            for key in (value, path.as_posix(), path.name, path.stem):
                indexed.setdefault(key, row)
    return indexed


def _read_csv(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

