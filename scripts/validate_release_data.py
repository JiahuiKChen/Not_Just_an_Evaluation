#!/usr/bin/env python3
"""Validate the schemas, counts, and checksums of the released data files."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "dailydialog_sense_subset.tsv": "60bf8b9237338efaefffd79e0888d094281a8d0d7732f7fe133175969f4918a7",
    "human_annotations.csv": "71d717a5fc57eaa9cb71e3f643ba0de14c2f2e0db150583d40c4b1d703d37222",
}


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def validate_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[path.name], (path.name, digest)


def validate_stimuli() -> None:
    path = ROOT / "data/dailydialog_sense_subset.tsv"
    validate_checksum(path)
    rows = read_rows(path, "\t")
    assert len(rows) == 180, len(rows)
    assert len({row["id"] for row in rows}) == 180
    assert Counter(row["word"] for row in rows) == {"just": 60, "only": 60, "not": 60}
    assert Counter(row["sense"] for row in rows) == {
        "exclusive": 90,
        "non-exclusive": 30,
        "truth-conditional control": 60,
    }
    assert sum(row["id"].startswith("just_e_") for row in rows) == 30
    assert sum(row["id"].startswith("just_j_") for row in rows) == 30
    required = {"id", "word", "sense", "context", "w_word", "wo_word", "followup"}
    assert set(rows[0]) == required
    for row in rows:
        assert all(row[column].strip() for column in required)
        assert row["w_word"] != row["wo_word"]


def validate_human() -> None:
    path = ROOT / "data/human_annotations.csv"
    validate_checksum(path)
    rows = read_rows(path, ",")
    assert len(rows) == 2160, len(rows)
    assert Counter(row["participant_id"] for row in rows) == {
        str(participant): 240 for participant in range(1, 10)
    }
    assert Counter(row["word"] for row in rows) == {
        "just": 540,
        "only": 540,
        "not": 540,
        "control": 540,
    }
    assert {int(row["prefers_w_word"]) for row in rows} <= {-1, 0, 1}
    assert {int(row["preference_strength"]) for row in rows} <= set(range(1, 8))
    item_counts = Counter(row["comparison_id"] for row in rows)
    assert len(item_counts) == 240
    assert set(item_counts.values()) == {9}


if __name__ == "__main__":
    validate_stimuli()
    validate_human()
    print("Data validation passed: 180 stimuli and 2,160 human judgments.")
