"""Strict TSV interfaces. No third-party dependencies are needed for scoring."""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PAIR_COLUMNS = ("source1_entity_id", "target_entity_id")
MATCH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def read_rows(path, required, exact=False):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)) or any(not h for h in headers):
            raise ValueError(f"{path}: missing, empty, or duplicate column names")
        if not set(required).issubset(headers) or (exact and set(headers) != set(required)):
            raise ValueError(f"{path}: expected columns {list(required)}, got {headers}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: wrong number of TSV fields")
            rows.append(row)
    return headers, rows


def identifier(value):
    if not isinstance(value, str):
        raise ValueError("IDs must be strings")
    value = value.strip()
    if not value or any(c.isspace() or c == ',' for c in value):
        raise ValueError(f"Invalid entity ID: {value!r}")
    return value


def validate_ids(ids):
    result = [identifier(value) for value in ids]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate Source 1 IDs")
    if not result:
        raise ValueError("Source 1 universe must not be empty")
    return result


def read_source_ids(path):
    _, rows = read_rows(path, ("entity_id",))
    return validate_ids(row["entity_id"] for row in rows)


def parse_matches(value):
    if not value.strip():
        return set()
    return {identifier(part) for part in value.split(',')}


def read_matches(path):
    _, rows = read_rows(path, MATCH_COLUMNS, exact=True)
    result = {}
    for row in rows:
        source = identifier(row["source1_entity_id"])
        if source in result:
            raise ValueError(f"{path}: duplicate Source 1 ID {source!r}")
        result[source] = parse_matches(row["matched_entity_ids"])
    if not result:
        raise ValueError(f"{path}: no Source 1 rows")
    return result


def validate_pairs(pairs, source_ids=None):
    result = [(identifier(source), identifier(target)) for source, target in pairs]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate candidate pairs")
    if source_ids is not None:
        unknown = {source for source, _ in result} - set(source_ids)
        if unknown:
            raise ValueError(f"Pairs contain unknown Source 1 IDs: {sorted(unknown)[:5]}")
    return result


def read_pairs(path):
    _, rows = read_rows(path, PAIR_COLUMNS, exact=True)
    return validate_pairs((row[PAIR_COLUMNS[0]], row[PAIR_COLUMNS[1]]) for row in rows)


@dataclass(frozen=True)
class ScoredPair:
    source1_entity_id: str
    target_entity_id: str
    match_probability: float


def validate_scores(scores, source_ids=None):
    scores = list(scores)
    pairs = validate_pairs(((s.source1_entity_id, s.target_entity_id) for s in scores), source_ids)
    result = []
    for (source, target), score in zip(pairs, scores):
        probability = float(score.match_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Match probabilities must be finite and between 0 and 1")
        result.append(ScoredPair(source, target, probability))
    return result


def read_scores(path):
    _, rows = read_rows(path, PAIR_COLUMNS + ("match_probability",), exact=True)
    return validate_scores(ScoredPair(row[PAIR_COLUMNS[0]], row[PAIR_COLUMNS[1]],
                                    float(row["match_probability"])) for row in rows)


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def write_matches(path, predictions):
    write_rows(path, MATCH_COLUMNS, ((source, ','.join(sorted(targets)))
                                    for source, targets in predictions.items()))


def write_scores(path, scores):
    scores = validate_scores(scores)
    write_rows(path, PAIR_COLUMNS + ("match_probability",),
               ((s.source1_entity_id, s.target_entity_id, repr(s.match_probability)) for s in scores))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
