"""Strict streaming TSV readers and disk-backed validation (standard library only)."""
import csv
import itertools
import json
import sqlite3
import sys
from pathlib import Path

from .io_utils import identifier, parse_matches


def iter_rows(path, required, exact=False):
    # Grouped candidate cells may exceed csv's default 128 KiB field limit.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", strict=True)
        headers = reader.fieldnames
        if (not headers or len(headers) != len(set(headers)) or any(not col for col in headers)
                or not set(required).issubset(headers) or (exact and set(headers) != set(required))):
            raise ValueError(f"{path}: expected unique columns {list(required)}, got {headers}")
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}:{reader.line_num}: wrong number of TSV fields")
                yield row
        except csv.Error as exc:
            raise ValueError(f"{path}:{reader.line_num}: malformed TSV: {exc}") from exc


def check_header(path, required, exact=False):
    # Starting the generator validates the header even for a header-only file.
    iterator = iter_rows(path, required, exact)
    try:
        next(iterator, None)
    finally:
        iterator.close()


def batches(rows, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Batch size must be a positive integer")
    rows = iter(rows)
    while True:
        batch = list(itertools.islice(rows, size))
        if not batch:
            return
        yield batch


def source_rows(path, columns, prefix):
    for row in iter_rows(path, columns):
        entity = identifier(row["entity_id"])
        if not entity.startswith(prefix) or len(entity) == len(prefix):
            raise ValueError(f"{path}: expected ID prefix {prefix}, got {entity!r}")
        row["entity_id"] = entity
        yield {column: row[column] for column in columns}


class ValidationStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.executescript("""
            CREATE TABLE sources (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE);
            CREATE TABLE truth (id TEXT PRIMARY KEY, matches TEXT NOT NULL);
            CREATE TABLE diagnostics (category TEXT, id TEXT);
            CREATE INDEX diagnostic_categories ON diagnostics(category);
        """)
        self.source_count = 0
        self.truth_count = 0

    def close(self):
        self.db.close()

    def add_source(self, entity):
        try:
            self.db.execute("INSERT INTO sources VALUES (?, ?)", (entity, self.source_count))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate Source 1 ID: {entity}") from exc
        self.source_count += 1

    def load_truth(self, path, target_ids):
        for row in iter_rows(path, ("source1_entity_id", "matched_entity_ids"), exact=True):
            entity = identifier(row["source1_entity_id"])
            actual = parse_matches(row["matched_entity_ids"])
            if not self.db.execute("SELECT 1 FROM sources WHERE id=?", (entity,)).fetchone():
                raise ValueError(f"Truth contains unknown Source 1 ID: {entity}")
            if any(target not in target_ids for target in actual):
                raise ValueError(f"Truth for {entity} contains targets absent from Sources 2/3")
            try:
                self.db.execute("INSERT INTO truth VALUES (?, ?)", (entity, ','.join(sorted(actual))))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate truth Source 1 ID: {entity}") from exc
            self.truth_count += 1
        if not self.truth_count:
            raise ValueError("Ground truth has no labeled Source 1 rows")
        self.db.commit()

    def actual(self, entity):
        row = self.db.execute("SELECT matches FROM truth WHERE id=?", (entity,)).fetchone()
        return parse_matches(row[0]) if row is not None else None

    def diagnostic(self, category, entity):
        self.db.execute("INSERT INTO diagnostics VALUES (?, ?)", (category, entity))

    def write_report(self, path, report, categories):
        # Preserve the existing ID-array schema without keeping those arrays in RAM.
        with open(path, "w", encoding="utf-8") as stream:
            stream.write('{')
            first = True
            for key, value in report.items():
                if not first:
                    stream.write(',')
                first = False
                stream.write(json.dumps(key) + ':' + json.dumps(value, allow_nan=False))
            for category in categories:
                count = self.db.execute("SELECT COUNT(*) FROM diagnostics WHERE category=?", (category,)).fetchone()[0]
                stream.write(',' + json.dumps(category + '_count') + ':' + str(count))
                stream.write(',' + json.dumps(category + '_ids') + ':[')
                separator = ''
                for (entity,) in self.db.execute("SELECT id FROM diagnostics WHERE category=? ORDER BY rowid", (category,)):
                    stream.write(separator + json.dumps(entity))
                    separator = ','
                stream.write(']')
            stream.write('}\n')


def validate_streamed_outputs(directory, store, targets):
    """Check order, coverage, uniqueness, IDs and grouped/long-form equivalence."""
    directory = Path(directory)
    matches = iter_rows(directory / 'matching_results.tsv', ('source1_entity_id', 'matched_entity_ids'), exact=True)
    candidates = iter_rows(directory / 'candidate_pairs.tsv', ('source1_entity_id', 'candidate_entity_ids'), exact=True)
    pairs = iter_rows(directory / 'candidate_pairs_long.tsv', ('source1_entity_id', 'target_entity_id'), exact=True)
    try:
        for (entity,) in store.db.execute('SELECT id FROM sources ORDER BY ordinal'):
            match = next(matches, None)
            candidate = next(candidates, None)
            if match is None or candidate is None or match['source1_entity_id'] != entity or candidate['source1_entity_id'] != entity:
                raise ValueError('Output order or Source 1 coverage mismatch')
            actual_candidates = parse_matches(candidate['candidate_entity_ids'])
            predicted = parse_matches(match['matched_entity_ids'])
            for value, parsed in ((candidate['candidate_entity_ids'], actual_candidates),
                                  (match['matched_entity_ids'], predicted)):
                if value != ','.join(sorted(parsed)):
                    raise ValueError(f'Output matches must be sorted and unique: {entity}')
            if predicted - actual_candidates or any(target not in targets for target in actual_candidates):
                raise ValueError(f'Invalid output candidate/match for {entity}')
            for target in sorted(actual_candidates):
                pair = next(pairs, None)
                if pair != {'source1_entity_id': entity, 'target_entity_id': target}:
                    raise ValueError(f'Grouped and long-form candidates disagree for {entity}')
        if next(matches, None) is not None or next(candidates, None) is not None or next(pairs, None) is not None:
            raise ValueError('Unexpected extra output rows')
    finally:
        for iterator in (matches, candidates, pairs):
            iterator.close()


def evaluate_files(predictions_path, truth_path, error_path=None):
    """Exact-coverage legacy evaluator with predictions indexed on temporary disk."""
    import shutil
    import tempfile
    from .evaluate import entity_score

    with tempfile.TemporaryDirectory(prefix='entity-evaluation-') as directory:
        db = sqlite3.connect(Path(directory)/'predictions.sqlite')
        db.execute('PRAGMA cache_size=-8192')
        db.execute('CREATE TABLE predictions (id TEXT PRIMARY KEY, matches TEXT, seen INTEGER DEFAULT 0)')
        try:
            predictions_count = 0
            for row in iter_rows(predictions_path, ('source1_entity_id', 'matched_entity_ids'), exact=True):
                entity = identifier(row['source1_entity_id'])
                matched = ','.join(sorted(parse_matches(row['matched_entity_ids'])))
                try:
                    db.execute('INSERT INTO predictions(id,matches) VALUES (?,?)', (entity, matched))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'Duplicate Source 1 ID: {entity}') from exc
                predictions_count += 1
                if predictions_count % 10000 == 0:
                    db.commit()
            db.commit()
            entities, total_score = 0, 0.0
            temporary_errors = Path(directory)/'errors.tsv'
            with temporary_errors.open('w', encoding='utf-8', newline='') as handle:
                errors = csv.writer(handle, delimiter='\t', lineterminator='\n')
                errors.writerow(('source1_entity_id', 'ground_truth', 'predicted', 'false_positives', 'false_negatives'))
                for row in iter_rows(truth_path, ('source1_entity_id', 'matched_entity_ids'), exact=True):
                    entity = identifier(row['source1_entity_id'])
                    actual = parse_matches(row['matched_entity_ids'])
                    predicted_row = db.execute('SELECT matches,seen FROM predictions WHERE id=?', (entity,)).fetchone()
                    if predicted_row is None:
                        raise ValueError(f'Prediction coverage mismatch: missing {entity}')
                    if predicted_row[1]:
                        raise ValueError(f'Duplicate Source 1 ID in truth: {entity}')
                    db.execute('UPDATE predictions SET seen=1 WHERE id=?', (entity,))
                    predicted = parse_matches(predicted_row[0])
                    total_score += entity_score(len(actual & predicted), len(predicted - actual), len(actual - predicted))
                    entities += 1
                    if error_path and actual != predicted:
                        errors.writerow((entity, ','.join(sorted(actual)), ','.join(sorted(predicted)),
                                         ','.join(sorted(predicted - actual)), ','.join(sorted(actual - predicted))))
                    if entities % 10000 == 0:
                        db.commit()
            if not entities:
                raise ValueError('Source 1 universe must not be empty')
            if entities != predictions_count:
                raise ValueError('Prediction coverage mismatch: extra prediction rows')
            if error_path:
                Path(error_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(temporary_errors, error_path)
            return total_score / entities, entities
        finally:
            db.close()
