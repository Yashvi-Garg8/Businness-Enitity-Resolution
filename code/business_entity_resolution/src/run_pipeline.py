"""Batched inference with a global target index and atomically published outputs."""
import argparse
import csv
import math
import os
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.blocking import BlockingConfig, SOURCE_COLUMNS, TargetIndex
from src.evaluate import IncrementalEvaluator
from src.features import FEATURE_COLS, FEATURE_VERSION, _compute_prepared_pairwise_features
from src.io_utils import PAIR_COLUMNS, MATCH_COLUMNS, write_json
from src.model import FeatureTable, load_model, predict, predict_matches
from src.normalize import NORMALIZED_FIELDS, normalize_dataset
from src.streaming import (ValidationStore, batches, check_header, source_rows,
                           validate_streamed_outputs)

PREPARED_COLUMNS = ["entity_id"] + sorted(NORMALIZED_FIELDS)
CATEGORIES = [name + '_source' for name in
              ('affected', 'recovered', 'unresolved_oversized', 'zero_candidate', 'zero_key', 'missing_country')]
CATEGORIES.append('missing_country_target')


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except (ImportError, AttributeError):
        return None


def count_distribution(histogram):
    total = sum(histogram.values())
    if not total:
        return dict(min=0, median=0, p95=0, max=0, mean=0)
    ordered = sorted(histogram.items())

    def value_at(rank):
        accumulated = 0
        for value, count in ordered:
            accumulated += count
            if rank < accumulated:
                return value

    def quantile(q):
        position = (total - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return value_at(lower) + (value_at(upper) - value_at(lower)) * (position - lower)

    return dict(min=ordered[0][0], median=quantile(.5), p95=quantile(.95),
                max=ordered[-1][0], mean=sum(value * count for value, count in ordered) / total)


def _writer(stack, path, columns):
    handle = stack.enter_context(open(path, 'w', encoding='utf-8', newline=''))
    writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
    writer.writerow(columns)
    return writer


def run(s1_path, s2_path, s3_path, out_dir, model_path=None, *, blocking_config=None,
        truth_path=None, source_batch_size=1000, pair_batch_size=50000):
    for size in (source_batch_size, pair_batch_size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('Batch sizes must be positive integers')
    paths = [Path(value).resolve() for value in (s1_path, s2_path, s3_path)]
    if len(set(paths)) != 3:
        raise ValueError('Sources 1, 2 and 3 must be distinct files')
    truth_path = Path(truth_path).resolve() if truth_path else None
    inputs = paths + ([truth_path] if truth_path else [])
    for path in paths:
        check_header(path, SOURCE_COLUMNS)
    if truth_path:
        check_header(truth_path, MATCH_COLUMNS, exact=True)
    output = Path(out_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output directory must be empty; choose a fresh run directory')
    artifact = load_model(model_path) if model_path else None
    if artifact is not None:
        if artifact.get('feature_version') != FEATURE_VERSION or set(artifact['feature_schema']) != set(FEATURE_COLS):
            raise ValueError('Model feature version/schema differs from the pipeline; retrain with current features')
    snapshots = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in inputs}
    started = time.perf_counter()
    timings = {}
    config = blocking_config or BlockingConfig()
    index = TargetIndex(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pipeline-', dir=output.parent) as directory:
        scratch = Path(directory)
        pending = scratch / 'results'
        pending.mkdir()
        store = ValidationStore(scratch / 'validation.sqlite')
        try:
            stage = time.perf_counter()
            print('[1/5] Validating Source 1 IDs...')
            for batch in batches(source_rows(paths[0], SOURCE_COLUMNS, 'S1-'), source_batch_size):
                for row in batch:
                    store.add_source(row['entity_id'])
                store.db.commit()
            if not store.source_count:
                raise ValueError('Source 1 universe must not be empty')
            timings['source_validation_seconds'] = time.perf_counter() - stage

            stage = time.perf_counter()
            print('[2/5] Normalizing targets and building the global blocking index...')
            for path, prefix in zip(paths[1:], ('S2-', 'S3-')):
                for batch in batches(source_rows(path, SOURCE_COLUMNS, prefix), source_batch_size):
                    normalized = normalize_dataset(pd.DataFrame(batch))[PREPARED_COLUMNS]
                    index.add(normalized.itertuples(index=False))
                    for row in normalized.itertuples(index=False):
                        if not row.block_country:
                            store.diagnostic('missing_country_target', row.entity_id)
                    store.db.commit()
            index.finalize()
            timings['target_preparation_index_seconds'] = time.perf_counter() - stage
            index_peak = peak_rss_bytes()

            stage = time.perf_counter()
            if truth_path:
                store.load_truth(truth_path, index.rows)
            timings['truth_validation_seconds'] = time.perf_counter() - stage
            metrics = IncrementalEvaluator()
            histogram = Counter()
            pair_count = 0
            processed = 0
            print('[3/5] Generating candidates and scoring bounded pair batches...')
            stage = time.perf_counter()
            with ExitStack() as stack:
                matches_writer = _writer(stack, pending/'matching_results.tsv', MATCH_COLUMNS)
                candidate_writer = _writer(stack, pending/'candidate_pairs.tsv', ('source1_entity_id', 'candidate_entity_ids'))
                pair_writer = _writer(stack, pending/'candidate_pairs_long.tsv', PAIR_COLUMNS)
                error_writer = (_writer(stack, pending/'errors.tsv',
                                ('source1_entity_id', 'ground_truth', 'predicted', 'false_positives', 'false_negatives'))
                                if truth_path else None)
                for batch in batches(source_rows(paths[0], SOURCE_COLUMNS, 'S1-'), source_batch_size):
                    prepared = normalize_dataset(pd.DataFrame(batch))[PREPARED_COLUMNS]
                    accepted = {entity: set() for entity in prepared['entity_id']}
                    entity_candidates = []
                    pair_buffer = []

                    def flush():
                        if not pair_buffer:
                            return
                        targets = list(dict.fromkeys(target for _, target in pair_buffer))
                        target_frame = pd.DataFrame([index.rows[target] for target in targets], columns=PREPARED_COLUMNS)
                        pair_frame = pd.DataFrame(pair_buffer, columns=PAIR_COLUMNS)
                        features = _compute_prepared_pairwise_features(pair_frame, prepared, target_frame)
                        if list(features[list(PAIR_COLUMNS)].itertuples(index=False, name=None)) != pair_buffer:
                            raise ValueError('Features must cover each candidate exactly once in order')
                        if artifact is None:
                            for source, target in predict_matches(features).itertuples(index=False, name=None):
                                accepted[source].add(target)
                        else:
                            names = artifact['feature_schema']
                            table = FeatureTable(list(pair_buffer), names, features[names].to_numpy(dtype=float))
                            predictions, _ = predict(artifact, list(accepted), table)
                            for source, targets_for_source in predictions.items():
                                accepted[source].update(targets_for_source)
                        pair_writer.writerows(pair_buffer)
                        pair_buffer.clear()

                    for row in prepared.itertuples(index=False):
                        candidates, flags = index.candidates(row)
                        entity_candidates.append((row, candidates))
                        histogram[len(candidates)] += 1
                        pair_count += len(candidates)
                        for category, enabled in flags.items():
                            if enabled:
                                store.diagnostic(category + '_source', row.entity_id)
                        candidate_writer.writerow((row.entity_id, ','.join(candidates)))
                        for target in candidates:
                            pair_buffer.append((row.entity_id, target))
                            if len(pair_buffer) == pair_batch_size:
                                flush()
                    flush()
                    for row, candidates in entity_candidates:
                        predicted = accepted[row.entity_id]
                        matches_writer.writerow((row.entity_id, ','.join(sorted(predicted))))
                        actual = store.actual(row.entity_id) if truth_path else None
                        if actual is not None:
                            metrics.add(actual, predicted, set(candidates), index.retained_before(row, actual))
                            if predicted != actual:
                                error_writer.writerow((row.entity_id, ','.join(sorted(actual)), ','.join(sorted(predicted)),
                                                       ','.join(sorted(predicted - actual)), ','.join(sorted(actual - predicted))))
                    processed += len(batch)
                    store.db.commit()
                    print(f'      Processed {processed}/{store.source_count} sources; {pair_count} candidate pairs')
            timings['candidate_scoring_seconds'] = time.perf_counter() - stage
            for path in inputs:
                if snapshots[str(path)] != (path.stat().st_size, path.stat().st_mtime_ns):
                    raise ValueError(f'Input changed during run: {path}')
            print('[4/5] Validating streamed outputs...')
            stage = time.perf_counter()
            validate_streamed_outputs(pending, store, index.rows)
            timings['output_validation_seconds'] = time.perf_counter() - stage
            possible_pairs = store.source_count * len(index.rows)
            blocking_report = {
                'config': asdict(config), 'source1_records': store.source_count, 'target_records': len(index.rows),
                'candidate_pairs': pair_count, 'possible_pairs': possible_pairs,
                'reduction_ratio': 1 - pair_count / possible_pairs if possible_pairs else None,
                **index.statistics(), 'candidate_count_distribution': count_distribution(histogram)}
            if truth_path:
                evaluation = metrics.report()
                evaluation['excluded_unlabeled_entities'] = store.source_count - store.truth_count
                write_json(pending/'evaluation.json', evaluation)
                actual_pairs = metrics.tp + metrics.fn
                retained_after = actual_pairs - metrics.blocking_misses
                blocking_report['evaluation'] = {
                    'labeled_source_count': metrics.entities, 'unlabeled_source_count': store.source_count - metrics.entities,
                    'true_pairs': actual_pairs, 'retained_before_filtering': metrics.retained_before,
                    'retained_after_filtering': retained_after,
                    'true_pairs_lost_to_bucket_filtering': metrics.retained_before - retained_after,
                    'pair_recall_before_filtering': metrics.retained_before / actual_pairs if actual_pairs else None,
                    'pair_recall_after_filtering': retained_after / actual_pairs if actual_pairs else None,
                    'candidate_oracle_before_filtering': metrics.before_score / metrics.entities,
                    'candidate_oracle_after_filtering': metrics.oracle_score / metrics.entities,
                    'singleton_fraction': metrics.singletons / metrics.entities}
            store.write_report(pending/'blocking_report.json', blocking_report, CATEGORIES)
            timings['total_seconds'] = time.perf_counter() - started
            manifest = {'status': 'complete', 'feature_version': FEATURE_VERSION,
                        'mode': 'model' if artifact is not None else 'baseline',
                        'threshold': artifact['threshold'] if artifact is not None else .86,
                        'source_batch_size': source_batch_size, 'pair_batch_size': pair_batch_size,
                        'blocking_config': asdict(config), 'timings': timings,
                        'peak_rss_bytes': peak_rss_bytes(), 'target_index_peak_rss_bytes': index_peak,
                        'source1_records': store.source_count, 'target_records': len(index.rows),
                        'candidate_pairs': pair_count, 'labeled_entities': store.truth_count,
                        'inputs': {path: {'bytes': stat[0], 'mtime_ns': stat[1]} for path, stat in snapshots.items()}}
            write_json(pending/'run_manifest.json', manifest)
        finally:
            store.close()
        print('[5/5] Publishing validated outputs...')
        if output.exists():
            output.rmdir()  # Fails safely if another process populated the directory.
        os.rename(pending, output)
    print(f'[PASS] Pipeline complete: {output}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('s1', 's2', 's3'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--out', default='output')
    parser.add_argument('--model', help='Optional compatible model artifact; defaults to baseline')
    parser.add_argument('--bucket-limit', type=int, default=500)
    parser.add_argument('--bucket-policy', choices=('refine', 'drop', 'uncapped'), default='refine')
    parser.add_argument('--truth', help='Optional truth for labeled-entity diagnostics only')
    parser.add_argument('--source-batch-size', type=int, default=1000)
    parser.add_argument('--pair-batch-size', type=int, default=50000)
    args = parser.parse_args(argv)
    try:
        run(args.s1, args.s2, args.s3, args.out, args.model,
            blocking_config=BlockingConfig(args.bucket_limit, args.bucket_policy), truth_path=args.truth,
            source_batch_size=args.source_batch_size, pair_batch_size=args.pair_batch_size)
    except (ValueError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
