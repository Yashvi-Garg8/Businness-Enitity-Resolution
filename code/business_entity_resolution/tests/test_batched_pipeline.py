"""Small-fixture equivalence and failure-safety checks for streaming inference."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.blocking import (BlockingConfig, SOURCE_COLUMNS, TargetIndex,
                          generate_candidate_pairs_with_report)
from src.evaluate import evaluate, IncrementalEvaluator
from src.features import compute_pairwise_features
from src.io_utils import read_matches, read_pairs, write_matches, write_rows
from src.model import predict_matches
from src.normalize import normalize_dataset
from src.run_pipeline import PREPARED_COLUMNS, run
from src.streaming import evaluate_files


class BatchedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = [
            [('S1-1', 'Acme', '12 Main Road', 'U.S.A.'),
             ('S1-2', 'Unique Unmatched', '', 'India'),
             ('S1-3', 'Acme', '12 Main Road', 'US'),
             ('S1-4', 'Empty Address', '', 'India')],
            [('S2-1', 'Acme', '12 Main Road', 'United States'),
             ('S2-2', 'Empty Address', '', 'India')],
            [('S3-1', 'Acme', '12 Main Road', 'US'),
             ('S3-2', 'Different', '99 Far Street', 'India')]]
        self.paths = [self.root/f's{i}.tsv' for i in (1, 2, 3)]
        self.truth = {'S1-1': {'S2-1', 'S3-1', 'S3-2'}, 'S1-2': set(), 'S1-4': {'S2-2'}}
        self.truth_path = self.root/'truth.tsv'
        self.write_inputs()

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        for path, rows in zip(self.paths, self.sources):
            write_rows(path, SOURCE_COLUMNS, rows)
        write_matches(self.truth_path, self.truth)

    def run_pipeline(self, name='output', **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(*self.paths, self.root/name, truth_path=self.truth_path, **kwargs)

    def test_batch_sizes_match_reference_and_preserve_metrics(self):
        s1 = normalize_dataset(pd.DataFrame(self.sources[0], columns=SOURCE_COLUMNS))
        targets = normalize_dataset(pd.DataFrame(self.sources[1] + self.sources[2], columns=SOURCE_COLUMNS))
        reference, blocking = generate_candidate_pairs_with_report(s1, targets, truth=self.truth)
        features = compute_pairwise_features(reference, s1, targets)
        expected = {entity: set() for entity in s1.entity_id}
        for source, target in predict_matches(features).itertuples(index=False, name=None):
            expected[source].add(target)
        reference_metrics = evaluate(self.truth, {key: expected[key] for key in self.truth},
                                     [pair for pair in reference.itertuples(index=False, name=None) if pair[0] in self.truth])
        previous = None
        for source_size, pair_size in ((1, 1), (2, 3), (10, 50)):
            name = f'output-{source_size}'
            manifest = self.run_pipeline(name, source_batch_size=source_size, pair_batch_size=pair_size)
            output = self.root/name
            self.assertEqual(read_matches(output/'matching_results.tsv'), expected)
            self.assertEqual(read_pairs(output/'candidate_pairs_long.tsv'), list(reference.itertuples(index=False, name=None)))
            self.assertEqual(json.loads((output/'blocking_report.json').read_text()), blocking)
            metrics = json.loads((output/'evaluation.json').read_text())
            for key, value in reference_metrics.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(metrics[key], value)
                else:
                    self.assertEqual(metrics[key], value, key)
            self.assertEqual(metrics['excluded_unlabeled_entities'], 1)
            self.assertEqual(manifest['status'], 'complete')
            payloads = [(output/name).read_bytes() for name in
                        ('candidate_pairs.tsv', 'candidate_pairs_long.tsv', 'matching_results.tsv', 'errors.tsv')]
            if previous is not None:
                self.assertEqual(payloads, previous)
            previous = payloads

    def test_normalized_once_and_feature_batches_bounded(self):
        from src import run_pipeline as pipeline
        original = pipeline._compute_prepared_pairwise_features
        sizes = []

        def score(pairs, *args, **kwargs):
            sizes.append(len(pairs))
            return original(pairs, *args, **kwargs)

        with patch('src.run_pipeline.normalize_dataset', wraps=normalize_dataset) as normalization, \
             patch('src.features.normalize_dataset', side_effect=AssertionError('Repeated normalization')), \
             patch('src.run_pipeline._compute_prepared_pairwise_features', side_effect=score):
            self.run_pipeline(source_batch_size=2, pair_batch_size=3)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 3)
        self.assertEqual(sum(len(call.args[0]) for call in normalization.call_args_list),
                         sum(map(len, self.sources)))

    def test_global_bucket_refinement_matches_reference(self):
        source = pd.DataFrame([('S1-1', 'Acme Holdings', '12 Main Rd', 'India')], columns=SOURCE_COLUMNS)
        targets = pd.DataFrame([(f'S2-{i:04d}', 'Acmex Services', f'{i} Main Road', 'India')
                                for i in range(1, 502)], columns=SOURCE_COLUMNS)
        normalized = normalize_dataset(targets)[PREPARED_COLUMNS]
        row = next(normalize_dataset(source)[PREPARED_COLUMNS].itertuples(index=False))
        for policy in ('refine', 'drop', 'uncapped'):
            config = BlockingConfig(500, policy)
            index = TargetIndex(config)
            for start in range(0, 501, 100):
                index.add(normalized.iloc[start:start+100].itertuples(index=False))
            index.finalize()
            actual, flags = index.candidates(row)
            expected, report = generate_candidate_pairs_with_report(source, targets, config=config)
            self.assertEqual(actual, expected.target_entity_id.tolist())
            self.assertEqual(index.statistics(), {key: report[key] for key in index.statistics()})
            self.assertEqual(index.retained_before(row, {'S2-0012'}), 1)
            for category, enabled in flags.items():
                self.assertEqual(enabled, 'S1-1' in report[category + '_source_ids'])

    def test_cross_chunk_duplicate_ids_and_truth_errors(self):
        for case in ('source', 'target', 'truth', 'unknown_source', 'unknown_target', 'prefix'):
            with self.subTest(case=case):
                self.write_inputs()
                if case in ('source', 'target'):
                    position = 0 if case == 'source' else 1
                    with self.paths[position].open('a') as handle:
                        handle.write('\t'.join(self.sources[position][0]) + '\n')
                elif case == 'truth':
                    with self.truth_path.open('a') as handle:
                        handle.write('S1-1\tS2-1\n')
                elif case == 'unknown_source':
                    write_matches(self.truth_path, {'S1-unknown': set()})
                elif case == 'unknown_target':
                    write_matches(self.truth_path, {'S1-1': {'S3-unknown'}})
                else:
                    with self.paths[1].open('a') as handle:
                        handle.write('S3-wrong\tAcme\t\tUS\n')
                with self.assertRaises(ValueError):
                    self.run_pipeline(source_batch_size=1)
                self.assertFalse((self.root/'output').exists())

    def test_malformed_rows_and_headers_leave_no_outputs(self):
        with self.paths[2].open('a') as handle:
            handle.write('S3-malformed\tMissing columns\n')
        with self.assertRaisesRegex(ValueError, 'fields'):
            self.run_pipeline(source_batch_size=1)
        self.assertFalse((self.root/'output').exists())
        self.paths[2].write_text('bad\theader\n')
        with self.assertRaisesRegex(ValueError, 'columns'):
            self.run_pipeline('new-parent/output')
        self.assertFalse((self.root/'new-parent').exists())

    def test_output_failure_is_atomic_and_existing_results_protected(self):
        with patch('src.run_pipeline.validate_streamed_outputs', side_effect=ValueError('Injected failure')):
            with self.assertRaisesRegex(ValueError, 'Injected'):
                self.run_pipeline()
        self.assertFalse((self.root/'output').exists())
        self.assertFalse(list(self.root.glob('.pipeline-*')))
        output = self.root/'output'
        output.mkdir()
        (output/'keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.run_pipeline()
        self.assertEqual((output/'keep.txt').read_text(), 'keep')

    def test_empty_targets_and_no_truth(self):
        for path in self.paths[1:]:
            write_rows(path, SOURCE_COLUMNS, [])
        with contextlib.redirect_stdout(io.StringIO()):
            run(*self.paths, self.root/'empty', source_batch_size=1, pair_batch_size=1)
        self.assertTrue(all(not matched for matched in read_matches(self.root/'empty/matching_results.tsv').values()))
        self.assertEqual(read_pairs(self.root/'empty/candidate_pairs_long.tsv'), [])
        self.assertFalse((self.root/'empty/evaluation.json').exists())

    def test_streamed_evaluator_reordering_duplicates_and_coverage(self):
        truth = self.root/'score-truth.tsv'
        predictions = self.root/'score-predictions.tsv'
        actual = {'S1-1': {'S2-1', 'S3-1'}, 'S1-2': set()}
        predicted = {'S1-2': set(), 'S1-1': {'S2-1'}}
        write_matches(truth, actual)
        write_matches(predictions, predicted)
        score, count = evaluate_files(predictions, truth, self.root/'errors.tsv')
        self.assertEqual(count, 2)
        self.assertEqual(score, evaluate(actual, predicted)['macro_f0_5'])
        predictions.write_text('source1_entity_id\tmatched_entity_ids\nS1-2\t\nS1-1\t S2-1, S2-1 \n')
        self.assertEqual(evaluate_files(predictions, truth)[0], score)
        with predictions.open('a') as handle:
            handle.write('S1-1\t\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluate_files(predictions, truth)
        write_matches(predictions, {'S1-1': set()})
        with self.assertRaisesRegex(ValueError, 'coverage'):
            evaluate_files(predictions, truth)
        write_matches(predictions, {**predicted, 'S1-extra': set()})
        with self.assertRaisesRegex(ValueError, 'coverage'):
            evaluate_files(predictions, truth)

    def test_model_is_loaded_once_and_uses_saved_feature_order(self):
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from src.features import FEATURE_COLS, FEATURE_VERSION
        model = DummyClassifier(strategy='constant', constant=1).fit(
            np.zeros((2, len(FEATURE_COLS))), [0, 1])
        artifact = {'model': model, 'feature_schema': list(reversed(FEATURE_COLS)),
                    'feature_version': FEATURE_VERSION, 'threshold': .5}
        with patch('src.run_pipeline.load_model', return_value=artifact) as loader:
            manifest = self.run_pipeline(model_path='fixture.pkl', source_batch_size=1, pair_batch_size=1)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(manifest['mode'], 'model')
        accepted = read_matches(self.root/'output/matching_results.tsv')
        expected = {row[0]: set() for row in self.sources[0]}
        for source, target in read_pairs(self.root/'output/candidate_pairs_long.tsv'):
            expected[source].add(target)
        self.assertEqual(accepted, expected)

    def test_stream_reader_allows_large_grouped_cells(self):
        from src.streaming import iter_rows
        path = self.root/'large-cell.tsv'
        text = ','.join(f'S2-{i:08d}' for i in range(15000))
        path.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\t' + text + '\n')
        rows = list(iter_rows(path, ('source1_entity_id', 'matched_entity_ids'), exact=True))
        self.assertEqual(rows[0]['matched_entity_ids'], text)


if __name__ == '__main__':
    unittest.main()
