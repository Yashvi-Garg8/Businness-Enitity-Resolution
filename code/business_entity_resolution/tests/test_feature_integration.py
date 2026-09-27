"""Regression coverage for normalization, ML-2 features, and ML-3 handoff."""
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.blocking import SOURCE_COLUMNS
from src.features import FEATURE_COLS, FEATURE_VERSION, compute_pairwise_features
from src.io_utils import PAIR_COLUMNS, write_matches, write_rows
from src.model import dependency_versions, predict_matches


def records(rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def features(left, right):
    pairs = pd.DataFrame([("S1-1", "S2-1")], columns=PAIR_COLUMNS)
    return compute_pairwise_features(pairs, records([("S1-1", *left)]), records([("S2-1", *right)]))


class FeatureIntegrationTests(unittest.TestCase):
    def test_conservative_fields_and_canonical_country(self):
        result = features(("Ｃafé & Société", "12 Main Rd", "U.S.A."),
                          ("Café and Société", "12 Main Rd", "United States")).iloc[0]
        self.assertEqual(result.name_ratio, 1)
        self.assertEqual(result.country_match, 1)
        self.assertEqual(result.street_number_match, 1)

    def test_missing_text_is_not_agreement(self):
        result = features(("", None, "India"), ("", None, "India")).iloc[0]
        for name in FEATURE_COLS:
            self.assertEqual(result[name], 1 if name == "country_match" else 0, name)

    def test_postal_and_ambiguous_numbers_are_not_street_agreement(self):
        for address in ("110001", "12-14 Main Road", "Near Main Road 12"):
            with self.subTest(address=address):
                result = features(("Acme", address, "India"), ("Acme", address, "India"))
                self.assertEqual(result.iloc[0].street_number_match, 0)

    def test_pair_validation_and_empty_schema(self):
        left = records([("S1-1", "Acme", "", "India")])
        right = records([("S2-1", "Acme", "", "India")])
        empty = compute_pairwise_features(pd.DataFrame(columns=PAIR_COLUMNS), left, right)
        self.assertEqual(list(empty.columns), list(PAIR_COLUMNS) + FEATURE_COLS)
        for rows in ([('S1-1', 'S2-1')] * 2, [('S1-missing', 'S2-1')], [('S1-1', 'S2-missing')]):
            with self.assertRaises(ValueError):
                compute_pairwise_features(pd.DataFrame(rows, columns=PAIR_COLUMNS), left, right)

    def test_model_feature_version_and_schema(self):
        frame = features(("Acme", "12 Main Rd", "India"), ("Acme", "12 Main Rd", "India"))
        classifier = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((2, len(FEATURE_COLS))), [0, 1])
        artifact = dict(artifact_version=1, model=classifier, feature_schema=FEATURE_COLS,
                        feature_version=FEATURE_VERSION, threshold=.5, versions=dependency_versions())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            path.write_bytes(pickle.dumps(artifact))
            self.assertEqual(len(predict_matches(frame, model_path=path, feature_version=FEATURE_VERSION)), 1)
            artifact['feature_version'] = None
            path.write_bytes(pickle.dumps(artifact))
            with self.assertRaisesRegex(ValueError, 'retrain'):
                predict_matches(frame, model_path=path, feature_version=FEATURE_VERSION)
            artifact['feature_version'] = FEATURE_VERSION
            artifact['feature_schema'] = ['token_sort', 'address_ratio']
            path.write_bytes(pickle.dumps(artifact))
            with self.assertRaisesRegex(ValueError, 'schema'):
                predict_matches(frame, model_path=path, feature_version=FEATURE_VERSION)

    def test_tuning_cli_scopes_truth_and_retains_baselines_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root/'s1.tsv', SOURCE_COLUMNS,
                       [('S1-1', 'Acme', '12 Main Rd', 'India'),
                        ('S1-unlabeled', 'Other', '', 'India')])
            write_rows(root/'s2.tsv', SOURCE_COLUMNS, [('S2-1', 'Acme', '12 Main Rd', 'India')])
            write_rows(root/'s3.tsv', SOURCE_COLUMNS, [])
            write_matches(root/'truth.tsv', {'S1-1': {'S2-1'}})
            args = ['--s1', str(root/'s1.tsv'), '--s2', str(root/'s2.tsv'), '--s3', str(root/'s3.tsv'),
                    '--ground-truth', str(root/'truth.tsv'), '--output-dir', str(root/'output'), '--synthetic']
            result = subprocess.run([sys.executable, 'src/tune_threshold.py', *args],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            report = json.loads((root/'output/metrics.json').read_text())
            self.assertEqual(report['status'], 'insufficient_data')
            self.assertEqual(report['excluded_unlabeled_entities'], 1)
            self.assertEqual(report['feature_version'], FEATURE_VERSION)
            self.assertEqual(report['feature_schema'], FEATURE_COLS)
            self.assertFalse((root/'output/model.pkl').exists())
            self.assertTrue((root/'output/features.tsv').is_file())
            result = subprocess.run([sys.executable, '-m', 'src.tune_threshold', '--help'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
