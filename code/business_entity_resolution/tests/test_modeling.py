import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.io_utils import PAIR_COLUMNS, read_matches, read_pairs, read_source_ids, write_matches, write_rows

HAS_ML = all(importlib.util.find_spec(name) is not None for name in ("numpy", "sklearn", "pandas"))


@unittest.skipUnless(HAS_ML, "Install requirements.txt to run model integration tests")
class ModelingTests(unittest.TestCase):
    def test_connected_groups_and_splits(self):
        from src.model import make_splits, truth_groups
        truth = {"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": set(), "e": set()}
        groups = truth_groups(truth)
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["d"], groups["e"])
        truth.update({f"s{i}": {f"t{i}"} for i in range(30)})
        assignments, limitation = make_splits(truth)
        self.assertIsNone(limitation)
        self.assertEqual(assignments, make_splits(dict(reversed(list(truth.items()))))[0])
        for left in truth:
            for right in truth:
                if truth[left] & truth[right]:
                    self.assertEqual(assignments[left], assignments[right])
        partitions = {(row["partition"], row["fold"]) for row in assignments.values()}
        self.assertEqual(partitions, {("holdout", None), ("development", 0),
                                      ("development", 1), ("development", 2)})

    def test_feature_contract(self):
        import numpy as np
        from src.model import read_features
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.tsv"
            write_rows(path, list(PAIR_COLUMNS) + ["good"], [("001", "002", "")])
            table = read_features(path, ["good"])
            self.assertEqual(table.pairs, [("001", "002")])
            self.assertTrue(np.isnan(table.values[0, 0]))
            for names in (["label"], ["source1_entity_id"], ["good", "good"], ["missing"], []):
                with self.assertRaises(ValueError):
                    read_features(path, names)
            for bad in ("infinity", "text"):
                write_rows(path, list(PAIR_COLUMNS) + ["good"], [("001", "002", bad)])
                with self.assertRaises(ValueError):
                    read_features(path, ["good"])

    def test_insufficient_data_retains_baselines(self):
        import numpy as np
        from src.model import FeatureTable, train
        truth = {"s": {"t"}, "singleton": set()}
        table = FeatureTable([("s", "t")], ["similarity"], np.array([[1.0]]))
        with tempfile.TemporaryDirectory() as directory:
            report = train(list(truth), truth, table.pairs, table, directory, "synthetic")
            self.assertEqual(report["status"], "insufficient_data")
            self.assertTrue((Path(directory) / "all_empty.tsv").exists())
            self.assertFalse((Path(directory) / "model.pkl").exists())
            with self.assertRaisesRegex(ValueError, "protocol"):
                train(list(truth), truth, table.pairs, table, directory, None)

    def test_one_class_folds_retain_baselines(self):
        import numpy as np
        from src.model import FeatureTable, train
        truth = {f"s{i}": {f"t{i}"} for i in range(20)}
        pairs = [(source, next(iter(targets))) for source, targets in truth.items()]
        table = FeatureTable(pairs, ["similarity"], np.ones((20, 1)))
        with tempfile.TemporaryDirectory() as directory:
            report = train(list(truth), truth, pairs, table, directory, "synthetic")
            self.assertEqual(report["status"], "insufficient_data")
            self.assertIn("both classes", report["limitation"])

    def test_synthetic_training_cli_and_reload(self):
        import numpy as np
        from src.examples.synthetic_fixture import FEATURES, create_fixture
        from src import aggregate, evaluate
        from src.model import FeatureTable, load_model, predict, read_features, score_table
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, experiment = root / "data", root / "experiment"
            create_fixture(data)
            command = [sys.executable, "-m", "src", "train", "--source1", str(data / "source1.tsv"),
                       "--truth", str(data / "truth.tsv"), "--candidates", str(data / "candidates.tsv"),
                       "--features", str(data / "features.tsv"), "--feature-columns", *FEATURES,
                       "--ml2-baseline", str(data / "ml2_baseline.tsv"), "--output-dir", str(experiment), "--synthetic"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((experiment / "metrics.json").read_text())
            self.assertEqual(report["status"], "trained")
            self.assertFalse(report["holdout_used_for_selection"])
            self.assertEqual(len(report["trials"]), 4)
            self.assertEqual(report["excluded_unlabeled_entities"], 1)
            self.assertEqual(report["excluded_unlabeled_pairs"], 1)
            self.assertEqual(len(report["input_sha256"]["truth"]), 64)
            self.assertGreater(report["holdout_classifier"]["macro_f0_5"],
                               report["holdout_baselines"]["all_empty"]["macro_f0_5"])
            self.assertLessEqual(report["holdout_classifier"]["macro_f0_5"],
                                 report["holdout_baselines"]["candidate_oracle"]["macro_f0_5"])
            artifact = load_model(experiment / "model.pkl")
            source_ids = read_source_ids(data / "source1.tsv")
            table = read_features(data / "features.tsv", FEATURES)
            expected, scores = predict(artifact, source_ids, table)
            reloaded, reloaded_scores = predict(load_model(experiment / "model.pkl"), source_ids, table)
            self.assertEqual(expected, reloaded)
            self.assertEqual(scores, reloaded_scores)
            self.assertEqual(list(expected), source_ids)
            import pandas as pd
            from src.model import predict_matches
            frame = pd.DataFrame(table.values, columns=table.names)
            frame.insert(0, "target_entity_id", [target for _, target in table.pairs])
            frame.insert(0, "source1_entity_id", [source for source, _ in table.pairs])
            matches = predict_matches(frame, model_path=experiment / "model.pkl")
            self.assertEqual(set(matches.itertuples(index=False, name=None)),
                             {(source, target) for source, targets in expected.items() for target in targets})
            bad_table = FeatureTable(table.pairs, list(reversed(table.names)), table.values[:, ::-1])
            with self.assertRaisesRegex(ValueError, "schema"):
                predict(artifact, source_ids, bad_table)
            empty_table = FeatureTable([], FEATURES, np.empty((0, 2)))
            self.assertTrue(all(not targets for targets in predict(artifact, source_ids, empty_table)[0].values()))
            for output_name in ("predictions.tsv", "predictions_again.tsv"):
                inference = subprocess.run([sys.executable, "-m", "src", "predict", "--model", str(experiment / "model.pkl"),
                                            "--source1", str(data / "source1.tsv"), "--features", str(data / "features.tsv"),
                                            "--output", str(root / output_name)], capture_output=True, text=True)
                self.assertEqual(inference.returncode, 0, inference.stderr)
                self.assertEqual(read_matches(root / output_name), expected)
            self.assertEqual((root / "predictions.tsv").read_bytes(), (root / "predictions_again.tsv").read_bytes())


if __name__ == "__main__":
    unittest.main()
