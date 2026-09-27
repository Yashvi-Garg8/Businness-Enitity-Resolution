import itertools
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import aggregate, candidate_oracle, empty_predictions, evaluate, tune_threshold
from src.io_utils import (MATCH_COLUMNS, ScoredPair, read_matches, read_pairs, read_scores,
                    write_matches, write_rows)


class EvaluationTests(unittest.TestCase):
    def test_known_scores(self):
        cases = [(set(), set(), 1), (set(), {"b"}, 0), ({"b"}, set(), 0),
                 ({"b"}, {"b"}, 1), ({"b", "c"}, {"b"}, 5 / 6),
                 ({"b"}, {"b", "c"}, 5 / 9)]
        for actual, prediction, expected in cases:
            with self.subTest(actual=actual, prediction=prediction):
                self.assertAlmostEqual(evaluate({"s": actual}, {"s": prediction})["macro_f0_5"], expected)

    def test_macro_and_singletons(self):
        truth = {"001": {"a", "b"}, "002": set(), "003": {"c"}}
        predictions = {"003": set(), "002": set(), "001": {"a"}}
        self.assertAlmostEqual(evaluate(truth, predictions)["macro_f0_5"], (5 / 6 + 1) / 3)
        self.assertAlmostEqual(evaluate(truth, empty_predictions(truth))["macro_f0_5"], 1 / 3)

    def test_coverage_is_strict(self):
        for predictions in ({"other": set()}, {"s": set(), "other": set()}, {}):
            with self.assertRaises(ValueError):
                evaluate({"s": set()}, predictions)

    def test_oracle_bounds_every_candidate_subset(self):
        truth = {"s1": {"a", "b"}, "s2": set(), "s3": {"d"}}
        pairs = [("s1", "a"), ("s1", "wrong"), ("s2", "wrong")]
        oracle = evaluate(truth, candidate_oracle(truth, pairs))["macro_f0_5"]
        for chosen in itertools.product([False, True], repeat=len(pairs)):
            predictions = empty_predictions(truth)
            for (source, target), accepted in zip(pairs, chosen):
                if accepted:
                    predictions[source].add(target)
            report = evaluate(truth, predictions, pairs)
            self.assertLessEqual(report["macro_f0_5"], oracle)
            self.assertEqual(report["blocking_misses"], 2)
            self.assertEqual(report["false_negatives"], report["blocking_misses"] + report["classifier_misses"])
        self.assertEqual(candidate_oracle(truth, []), empty_predictions(truth))
        with self.assertRaises(ValueError):
            evaluate(truth, truth, pairs)

    def test_tsv_whitespace_duplicates_order_and_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.tsv"
            write_rows(path, MATCH_COLUMNS, [("001", " S3_b, S2_a,S2_a "), ("002", "")])
            predictions = read_matches(path)
            self.assertEqual(predictions, {"001": {"S2_a", "S3_b"}, "002": set()})
            write_matches(path, predictions)
            self.assertEqual(path.read_text(), "source1_entity_id\tmatched_entity_ids\n001\tS2_a,S3_b\n002\t\n")

    def test_bad_tsvs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tsv"
            for content in (
                "source1_entity_id\tmatched_entity_ids\ns\ta\ns\tb\n",
                "source1_entity_id\tmatched_entity_ids\ns\ta\textra\n",
                "source1_entity_id\tmatched_entity_ids\ns\n",
                "source1_entity_id\tmatched_entity_ids\ns\ta,,b\n",
                "source1_entity_id\tmatched_entity_ids\tmatched_entity_ids\n",
                "source1_entity_id\twrong\n",
            ):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_matches(path)
            path.write_text("source1_entity_id\ttarget_entity_id\ns\ta\ns\ta\n")
            with self.assertRaises(ValueError):
                read_pairs(path)

    def test_aggregate_empty_threshold_and_unknown_ids(self):
        scores = [ScoredPair("s", "S3_b", 0.9), ScoredPair("s", "S2_a", 0.9)]
        self.assertEqual(list(aggregate(["z", "s"], scores, 0.9)), ["z", "s"])
        self.assertEqual(aggregate(["z", "s"], scores, 0.9)["s"], {"S2_a", "S3_b"})
        self.assertEqual(aggregate(["s"], scores, None), {"s": set()})
        self.assertEqual(aggregate(["s"], [], 0.5), {"s": set()})
        for bad in (float("nan"), float("inf"), -1, 1.1):
            with self.assertRaises(ValueError):
                aggregate(["s"], [], bad)
            with self.assertRaises(ValueError):
                aggregate(["s"], [ScoredPair("s", "a", bad)], 0.5)
        with self.assertRaises(ValueError):
            aggregate(["other"], scores, 0.5)
        with self.assertRaises(ValueError):
            aggregate(["s"], scores + scores, 0.5)

    def test_threshold_sweep_matches_brute_force(self):
        rng = random.Random(42)
        for _ in range(40):
            truth = {str(i): ({"a", "b"} if i % 2 else set()) for i in range(5)}
            scores = [ScoredPair(source, target, rng.choice([0, 0.2, 0.7, 1]))
                      for source in truth for target in ("a", "x")]
            policies = [None] + sorted({s.match_probability for s in scores}, reverse=True)
            metrics = [evaluate(truth, aggregate(truth, scores, threshold))["macro_f0_5"] for threshold in policies]
            best = max(metrics)
            expected = next(p for p, value in zip(policies, metrics) if math.isclose(value, best, abs_tol=1e-12))
            tuned = tune_threshold(truth, scores)
            self.assertAlmostEqual(tuned["macro_f0_5"], best)
            self.assertEqual(tuned["threshold"], expected)
        self.assertIsNone(tune_threshold({"s": set()}, [ScoredPair("s", "a", 1)])["threshold"])
        # Both numeric policies score 1/2; retain the higher numeric threshold.
        truth = {"a": {"x"}, "b": {"y"}, "c": set(), "d": {"z"}}
        scores = [ScoredPair("a", "x", .9), ScoredPair("b", "y", .5), ScoredPair("c", "bad", .5)]
        self.assertEqual(tune_threshold(truth, scores)["threshold"], .9)

    def test_cli_runs_without_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.tsv"
            output = Path(directory) / "empty.tsv"
            write_matches(truth, {"001": set(), "002": {"b"}})
            result = subprocess.run([sys.executable, "-S", "-m", "src", "baseline", "--truth", str(truth),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"macro_f0_5": 0.5', result.stdout)


if __name__ == "__main__":
    unittest.main()
