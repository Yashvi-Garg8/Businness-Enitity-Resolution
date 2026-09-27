"""Compatibility tests for the repository's existing BE-1 entry points."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.io_utils import PAIR_COLUMNS, read_matches, write_matches, write_rows


class LegacyEvaluatorTests(unittest.TestCase):
    def test_direct_script_and_error_dump_without_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, predictions, errors = root / "truth.tsv", root / "predictions.tsv", root / "errors.tsv"
            write_matches(truth, {"S1-001": {"S2-001"}, "S1-002": set()})
            write_matches(predictions, {"S1-001": set(), "S1-002": set()})
            result = subprocess.run([sys.executable, "-S", "src/evaluate.py", str(predictions),
                                     str(truth), str(errors)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0.50000", result.stdout)
            self.assertIn("S1-001", errors.read_text())


HAS_PIPELINE = all(importlib.util.find_spec(name) is not None
                   for name in ("numpy", "pandas", "sklearn", "rapidfuzz", "tqdm"))


@unittest.skipUnless(HAS_PIPELINE, "Install the shared pipeline requirements")
class SharedPipelineTests(unittest.TestCase):
    def test_baseline_api_preserves_input(self):
        import pandas as pd
        from src.model import predict_matches

        frame = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"],
                              "target_entity_id": ["S2-1", "S3-1"],
                              "token_sort": [.95, .3], "address_ratio": [.95, .3]})
        original = frame.copy(deep=True)
        result = predict_matches(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(list(result.itertuples(index=False, name=None)), [("S1-1", "S2-1")])

    def test_grouped_candidate_schema_order_and_deduplication(self):
        import pandas as pd
        from src.aggregate import aggregate_to_tsv_format

        frame = pd.DataFrame([("S1-1", "S3-1"), ("S1-1", "S2-1"), ("S1-1", "S2-1")],
                             columns=list(PAIR_COLUMNS))
        result = aggregate_to_tsv_format(["S1-2", "S1-1"], frame, "candidate_entity_ids")
        self.assertEqual(list(result.columns), ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(result["source1_entity_id"].tolist(), ["S1-2", "S1-1"])
        self.assertEqual(result["candidate_entity_ids"].tolist(), ["", "S2-1,S3-1"])

    def test_existing_pipeline_runs_for_unseen_country(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_rows(root / "s1.tsv", columns,
                       [("S1-001", "Example Commerce", "12 Main Road", "France"),
                        ("S1-002", "Unique Cafe", "99 Other Road", "India")])
            write_rows(root / "s2.tsv", columns,
                       [("S2-001", "Example Commerce", "12 Main Road", "France")])
            write_rows(root / "s3.tsv", columns,
                       [("S3-001", "Different Shop", "32 Side Road", "US")])
            for name, entry in (("direct", ["src/run_pipeline.py"]), ("module", ["-m", "src.run_pipeline"])):
                out = root / name
                result = subprocess.run([sys.executable, *entry, "--s1", str(root / "s1.tsv"),
                                         "--s2", str(root / "s2.tsv"), "--s3", str(root / "s3.tsv"),
                                         "--out", str(out)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertEqual(read_matches(out / "matching_results.tsv"),
                                 {"S1-001": {"S2-001"}, "S1-002": set()})
                candidates = pd.read_csv(out / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
                self.assertEqual(candidates.columns.tolist(), ["source1_entity_id", "candidate_entity_ids"])
                self.assertEqual(candidates["candidate_entity_ids"].tolist(), ["S2-001", ""])


if __name__ == "__main__":
    unittest.main()
