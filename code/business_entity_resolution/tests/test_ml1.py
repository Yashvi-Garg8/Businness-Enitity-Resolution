"""ML-1 regression, diagnostic, and CLI integration tests using synthetic records."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
if HAS_PANDAS:
    import pandas as pd
    from src.blocking import (BlockingConfig, SOURCE_COLUMNS, format_and_save_candidates,
                              generate_candidate_pairs, generate_candidate_pairs_with_report,
                              get_keys_for_row, read_source)
    from src.io_utils import PAIR_COLUMNS, read_pairs, write_matches, write_rows
    from src.normalize import address_components, canonical_country, normalize_dataset


def records(rows):
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def pair_set(frame):
    return set(frame.itertuples(index=False, name=None))


@unittest.skipUnless(HAS_PANDAS, "Install pandas for ML-1 tests")
class NormalizationTests(unittest.TestCase):
    def test_conservative_fields_and_legacy_compatibility(self):
        raw = records([("S1-0001", "A & B New Bank Hotel Services", "12 Main Rd", "U.S.A.")])
        before = raw.copy(deep=True)
        result = normalize_dataset(raw).iloc[0]
        pd.testing.assert_frame_equal(raw, before)
        self.assertEqual(result.norm_business_name, "a b new bank hotel services")
        self.assertEqual(result.norm_business_address, "12 main rd")
        self.assertEqual(result.norm_country, "u s a")
        self.assertEqual(result.feature_business_name, "a and b new bank hotel services")
        self.assertEqual(result.block_business_name, "b")
        self.assertEqual(result.feature_business_address, "12 main rd")
        self.assertEqual(result.block_business_address, "12 main road")
        self.assertEqual(result.block_country, "us")

    def test_unicode_missing_fields_and_idempotence(self):
        raw = records([("S1-0001", "Ｃafé & Société", None, None)])
        normalized = normalize_dataset(raw)
        self.assertEqual(normalized.iloc[0].feature_business_name, "café and société")
        self.assertEqual(normalized.iloc[0].block_country, "")
        self.assertEqual(normalized.iloc[0].postal_code, "")
        pd.testing.assert_frame_equal(normalized, normalize_dataset(normalized))
        self.assertEqual(canonical_country("New Zealand"), "new zealand")
        self.assertEqual(canonical_country("India"), "india")
        self.assertEqual(canonical_country("FR"), "france")

    def test_address_components_are_conservative(self):
        cases = [
            ("12 Main Rd 110001", "india", ("110001", "12", "main")),
            ("110001", "india", ("110001", "", "")),
            ("75001 Paris", "france", ("75001", "", "")),
            ("No. 12A Rue Victor 75001", "france", ("75001", "12a", "rue")),
            ("12 Main Street 90210-1234", "us", ("90210", "12", "main")),
            ("12-14 Main Rd", "india", ("", "", "")),
            ("12/14 Main Rd", "india", ("", "", "")),
            ("12 14 Main Rd", "india", ("", "", "")),
            ("Near Main Road 12", "india", ("", "", "")),
            ("12345", "unseen country", ("", "", "")),
        ]
        for address, country, expected in cases:
            with self.subTest(address=address):
                self.assertEqual(address_components(address, country), expected)


@unittest.skipUnless(HAS_PANDAS, "Install pandas for ML-1 tests")
class BlockingTests(unittest.TestCase):
    def test_acronym_and_short_exact_name_recovery(self):
        s1 = records([("S1-1", "State Bank of India", "12 Main Rd", "India"),
                      ("S1-2", "XY", "", "India")])
        targets = records([("S2-1", "SBI", "12 Main Road", "IN"), ("S3-1", "xy", "", "India")])
        self.assertEqual(pair_set(generate_candidate_pairs(s1, targets)),
                         {("S1-1", "S2-1"), ("S1-2", "S3-1")})

    def test_country_aliases_unknown_labels_and_missing_country(self):
        s1 = records([("S1-1", "Acme Global", "", "U.S."),
                      ("S1-2", "Café Bleu", "", "France"),
                      ("S1-3", "Future Country Shop", "", "New Zealand"),
                      ("S1-4", "Missing Country Shop", "", "")])
        targets = records([("S2-1", "Acme Global", "", "United States of America"),
                           ("S2-2", "Acme Global", "", "France"),
                           ("S3-1", "Café Bleu", "", "FR"),
                           ("S3-2", "Future Country Shop", "", "New Zealand"),
                           ("S3-3", "Missing Country Shop", "", "India")])
        pairs, report = generate_candidate_pairs_with_report(s1, targets)
        self.assertEqual(pair_set(pairs), {("S1-1", "S2-1"), ("S1-2", "S3-1"), ("S1-3", "S3-2")})
        self.assertEqual(report["missing_country_source_ids"], ["S1-4"])
        self.assertEqual(report["zero_candidate_source_ids"], ["S1-4"])

    def test_bucket_boundary_and_explicit_loss(self):
        s1 = records([("S1-1", "Acme Global", "", "India")])
        for size in (500, 501):
            targets = records([(f"S2-{i:04d}", "Acme Global", "", "India") for i in range(size)])
            pairs, report = generate_candidate_pairs_with_report(s1, targets, truth={"S1-1": {"S2-0000"}})
            self.assertEqual(len(pairs), 500 if size == 500 else 0)
            self.assertEqual(report["evaluation"]["pair_recall_before_filtering"], 1)
            self.assertEqual(report["evaluation"]["true_pairs_lost_to_bucket_filtering"], int(size == 501))
            if size == 501:
                self.assertGreater(report["skipped_buckets"], 0)
                self.assertEqual(report["unresolved_oversized_source_ids"], ["S1-1"])
                uncapped = generate_candidate_pairs(s1, targets, config=BlockingConfig(bucket_policy="uncapped"))
                self.assertEqual(len(uncapped), 501)

    def test_refinement_recovers_match_missed_by_drop(self):
        s1 = records([("S1-1", "Acme Holdings", "12 Main Rd", "India")])
        targets = records([(f"S2-{i:04d}", "Acmex Services", f"{i} Main Road", "India")
                           for i in range(1, 502)])
        truth = {"S1-1": {"S2-0012"}}
        refined, report = generate_candidate_pairs_with_report(s1, targets, truth=truth)
        dropped = generate_candidate_pairs(s1, targets, config=BlockingConfig(bucket_policy="drop"))
        uncapped = generate_candidate_pairs(s1, targets, config=BlockingConfig(bucket_policy="uncapped"))
        self.assertFalse(len(dropped))
        self.assertEqual(pair_set(refined), {("S1-1", "S2-0012")})
        self.assertLessEqual(pair_set(refined), pair_set(uncapped))
        self.assertEqual(report["recovered_source_ids"], ["S1-1"])
        self.assertGreater(report["refined_buckets"], 0)
        self.assertEqual(report["evaluation"]["pair_recall_after_filtering"], 1)

    def test_oversized_secondary_bucket_is_reported(self):
        s1 = records([("S1-1", "Acme Holdings", "12 Main Rd 110001", "India")])
        targets = records([(f"S2-{i}", "Acmex Services", "12 Main Road 110001", "India") for i in range(501)])
        pairs, report = generate_candidate_pairs_with_report(s1, targets)
        self.assertEqual(len(pairs), 0)
        self.assertGreater(report["skipped_sub_buckets"], 0)
        self.assertEqual(report["zero_candidate_source_count"], 1)

    def test_final_union_has_no_arbitrary_top_15_limit(self):
        s1 = records([("S1-1", "Acme Global", "", "India")])
        targets = records([(f"S2-{i:04d}", "Acme Global", "", "India") for i in range(30)])
        self.assertEqual(len(generate_candidate_pairs(s1, targets)), 30)

    def test_no_ambiguous_string_key_collisions(self):
        left = normalize_dataset(records([("S1-1", "ab c", "", "India")])).iloc[0]
        right = normalize_dataset(records([("S2-1", "a bc", "", "India")])).iloc[0]
        left_sorted = {key for key in get_keys_for_row(left) if key[0] == "name_sorted"}
        right_sorted = {key for key in get_keys_for_row(right) if key[0] == "name_sorted"}
        self.assertFalse(left_sorted & right_sorted)
        self.assertTrue(all(isinstance(key, tuple) for key in get_keys_for_row(left)))

    def test_truth_is_diagnostic_only_and_oracle_includes_singletons(self):
        s1 = records([("S1-1", "Acme Global", "", "India"), ("S1-2", "Unknown Shop", "", "India"),
                      ("S1-3", "Unlabeled Shop", "", "India")])
        targets = records([("S2-1", "Acme Global", "", "India")])
        plain = generate_candidate_pairs(s1, targets)
        for truth in ({"S1-1": {"S2-1"}, "S1-2": set()}, {"S1-1": set(), "S1-2": {"S2-1"}}):
            pairs, report = generate_candidate_pairs_with_report(s1, targets, truth=truth)
            pd.testing.assert_frame_equal(plain, pairs)
            self.assertEqual(report["evaluation"]["labeled_source_count"], 2)
            self.assertEqual(report["evaluation"]["unlabeled_source_count"], 1)
        _, report = generate_candidate_pairs_with_report(s1, targets, truth={"S1-2": set()})
        self.assertIsNone(report["evaluation"]["pair_recall_after_filtering"])
        self.assertEqual(report["evaluation"]["candidate_oracle_after_filtering"], 1)

    def test_input_validation_and_empty_tables(self):
        s1 = records([("S1-1", "Acme Global", "", "India")])
        targets = records([("S2-1", "Acme Global", "", "India")])
        for bad in (pd.concat([targets, targets]), records([("S1-wrong", "Acme", "", "India")]),
                    records([("", "Acme", "", "India")]), records([(123, "Acme", "", "India")])):
            with self.assertRaises(ValueError):
                generate_candidate_pairs(s1, bad)
        with self.assertRaises(ValueError):
            generate_candidate_pairs(pd.concat([s1, s1]), targets)
        for config in ((0, "refine"), (1, "unknown"), (True, "refine")):
            with self.assertRaises(ValueError):
                BlockingConfig(*config)
        for sources, candidates in ((s1, records([])), (records([]), targets), (records([]), records([]))):
            pairs, report = generate_candidate_pairs_with_report(sources, candidates)
            self.assertEqual(list(pairs.columns), list(PAIR_COLUMNS))
            self.assertEqual(len(pairs), 0)
            self.assertIsNone(report["reduction_ratio"])

    def test_target_order_does_not_change_outputs(self):
        s1 = records([("S1-2", "Acme Global", "", "India"), ("S1-1", "Acme Global", "", "India")])
        targets = records([("S3-2", "Acme Global", "", "India"), ("S2-1", "Acme Global", "", "India")])
        result = generate_candidate_pairs(s1, targets)
        pd.testing.assert_frame_equal(result, generate_candidate_pairs(s1, targets.iloc[::-1]))
        self.assertEqual(result.values.tolist(), [["S1-2", "S2-1"], ["S1-2", "S3-2"],
                                                  ["S1-1", "S2-1"], ["S1-1", "S3-2"]])


@unittest.skipUnless(HAS_PANDAS, "Install pandas for ML-1 tests")
class BlockingCLITests(unittest.TestCase):
    def test_grouped_writer_handles_bare_filename_and_empty_rows(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                s1 = records([("S1-0002", "", "", ""), ("S1-0001", "", "", "")])
                pairs = pd.DataFrame([("S1-0001", "S3-0001"), ("S1-0001", "S2-0001")], columns=PAIR_COLUMNS)
                format_and_save_candidates(pairs, s1, "candidate_pairs.tsv")
                self.assertEqual(Path("candidate_pairs.tsv").read_text(),
                                 "source1_entity_id\tcandidate_entity_ids\nS1-0002\t\nS1-0001\tS2-0001,S3-0001\n")
                format_and_save_candidates(pairs.iloc[:0], s1.iloc[:0], "empty.tsv")
                self.assertEqual(Path("empty.tsv").read_text(), "source1_entity_id\tcandidate_entity_ids\n")
            finally:
                os.chdir(original)

    def test_cli_outputs_are_deterministic_and_ready_for_ml3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "s1.tsv", SOURCE_COLUMNS, [("S1-0001", "Acme Global", "12 Main Rd", "France"),
                                                         ("S1-0002", "Unique Shop", "", "NA")])
            write_rows(root / "s2.tsv", SOURCE_COLUMNS, [("S2-0001", "Acme Global", "12 Main Road", "FR")])
            write_rows(root / "s3.tsv", SOURCE_COLUMNS, [])
            write_matches(root / "truth.tsv", {"S1-0001": {"S2-0001"}, "S1-0002": set()})
            self.assertEqual(read_source(root / "s1.tsv", "S1-").iloc[1].country, "NA")
            for run in (1, 2):
                result = subprocess.run([sys.executable, "-m", "src.blocking", "--s1", str(root / "s1.tsv"),
                                         "--s2", str(root / "s2.tsv"), "--s3", str(root / "s3.tsv"),
                                         "--truth", str(root / "truth.tsv"), "--out", str(root / f"out{run}")],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("candidate_pairs.tsv", "candidate_pairs_long.tsv", "blocking_report.json"):
                self.assertEqual((root / "out1" / name).read_bytes(), (root / "out2" / name).read_bytes())
            self.assertEqual(read_pairs(root / "out1/candidate_pairs_long.tsv"), [("S1-0001", "S2-0001")])
            report = json.loads((root / "out1/blocking_report.json").read_text())
            self.assertEqual(report["evaluation"]["candidate_oracle_after_filtering"], 1)
            self.assertEqual(report["reduction_ratio"], .5)

    def test_import_has_no_dataset_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PYTHONPATH=str(Path.cwd()), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-c", "import src.blocking"], cwd=directory,
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
