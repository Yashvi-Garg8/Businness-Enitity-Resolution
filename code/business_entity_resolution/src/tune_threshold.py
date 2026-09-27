"""Generate current ML-2 features and run ML-3's grouped model/threshold selection.

Use --help for inputs. Real training requires --protocol-confirmed; --synthetic
is only for fixtures. The fixed holdout is never used to select a threshold.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import pandas as pd

from .aggregate import aggregate_to_tsv_format
from .blocking import BlockingConfig, generate_candidate_pairs_with_report, read_source
from .features import FEATURE_COLS, FEATURE_VERSION, compute_pairwise_features
from .io_utils import PAIR_COLUMNS, read_matches, write_json, write_rows
from .model import FeatureTable, file_fingerprint, predict_matches, train
from .normalize import normalize_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("s1", "s2", "s3", "ground-truth", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bucket-limit", type=int, default=500)
    parser.add_argument("--bucket-policy", choices=("refine", "drop", "uncapped"), default="refine")
    protocol = parser.add_mutually_exclusive_group(required=True)
    protocol.add_argument("--protocol-confirmed", action="store_true",
                          help="Declare official metric/schema/ID rules and exhaustive labels verified")
    protocol.add_argument("--synthetic", action="store_true", help="Use synthetic fixtures only")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory must be empty to avoid mixing experiment artifacts")
        source = normalize_dataset(read_source(args.s1, "S1-"))
        targets = normalize_dataset(pd.concat([
            read_source(args.s2, "S2-"), read_source(args.s3, "S3-")], ignore_index=True))
        truth = read_matches(args.ground_truth)
        source_ids = source["entity_id"].tolist()
        if set(truth) - set(source_ids):
            raise ValueError("Ground truth contains IDs outside the Source 1 universe")
        # Blocking diagnostics require a fully scoped universe; unlabeled rows
        # must not be silently interpreted as negative examples.
        labeled_source = source[source["entity_id"].isin(truth)].copy()
        pairs, blocking = generate_candidate_pairs_with_report(
            labeled_source, targets, config=BlockingConfig(args.bucket_limit, args.bucket_policy), truth=truth)
        features = compute_pairwise_features(pairs, labeled_source, targets, n_jobs=args.workers)
        table = FeatureTable(list(pairs.itertuples(index=False, name=None)),
                             list(FEATURE_COLS), features[FEATURE_COLS].to_numpy(dtype=float))
        baseline_frame = aggregate_to_tsv_format(
            source_ids, predict_matches(features), "matched_entity_ids")
        baseline = {source_id: set(matches.split(',')) if matches else set()
                    for source_id, matches in baseline_frame.itertuples(index=False, name=None)}
        report = train(source_ids, truth, table.pairs, table, output,
                       "synthetic" if args.synthetic else "official_confirmed", baseline=baseline,
                       fingerprints={name: file_fingerprint(getattr(args, name))
                                     for name in ("s1", "s2", "s3", "ground_truth")},
                       feature_version=FEATURE_VERSION)
        write_json(output / "blocking_report.json", blocking)
        write_rows(output / "candidate_pairs_long.tsv", PAIR_COLUMNS, table.pairs)
        features.to_csv(output / "features.tsv", sep="\t", index=False)
        baseline_frame.to_csv(output / "ml2_baseline.tsv", sep="\t", index=False)
        print(json.dumps({key: report[key] for key in
                         ("status", "limitation", "threshold", "predict_none") if key in report}, indent=2))
        return 2 if report["status"] == "insufficient_data" else 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
