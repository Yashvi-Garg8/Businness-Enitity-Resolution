"""Generate small deterministic inputs to exercise the ML-3 contract."""

import argparse
import random
from pathlib import Path

from src.io_utils import PAIR_COLUMNS, write_matches, write_rows

FEATURES = ["name_similarity", "address_similarity"]


def create_fixture(directory, count=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    truth, baseline, pairs, feature_rows = {}, {}, [], []
    source_ids = [f"S1-{index:04d}" for index in range(count)]
    for index, source in enumerate(source_ids):
        # Adjacent non-singletons share a true target and must stay in one group.
        target = f"S2-{index // 2:04d}"
        truth[source] = {target} if index % 5 else set()
        baseline[source] = set()
        if index % 13 == 0:
            continue  # Some true entities have no candidate rows at all.
        if truth[source] and index % 11:
            pairs.append((source, target))
            feature_rows.append((source, target, round(rng.uniform(.88, .99), 5), .92))
            baseline[source].add(target)
        for number in range(3):
            negative = f"S3-wrong_{number}"
            name = .93 if index % 5 == 0 and number == 0 else round(rng.uniform(.1, .4), 5)
            pairs.append((source, negative))
            feature_rows.append((source, negative, name, "" if number == 2 else .15))
            if name >= .9:
                baseline[source].add(negative)
    source_ids.append("S1-unlabeled")
    pairs.append(("S1-unlabeled", "S2-unlabeled"))
    feature_rows.append(("S1-unlabeled", "S2-unlabeled", .9, .9))
    baseline["S1-unlabeled"] = set()
    write_rows(directory / "source1.tsv", ("entity_id",), ((source,) for source in source_ids))
    write_matches(directory / "truth.tsv", truth)
    write_rows(directory / "candidates.tsv", PAIR_COLUMNS, pairs)
    write_rows(directory / "features.tsv", list(PAIR_COLUMNS) + FEATURES, feature_rows)
    write_matches(directory / "ml2_baseline.tsv", baseline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_fixture(args.output_dir)
    print(f"Synthetic fixture written to {args.output_dir}")
