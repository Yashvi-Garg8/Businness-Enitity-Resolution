"""Deterministic multi-key blocking with bounded oversized-bucket recovery.

Run ``python -m src.blocking --help`` for the standalone ML-1 CLI.
Ground truth is optional diagnostics only and never participates in generation.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .evaluate import candidate_oracle, evaluate, validate_match_map
from .io_utils import PAIR_COLUMNS, identifier, read_matches, read_rows, validate_pairs, write_json, write_rows
from .normalize import LEGAL_SUFFIX_WORDS, NORMALIZED_FIELDS, normalize_dataset

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")
ACRONYM_CONNECTORS = {"a", "an", "the", "of", "and", "in", "for", "at", "on"}


@dataclass(frozen=True)
class BlockingConfig:
    bucket_limit: int = 500
    bucket_policy: str = "refine"

    def __post_init__(self):
        if isinstance(self.bucket_limit, bool) or not isinstance(self.bucket_limit, int) or self.bucket_limit < 1:
            raise ValueError("bucket_limit must be a positive integer")
        if self.bucket_policy not in {"refine", "drop", "uncapped"}:
            raise ValueError("bucket_policy must be refine, drop, or uncapped")


def validate_records(frame, prefixes):
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing or frame.columns.duplicated().any():
        raise ValueError(f"Source records require unique columns {SOURCE_COLUMNS}; missing={sorted(missing)}")
    ids = [identifier(value) for value in frame["entity_id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate entity IDs in source records")
    for value in ids:
        if not any(value.startswith(prefix) and len(value) > len(prefix) for prefix in prefixes):
            raise ValueError(f"Expected ID prefix {prefixes}, got {value!r}")
    result = frame.copy()
    result["entity_id"] = ids
    return result


def read_source(path, prefix):
    headers, rows = read_rows(path, SOURCE_COLUMNS)
    return validate_records(pd.DataFrame(rows, columns=headers), (prefix,))


def extract_clean_stem(text):
    # Four-character stems retain typo-tolerant recall from the previous blocker.
    packed = "".join(character for character in text if character.isalnum())
    return packed[:4] if len(packed) >= 3 else ""


def name_acronyms(name):
    words = name.split()
    while words and words[-1] in LEGAL_SUFFIX_WORDS:
        words.pop()
    variants = set()
    if len(words) >= 2:
        variants.add("".join(word[0] for word in words))
        significant = [word for word in words if word not in ACRONYM_CONNECTORS]
        variants.add("".join(word[0] for word in significant))
    elif words and words[0].isalpha():
        variants.add(words[0])
    return {variant for variant in variants if 2 <= len(variant) <= 8 and variant.isalpha()}


def get_keys_for_row(row):
    country = row.block_country
    name, address = row.block_business_name, row.block_business_address
    name_words, address_words = name.split(), address.split()
    first_name = name_words[0] if name_words else ""
    street, postal = row.street_number, row.postal_code
    keys = set()
    if len(name_words) >= 2:
        keys.add(("name_first_two", country, *name_words[:2]))
        keys.add(("name_sorted", country, *sorted(name_words)))
    stem = extract_clean_stem(name)
    if stem:
        keys.add(("name_stem", country, stem))
    packed_name = "".join(name_words)
    if len(packed_name) >= 3 and street:
        keys.add(("name_street", country, packed_name[:5], street))
    for acronym in name_acronyms(row.feature_business_name):
        if street:
            keys.add(("acronym_street", country, acronym, street))
        if postal:
            keys.add(("acronym_postal", country, acronym, postal))
    address_prefix = "".join(address_words)[:4]
    if len(first_name) >= 3 and len(address_prefix) >= 3:
        keys.add(("word_address", country, first_name, address_prefix))
    if postal and len(first_name) >= 2:
        keys.add(("word_postal", country, first_name, postal))
    if postal and street:
        keys.add(("location", country, postal, street))
    if len(address_words) >= 2 and len(first_name) >= 2:
        keys.add(("landmark", country, first_name, *address_words[:2]))
    if sum(character.isalnum() for character in row.feature_business_name) >= 2:
        keys.add(("exact_name", country, row.feature_business_name))
    return keys


def refinement_keys(row):
    keys = set()
    if row.postal_code:
        keys.add(("postal", row.postal_code))
    if row.street_number and row.street_token:
        keys.add(("street", row.street_number, row.street_token))
    return keys


def _prepared_rows(frame, prefixes):
    frame = validate_records(frame, prefixes)
    if not NORMALIZED_FIELDS.issubset(frame.columns):
        frame = normalize_dataset(frame)
    return list(frame.itertuples(index=False))


def generate_candidate_pairs_with_report(s1_df, target_df, *, config=None, truth=None):
    """Return (pairs DataFrame, JSON-ready diagnostics) in stable Source 1 order.

    Limits apply to buckets, never to the final union for a source. Normalized
    input must come from normalize_dataset; raw records are normalized here.
    """
    config = config or BlockingConfig()
    sources = _prepared_rows(s1_df, ("S1-",))
    targets = _prepared_rows(target_df, ("S2-", "S3-"))
    source_keys = {row.entity_id: get_keys_for_row(row) for row in sources}
    target_keys = {row.entity_id: get_keys_for_row(row) for row in targets}
    target_refinements = {row.entity_id: refinement_keys(row) for row in targets}
    index = defaultdict(list)
    for target_id, keys in target_keys.items():
        for key in keys:
            index[key].append(target_id)
    oversized = {key for key, ids in index.items() if len(ids) > config.bucket_limit}
    refined = {}
    skipped_sub_buckets = 0
    if config.bucket_policy == "refine":
        for key in oversized:
            sub_buckets = defaultdict(list)
            for target_id in index[key]:
                for secondary in target_refinements[target_id]:
                    sub_buckets[secondary].append(target_id)
            refined[key] = {secondary: ids for secondary, ids in sub_buckets.items()
                            if len(ids) <= config.bucket_limit}
            skipped_sub_buckets += sum(len(ids) > config.bucket_limit for ids in sub_buckets.values())

    pairs = []
    candidate_sets = {}
    affected, recovered, unresolved = [], [], []
    for row in sources:
        candidates = set()
        touched_oversized = recovered_any = unresolved_any = False
        for key in source_keys[row.entity_id]:
            if key not in index:
                continue
            if key not in oversized or config.bucket_policy == "uncapped":
                candidates.update(index[key])
            else:
                touched_oversized = True
                recovered_for_key = set()
                if config.bucket_policy == "refine":
                    for secondary in refinement_keys(row):
                        recovered_for_key.update(refined[key].get(secondary, ()))
                if recovered_for_key:
                    recovered_any = True
                    candidates.update(recovered_for_key)
                else:
                    unresolved_any = True
        candidate_sets[row.entity_id] = candidates
        pairs.extend((row.entity_id, target) for target in sorted(candidates))
        if touched_oversized:
            affected.append(row.entity_id)
        if recovered_any:
            recovered.append(row.entity_id)
        if unresolved_any:
            unresolved.append(row.entity_id)

    pairs_df = pd.DataFrame(pairs, columns=list(PAIR_COLUMNS))
    counts = pd.Series([len(candidate_sets[row.entity_id]) for row in sources], dtype=float)
    possible_pairs = len(sources) * len(targets)
    report = {
        "config": asdict(config), "source1_records": len(sources), "target_records": len(targets),
        "candidate_pairs": len(pairs), "possible_pairs": possible_pairs,
        "reduction_ratio": 1 - len(pairs) / possible_pairs if possible_pairs else None,
        "raw_buckets": len(index), "oversized_buckets": len(oversized),
        "refined_buckets": sum(bool(sub_buckets) for sub_buckets in refined.values()),
        "skipped_buckets": (len(oversized) if config.bucket_policy == "drop" else
                            sum(not sub_buckets for sub_buckets in refined.values())),
        "skipped_sub_buckets": skipped_sub_buckets,
        "affected_source_ids": affected, "recovered_source_ids": recovered,
        "unresolved_oversized_source_ids": unresolved,
        "zero_candidate_source_ids": [row.entity_id for row in sources if not candidate_sets[row.entity_id]],
        "zero_key_source_ids": [row.entity_id for row in sources if not source_keys[row.entity_id]],
        "missing_country_source_ids": [row.entity_id for row in sources if not row.block_country],
        "missing_country_target_ids": [row.entity_id for row in targets if not row.block_country],
        "candidate_count_distribution": {
            "min": int(counts.min()) if len(counts) else 0,
            "median": float(counts.median()) if len(counts) else 0,
            "p95": float(counts.quantile(.95)) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "mean": float(counts.mean()) if len(counts) else 0,
        },
    }
    for category in ("affected", "recovered", "unresolved_oversized", "zero_candidate", "zero_key", "missing_country"):
        report[category + "_source_count"] = len(report[category + "_source_ids"])
    report["missing_country_target_count"] = len(report["missing_country_target_ids"])
    if truth is not None:
        # Generation is already finished: labels influence only this report.
        validate_match_map(truth)
        if set(truth) - set(source_keys):
            raise ValueError("Ground truth references unknown Source 1 IDs")
        known_targets = set(target_keys)
        if any(actual - known_targets for actual in truth.values()):
            raise ValueError("Ground truth references targets absent from the supplied sources")
        before_predictions = {source: {target for target in actual if source_keys[source] & target_keys[target]}
                              for source, actual in truth.items()}
        scoped_pairs = [(source, target) for source, target in pairs if source in truth]
        after_predictions = candidate_oracle(truth, scoped_pairs)
        before = evaluate(truth, before_predictions)
        after = evaluate(truth, after_predictions, scoped_pairs)
        total = sum(len(actual) for actual in truth.values())
        retained_before = sum(len(actual) for actual in before_predictions.values())
        retained_after = sum(len(actual) for actual in after_predictions.values())
        report["evaluation"] = {
            "labeled_source_count": len(truth), "unlabeled_source_count": len(sources) - len(truth),
            "true_pairs": total, "retained_before_filtering": retained_before,
            "retained_after_filtering": retained_after,
            "true_pairs_lost_to_bucket_filtering": retained_before - retained_after,
            "pair_recall_before_filtering": retained_before / total if total else None,
            "pair_recall_after_filtering": retained_after / total if total else None,
            "candidate_oracle_before_filtering": before["macro_f0_5"],
            "candidate_oracle_after_filtering": after["macro_f0_5"],
            "singleton_fraction": sum(not actual for actual in truth.values()) / len(truth),
        }
    return pairs_df, report


def generate_candidate_pairs(s1_df, target_df, *, config=None):
    """Backward-compatible two-argument DataFrame API; there is no top-N cap."""
    return generate_candidate_pairs_with_report(s1_df, target_df, config=config)[0]


def format_and_save_candidates(pairs_df, s1_df, output_path):
    """Write official grouped candidates, including zero-candidate Source 1 rows."""
    ids = [identifier(value) for value in s1_df["entity_id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Source 1 IDs")
    pairs = validate_pairs(pairs_df[list(PAIR_COLUMNS)].itertuples(index=False, name=None), ids)
    grouped = {source: set() for source in ids}
    for source, target in pairs:
        if not target.startswith(("S2-", "S3-")):
            raise ValueError(f"Invalid target prefix: {target!r}")
        grouped[source].add(target)
    write_rows(output_path, CANDIDATE_COLUMNS,
               ((source, ','.join(sorted(targets))) for source, targets in grouped.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("s1", "s2", "s3"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--out", default="output")
    parser.add_argument("--truth", help="Optional training truth, used for diagnostics only")
    parser.add_argument("--bucket-limit", type=int, default=500)
    parser.add_argument("--bucket-policy", choices=("refine", "drop", "uncapped"), default="refine")
    args = parser.parse_args(argv)
    try:
        sources = read_source(args.s1, "S1-")
        targets = pd.concat([read_source(args.s2, "S2-"), read_source(args.s3, "S3-")], ignore_index=True)
        config = BlockingConfig(args.bucket_limit, args.bucket_policy)
        pairs, report = generate_candidate_pairs_with_report(
            sources, targets, config=config, truth=read_matches(args.truth) if args.truth else None)
        out = Path(args.out)
        format_and_save_candidates(pairs, sources, out / "candidate_pairs.tsv")
        write_rows(out / "candidate_pairs_long.tsv", PAIR_COLUMNS, pairs.itertuples(index=False, name=None))
        write_json(out / "blocking_report.json", report)
        print(f"Wrote {len(pairs)} candidate pairs and blocking diagnostics to {out}")
        if "evaluation" in report:
            print(f"Blocking pair recall: {report['evaluation']['pair_recall_after_filtering']}")
            print(f"Candidate oracle macro F0.5: {report['evaluation']['candidate_oracle_after_filtering']}")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


class TargetIndex:
    """One global target index, reusable across Source 1 batches.

    Call add() with normalized rows, then finalize() before candidate queries.
    Oversized memberships are discarded after refinement; retained-before counts
    inspect only labeled targets, not the full unfiltered Cartesian candidates.
    """
    def __init__(self, config=None):
        self.config = config or BlockingConfig()
        self.rows = {}
        self.buckets = defaultdict(list)
        self.oversized = set()
        self.refined = {}
        self.skipped_sub_buckets = 0
        self.ready = False

    def add(self, rows):
        if self.ready:
            raise ValueError("Cannot add targets after index finalization")
        for row in rows:
            if row.entity_id in self.rows:
                raise ValueError(f"Duplicate target entity ID: {row.entity_id}")
            self.rows[row.entity_id] = row
            for key in get_keys_for_row(row):
                self.buckets[key].append(row.entity_id)

    def finalize(self):
        if self.ready:
            return
        self.raw_buckets = len(self.buckets)
        self.oversized = {key for key, ids in self.buckets.items()
                          if len(ids) > self.config.bucket_limit}
        if self.config.bucket_policy != "uncapped":
            for key in self.oversized:
                ids = self.buckets.pop(key)
                if self.config.bucket_policy == "refine":
                    secondary = defaultdict(list)
                    overflow = set()
                    for target in ids:
                        for route in refinement_keys(self.rows[target]):
                            if route in overflow:
                                continue
                            secondary[route].append(target)
                            if len(secondary[route]) > self.config.bucket_limit:
                                del secondary[route]
                                overflow.add(route)
                    self.refined[key] = dict(secondary)
                    self.skipped_sub_buckets += len(overflow)
        self.ready = True

    def candidates(self, row):
        if not self.ready:
            raise ValueError("Finalize target index before querying")
        result = set()
        affected = recovered = unresolved = False
        keys = get_keys_for_row(row)
        for key in keys:
            if key in self.buckets:
                result.update(self.buckets[key])
            elif key in self.oversized:
                affected = True
                recovered_for_key = set()
                for route in refinement_keys(row):
                    recovered_for_key.update(self.refined.get(key, {}).get(route, ()))
                if recovered_for_key:
                    result.update(recovered_for_key)
                    recovered = True
                else:
                    unresolved = True
        return sorted(result), {
            "affected": affected, "recovered": recovered, "unresolved_oversized": unresolved,
            "zero_candidate": not result, "zero_key": not keys, "missing_country": not row.block_country}

    def retained_before(self, row, actual):
        keys = get_keys_for_row(row)
        return sum(bool(keys & get_keys_for_row(self.rows[target])) for target in actual)

    def statistics(self):
        return {"raw_buckets": self.raw_buckets, "oversized_buckets": len(self.oversized),
                "refined_buckets": sum(bool(value) for value in self.refined.values()),
                "skipped_buckets": (len(self.oversized) if self.config.bucket_policy == "drop" else
                                    sum(not value for value in self.refined.values())),
                "skipped_sub_buckets": self.skipped_sub_buckets}
