import os
import sys
import time
import argparse
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.normalize import normalize_dataset
from src.blocking import BlockingConfig, generate_candidate_pairs_with_report, read_source
from src.features import compute_pairwise_features
from src.model import predict_matches
from src.aggregate import aggregate_to_tsv_format
from src.validate import validate_outputs
from src.io_utils import PAIR_COLUMNS, read_matches, validate_pairs, write_json, write_rows

def run(s1_path: str, s2_path: str, s3_path: str, out_dir: str, model_path: str = None,
        *, blocking_config=None, truth_path=None):
    total_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    match_file = os.path.join(out_dir, "matching_results.tsv")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")

    print("[1/5] Ingesting & Normalizing Sources...")
    s1_df = read_source(s1_path, "S1-")
    s2_df = read_source(s2_path, "S2-")
    s3_df = read_source(s3_path, "S3-")

    target_combined = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_norm = normalize_dataset(s1_df)
    target_norm = normalize_dataset(target_combined)

    print("[2/5] Running Blocking / Candidate Generation...")
    candidate_pairs, blocking_report = generate_candidate_pairs_with_report(
        s1_norm, target_norm, config=blocking_config,
        truth=read_matches(truth_path) if truth_path else None)
    all_s1_ids = s1_df["entity_id"].tolist()
    write_json(os.path.join(out_dir, "blocking_report.json"), blocking_report)
    write_rows(os.path.join(out_dir, "candidate_pairs_long.tsv"), PAIR_COLUMNS,
               candidate_pairs.itertuples(index=False, name=None))

    cand_aggregated = aggregate_to_tsv_format(all_s1_ids, candidate_pairs, "candidate_entity_ids")
    cand_aggregated.to_csv(cand_file, sep="\t", index=False)
    print(f"      Saved {cand_file} ({len(candidate_pairs)} candidate pairs shortlisted)")

    print("[3/5] Extracting Pairwise Features...")
    features = compute_pairwise_features(candidate_pairs, s1_norm, target_norm)
    expected_pairs = validate_pairs(candidate_pairs[["source1_entity_id", "target_entity_id"]].itertuples(index=False, name=None))
    scored_pairs = validate_pairs(features[["source1_entity_id", "target_entity_id"]].itertuples(index=False, name=None))
    if set(scored_pairs) != set(expected_pairs):
        raise ValueError("Feature rows must cover exactly the exported candidate pairs")

    print("[4/5] ML Matching Inference & Precision Thresholding...")
    matches_pairwise = predict_matches(features, model_path=model_path)

    print("[5/5] Aggregating Final Matches...")
    matches_aggregated = aggregate_to_tsv_format(all_s1_ids, matches_pairwise, "matched_entity_ids")
    matches_aggregated.to_csv(match_file, sep="\t", index=False)
    print(f"      Saved {match_file}")

    print(f"[*] Pipeline finished in {time.time() - total_start:.2f}s")

    if not validate_outputs(match_file, cand_file, s1_path, s2_path, s3_path):
        print("[CRITICAL] Pipeline outputs failed validation!")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--out", default="output")
    parser.add_argument("--model", help="Optional trained ML-3 model.pkl; otherwise use the existing threshold baseline")
    parser.add_argument("--bucket-limit", type=int, default=500)
    parser.add_argument("--bucket-policy", choices=("refine", "drop", "uncapped"), default="refine")
    parser.add_argument("--truth", help="Optional training truth for blocking diagnostics only")
    args = parser.parse_args()

    run(args.s1, args.s2, args.s3, args.out, args.model,
        blocking_config=BlockingConfig(args.bucket_limit, args.bucket_policy), truth_path=args.truth)
