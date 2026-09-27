"""
Diagnose and tune against train_ground_truth.tsv instead of guessing.

Reports:
  1. Blocking recall ceiling on a held-out validation split (the hard upper
     bound on F0.5 -- no matcher can beat this).
  2. A threshold sweep for the heuristic composite score in model.py.
  3. A threshold sweep for a trained logistic-regression classifier, fit on
     a disjoint training split so this isn't just memorizing the validation set.

Usage:
    python tune_threshold.py --s1 dataset/train/train_source1.tsv \
        --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv
"""
import argparse

import numpy as np
import pandas as pd

from normalize import normalize_dataset
from blocking import generate_candidate_pairs, check_blocking_recall
from features import compute_pairwise_features
from model import predict_matches, label_pairs, train_classifier, predict_matches_ml
from aggregate import aggregate_to_tsv_format
from evaluate import compute_macro_f05


def _score(pred_pairs_df, all_s1_ids, ground_truth_path, tmp_path="_tmp_tune_preds.tsv"):
    agg = aggregate_to_tsv_format(all_s1_ids, pred_pairs_df, "matched_entity_ids")
    agg.to_csv(tmp_path, sep="\t", index=False)
    return compute_macro_f05(tmp_path, ground_truth_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--max-entities", type=int, default=None,
                         help="Subsample to at most this many Source 1 entities before "
                              "splitting train/val -- use this for fast iteration on a "
                              "large dataset instead of tuning against the full file "
                              "every time.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel workers for feature scoring (CPU-bound; set to "
                              "your core count - 1 on large datasets).")
    args = parser.parse_args()

    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str)
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str)
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_norm = normalize_dataset(s1_df)
    target_norm = normalize_dataset(target_df)

    rng = np.random.RandomState(42)
    all_ids = s1_norm["entity_id"].dropna().unique()
    rng.shuffle(all_ids)

    if args.max_entities and len(all_ids) > args.max_entities:
        print(f"[tune] Subsampling {args.max_entities} of {len(all_ids)} Source 1 "
              f"entities for a fast iteration cycle.")
        all_ids = all_ids[: args.max_entities]
    n_val = int(len(all_ids) * args.val_fraction)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    val_df = s1_norm[s1_norm["entity_id"].isin(val_ids)].reset_index(drop=True)
    train_df = s1_norm[s1_norm["entity_id"].isin(train_ids)].reset_index(drop=True)

    print(f"[tune] {len(val_df)} validation entities, {len(train_df)} training entities")

    print("\n[tune] Blocking recall ceiling on validation split:")
    val_candidates = generate_candidate_pairs(val_df, target_norm)
    check_blocking_recall(val_candidates, args.ground_truth)
    val_features = compute_pairwise_features(val_candidates, val_df, target_norm, n_jobs=args.workers)

    print("\n[tune] Heuristic composite-score threshold sweep:")
    best_t, best_f05 = None, -1.0
    for t in np.arange(0.55, 0.96, 0.02):
        preds = predict_matches(val_features, threshold=round(t, 2))
        f05 = _score(preds, val_df["entity_id"].tolist(), args.ground_truth)
        print(f"    threshold={t:.2f}  macro_f0.5={f05:.4f}")
        if f05 > best_f05:
            best_t, best_f05 = round(t, 2), f05
    print(f"[tune] BEST heuristic threshold: {best_t} -> macro F0.5 = {best_f05:.4f}")

    print("\n[tune] Training classifier on the disjoint training split...")
    train_candidates = generate_candidate_pairs(train_df, target_norm)
    train_features = compute_pairwise_features(train_candidates, train_df, target_norm, n_jobs=args.workers)
    labeled_train = label_pairs(train_features, args.ground_truth)

    if labeled_train["label"].nunique() < 2:
        print("[tune] Training split's candidate pairs contain only one label "
              "class (all match or all non-match) -- skipping classifier. On a "
              "real-size dataset this shouldn't happen; on a tiny sample it's "
              "just not enough data. If blocking recall (above) is low, fix "
              "that first -- it's the more likely real bottleneck.")
        return

    clf = train_classifier(labeled_train)

    print("\n[tune] Classifier threshold sweep (evaluated on validation split):")
    best_ct, best_cf05 = None, -1.0
    for t in np.arange(0.3, 0.91, 0.02):
        preds = predict_matches_ml(clf, val_features, threshold=round(t, 2))
        f05 = _score(preds, val_df["entity_id"].tolist(), args.ground_truth)
        print(f"    threshold={t:.2f}  macro_f0.5={f05:.4f}")
        if f05 > best_cf05:
            best_ct, best_cf05 = round(t, 2), f05
    print(f"[tune] BEST classifier threshold: {best_ct} -> macro F0.5 = {best_cf05:.4f}")

    print("\n=== Summary ===")
    print(f"heuristic best:  threshold={best_t}  macro F0.5={best_f05:.4f}")
    print(f"classifier best: threshold={best_ct} macro F0.5={best_cf05:.4f}")
    print("Plug whichever is higher into model.py / run_pipeline.py.")


if __name__ == "__main__":
    main()