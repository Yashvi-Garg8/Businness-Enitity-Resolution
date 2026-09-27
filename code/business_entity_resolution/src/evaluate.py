"""Entity-macro F0.5, including the official brief's singleton convention."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .io_utils import identifier, validate_ids, validate_pairs

METRIC = "entity_macro_f0.5"


def entity_score(tp, fp, fn):
    denominator = 5 * tp + 4 * fp + fn
    return 5 * tp / denominator if denominator else 1.0


def validate_match_map(matches):
    validate_ids(matches)
    for source, targets in matches.items():
        if source != identifier(source):
            raise ValueError("Source IDs must already be stripped")
        if not isinstance(targets, (set, frozenset)):
            raise ValueError("Match maps must contain sets of target IDs")
        for target in targets:
            if target != identifier(target):
                raise ValueError("Target IDs must already be stripped")


def empty_predictions(source_ids):
    return {source: set() for source in validate_ids(source_ids)}


def candidate_map(source_ids, pairs):
    result = empty_predictions(source_ids)
    for source, target in validate_pairs(pairs, result):
        result[source].add(target)
    return result


def candidate_oracle(truth, pairs):
    validate_match_map(truth)
    candidates = candidate_map(truth, pairs)
    return {source: targets & candidates[source] for source, targets in truth.items()}


def evaluate(truth, predictions, candidates=None):
    """Score every truth entity. Optional candidates require constrained predictions.

    Pair precision/recall are diagnostics, not the optimized macro metric. Recall
    with no true pairs and singleton rate with no singletons are reported as null.
    """
    validate_match_map(truth)
    validate_match_map(predictions)
    if set(truth) != set(predictions):
        missing = sorted(set(truth) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(truth))[:5]
        raise ValueError(f"Prediction coverage mismatch: missing={missing}, extra={extra}")
    shortlisted = candidate_map(truth, candidates) if candidates is not None else None
    tp = fp = fn = singleton_count = singleton_errors = blocking_misses = classifier_misses = 0
    scores = []
    for source, actual in truth.items():
        predicted = predictions[source]
        correct, extra, missed = len(actual & predicted), len(predicted - actual), len(actual - predicted)
        scores.append(entity_score(correct, extra, missed))
        tp += correct
        fp += extra
        fn += missed
        if not actual:
            singleton_count += 1
            singleton_errors += bool(predicted)
        if shortlisted is not None:
            if predicted - shortlisted[source]:
                raise ValueError(f"Predictions outside candidate set for {source!r}")
            blocking_misses += len(actual - shortlisted[source])
            classifier_misses += len((actual & shortlisted[source]) - predicted)
    report = {
        "metric": METRIC, "macro_f0_5": sum(scores) / len(scores),
        "entities": len(truth), "true_pairs": tp + fn, "predicted_pairs": tp + fp,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "pair_precision": tp / (tp + fp) if tp + fp else None,
        "pair_recall": tp / (tp + fn) if tp + fn else None,
        "singleton_count": singleton_count, "singleton_false_matches": singleton_errors,
        "singleton_false_match_rate": singleton_errors / singleton_count if singleton_count else None,
    }
    if shortlisted is not None:
        oracle = {source: actual & shortlisted[source] for source, actual in truth.items()}
        report.update({
            "blocking_misses": blocking_misses, "classifier_misses": classifier_misses,
            "blocking_pair_recall": (tp + fn - blocking_misses) / (tp + fn) if tp + fn else None,
            "candidate_oracle_macro_f0_5": evaluate(truth, oracle)["macro_f0_5"],
            "zero_candidate_entities": sum(not targets for targets in shortlisted.values()),
        })
    return report


def compute_macro_f05(pred_file, ground_truth_file, dump_error_tsv=None):
    """Preserve the shared repository's scoring API and mismatch TSV interface."""
    from .io_utils import read_matches, write_rows

    predictions = read_matches(pred_file)
    truth = read_matches(ground_truth_file)
    report = evaluate(truth, predictions)
    if dump_error_tsv:
        write_rows(dump_error_tsv,
                   ("source1_entity_id", "ground_truth", "predicted", "false_positives", "false_negatives"),
                   ((source, ','.join(sorted(actual)), ','.join(sorted(predictions[source])),
                     ','.join(sorted(predictions[source] - actual)), ','.join(sorted(actual - predictions[source])))
                    for source, actual in truth.items() if actual != predictions[source]))
    print(f"Macro F_0.5 Evaluation Score: {report['macro_f0_5']:.5f}")
    print(f"Evaluated over {report['entities']} Source 1 records")
    return report["macro_f0_5"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("truth")
    parser.add_argument("errors", nargs="?")
    args = parser.parse_args()
    compute_macro_f05(args.predictions, args.truth, args.errors)
