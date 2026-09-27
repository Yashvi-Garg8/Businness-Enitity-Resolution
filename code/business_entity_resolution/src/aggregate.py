"""Deterministic prediction aggregation and exact entity-macro threshold tuning."""

import math
from itertools import groupby

from .evaluate import empty_predictions, entity_score, validate_match_map
from .io_utils import validate_scores


def aggregate(source_ids, scored_pairs, threshold):
    """threshold=None explicitly means predict no matches, even at probability 1."""
    predictions = empty_predictions(source_ids)
    scores = validate_scores(scored_pairs, predictions)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Threshold must be null (predict none) or a finite number in [0, 1]")
    if threshold is not None:
        for score in scores:
            if score.match_probability >= threshold:
                predictions[score.source1_entity_id].add(score.target_entity_id)
    return predictions


def tune_threshold(truth, scored_pairs):
    """Sweep tied probabilities together in O(pairs log pairs), retaining all entities.

    Descending traversal keeps the higher threshold on ties. The initial
    predict-none policy outranks every numeric threshold on a tie.
    """
    validate_match_map(truth)
    scores = sorted(validate_scores(scored_pairs, truth), key=lambda s: -s.match_probability)
    counts = {source: [0, 0, len(actual)] for source, actual in truth.items()}
    total = float(sum(not actual for actual in truth.values()))
    best_score = total / len(truth)
    best_threshold = None
    for probability, tied in groupby(scores, key=lambda s: s.match_probability):
        for pair in tied:
            counts_for_source = counts[pair.source1_entity_id]
            previous = entity_score(*counts_for_source)
            if pair.target_entity_id in truth[pair.source1_entity_id]:
                counts_for_source[0] += 1
                counts_for_source[2] -= 1
            else:
                counts_for_source[1] += 1
            total += entity_score(*counts_for_source) - previous
        score = total / len(truth)
        if score > best_score + 1e-12:
            best_score, best_threshold = score, probability
    return {"threshold": best_threshold, "predict_none": best_threshold is None,
            "macro_f0_5": best_score}


def aggregate_to_tsv_format(all_s1_ids, pairwise_df, target_col_name):
    """BE-1's DataFrame interface for official grouped candidates or matches."""
    import pandas as pd
    from .io_utils import identifier

    if target_col_name not in {"candidate_entity_ids", "matched_entity_ids"}:
        raise ValueError("Expected candidate_entity_ids or matched_entity_ids")
    all_s1_ids = list(all_s1_ids)
    if not all_s1_ids:
        if not pairwise_df.empty:
            raise ValueError("Pairs cannot reference an empty Source 1 universe")
        return pd.DataFrame(columns=["source1_entity_id", target_col_name])
    grouped = empty_predictions(all_s1_ids)
    for source, target in pairwise_df[["source1_entity_id", "target_entity_id"]].itertuples(index=False, name=None):
        source, target = identifier(source), identifier(target)
        if source not in grouped:
            raise ValueError(f"Unknown Source 1 ID: {source!r}")
        grouped[source].add(target)
    return pd.DataFrame({"source1_entity_id": list(grouped),
                         target_col_name: [','.join(sorted(targets)) for targets in grouped.values()]})
