"""ML dependencies are imported only when this module is explicitly requested."""

import hashlib
import importlib.metadata
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .aggregate import aggregate, tune_threshold
from .evaluate import candidate_oracle, empty_predictions, evaluate, validate_match_map
from .io_utils import (PAIR_COLUMNS, ScoredPair, read_rows, validate_ids,
                 validate_pairs, write_json, write_matches, write_rows, write_scores)

SEED = 42
FORBIDDEN_FEATURES = set(PAIR_COLUMNS) | {
    "entity_id", "matched_entity_ids", "label", "labels", "target", "y", "match_probability"
}


@dataclass
class FeatureTable:
    pairs: list
    names: list
    values: np.ndarray


def validate_feature_names(feature_names):
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("Register a nonempty, unique ordered list of numeric features")
    if any(name in FORBIDDEN_FEATURES or name.endswith("_entity_id") for name in names):
        raise ValueError("Identifiers, labels, and prediction columns cannot be model features")
    return names


def validate_feature_table(table):
    validate_feature_names(table.names)
    if table.values.shape != (len(table.pairs), len(table.names)):
        raise ValueError("Feature matrix shape does not match the registered schema")
    if not np.issubdtype(table.values.dtype, np.number) or np.isinf(table.values).any():
        raise ValueError("Features must be numeric with no infinite values; NaNs are allowed")


def read_features(path, feature_names):
    names = validate_feature_names(feature_names)
    _, rows = read_rows(path, list(PAIR_COLUMNS) + names, exact=True)
    pairs = validate_pairs((row[PAIR_COLUMNS[0]], row[PAIR_COLUMNS[1]]) for row in rows)
    values = []
    for line, row in enumerate(rows, 2):
        vector = []
        for name in names:
            try:
                value = float(row[name]) if row[name].strip() else float("nan")
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: feature {name!r} must be numeric or empty") from exc
            if np.isinf(value):
                raise ValueError(f"{path}:{line}: infinite feature {name!r}")
            vector.append(value)
        values.append(vector)
    return FeatureTable(pairs, names, np.asarray(values, dtype=float).reshape(len(rows), len(names)))


def truth_groups(truth):
    """Connected components through shared true targets, including singletons."""
    validate_match_map(truth)
    parent = {source: source for source in truth}

    def find(source):
        while source != parent[source]:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    owners = {}
    for source in sorted(truth):
        for target in sorted(truth[source]):
            if target in owners:
                left, right = find(source), find(owners[target])
                parent[max(left, right)] = min(left, right)
            else:
                owners[target] = source
    return {source: find(source) for source in sorted(truth)}


def make_splits(truth):
    groups = truth_groups(truth)
    sources = sorted(truth)
    assignments = {source: {"group": groups[source], "partition": "unsplit", "fold": None}
                   for source in sources}
    if len(set(groups.values())) < 2:
        return assignments, "Need at least two independent entity groups for a holdout"
    group_array = [groups[source] for source in sources]
    dev_indices, holdout_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED).split(sources, groups=group_array))
    development = [sources[index] for index in dev_indices]
    for source in development:
        assignments[source]["partition"] = "development"
    for index in holdout_indices:
        assignments[sources[index]]["partition"] = "holdout"
    if len({groups[source] for source in development}) < 3:
        return assignments, "Need at least three development groups after the fixed 20% holdout"
    for fold, (_, valid_indices) in enumerate(GroupKFold(n_splits=3).split(
            development, groups=[groups[source] for source in development])):
        for index in valid_indices:
            assignments[development[index]]["fold"] = fold
    return assignments, None


def dependency_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn")}}


def new_model(parameters):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, min_samples_leaf=20,
        random_state=SEED, early_stopping=False, categorical_features=None, **parameters)


def score_table(model, table, indices=None):
    indices = np.arange(len(table.pairs)) if indices is None else np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    probabilities = model.predict_proba(table.values[indices])[:, list(model.classes_).index(1)]
    return [ScoredPair(*table.pairs[index], float(probability))
            for index, probability in zip(indices, probabilities)]


def file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(source_ids, truth, candidates, table, output_dir, protocol, baseline=None, fingerprints=None, feature_version=None):
    """Train only when callers explicitly declare verified or synthetic inputs.

    `protocol='official_confirmed'` declares the metric, schema, ID rules and
    exhaustive labels were checked against the official brief. `synthetic` is
    for fixtures only. Neither mode establishes official compliance by itself.
    """
    if protocol not in {"official_confirmed", "synthetic"}:
        raise ValueError("Confirm the official protocol or explicitly use synthetic data before training")
    source_ids = validate_ids(source_ids)
    validate_match_map(truth)
    if set(truth) - set(source_ids):
        raise ValueError("Ground truth contains IDs outside the Source 1 universe")
    candidates = validate_pairs(candidates, source_ids)
    validate_pairs(table.pairs, source_ids)
    if set(table.pairs) != set(candidates):
        raise ValueError("Feature table must cover exactly the supplied candidate pairs")
    validate_feature_table(table)
    if baseline is not None:
        # A baseline may cover all source rows or only labeled source rows.
        validate_match_map(baseline)
        if set(truth) - set(baseline) or set(baseline) - set(source_ids):
            raise ValueError("ML-2 baseline must cover every labeled entity with no unknown Source 1 IDs")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output directory must be empty to avoid mixing experiment artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_indices = [i for i, (source, _) in enumerate(table.pairs) if source in truth]
    table = FeatureTable([table.pairs[i] for i in labeled_indices], table.names, table.values[labeled_indices])
    pairs = table.pairs
    labels = np.asarray([int(target in truth[source]) for source, target in pairs], dtype=int)
    assignments, limitation = make_splits(truth)
    write_rows(output_dir / "splits.tsv", ("source1_entity_id", "group_id", "partition", "fold"),
               ((source, row["group"], row["partition"], "" if row["fold"] is None else row["fold"])
                for source, row in assignments.items()))

    def subset_truth(partition):
        return {source: truth[source] for source in truth if assignments[source]["partition"] == partition}

    def subset_pairs(actual):
        return [(source, target) for source, target in pairs if source in actual]

    def baseline_reports(actual):
        scoped_pairs = subset_pairs(actual)
        reports = {
            "all_empty": evaluate(actual, empty_predictions(actual), scoped_pairs),
            "candidate_oracle": evaluate(actual, candidate_oracle(actual, scoped_pairs), scoped_pairs),
            "ml2": None,
        }
        if baseline is not None:
            # Validate comparable baseline candidates as well as coverage.
            reports["ml2"] = evaluate(actual, {source: baseline[source] for source in actual}, scoped_pairs)
        return reports

    report = {
        "status": "pending", "protocol": protocol,
        "metric": "entity_macro_f0.5", "seed": SEED,
        "feature_schema": table.names, "feature_version": feature_version, "versions": dependency_versions(),
        "input_sha256": fingerprints or {},
        "labeled_entities": len(truth), "excluded_unlabeled_entities": len(source_ids) - len(truth),
        "excluded_unlabeled_pairs": len(candidates) - len(pairs),
        "labeled_candidate_pairs": len(pairs), "positive_candidate_pairs": int(labels.sum()),
        "full_data_baselines": baseline_reports(truth),
        "ml2_baseline_available": baseline is not None,
    }
    write_matches(output_dir / "all_empty.tsv", empty_predictions(source_ids))
    development, holdout = subset_truth("development"), subset_truth("holdout")
    if holdout:
        report["holdout_baselines"] = baseline_reports(holdout)

    def indices_for(sources):
        return np.asarray([index for index, (source, _) in enumerate(pairs) if source in sources], dtype=int)

    dev_indices, holdout_indices = indices_for(development), indices_for(holdout)
    fold_indices = []
    if limitation is None:
        if set(labels[holdout_indices]) != {0, 1}:
            limitation = "The fixed holdout candidate labels do not contain both classes"
        for fold in range(3):
            valid_sources = {source for source in development if assignments[source]["fold"] == fold}
            train_indices = indices_for(set(development) - valid_sources)
            valid_indices = indices_for(valid_sources)
            if set(labels[train_indices]) != {0, 1} or set(labels[valid_indices]) != {0, 1}:
                limitation = f"Fold {fold} training/validation candidate labels do not contain both classes"
            fold_indices.append((train_indices, valid_indices))
    if limitation:
        report.update(status="insufficient_data", limitation=limitation)
        write_json(output_dir / "metrics.json", report)
        return report

    trials = []
    best = None
    for leaves in (15, 31):
        for regularization in (0, 1):
            parameters = {"max_leaf_nodes": leaves, "l2_regularization": regularization}
            out_of_fold = []
            fold_metrics = []
            for train_indices, valid_indices in fold_indices:
                model = new_model(parameters)
                model.fit(table.values[train_indices], labels[train_indices])
                out_of_fold.extend(score_table(model, table, valid_indices))
            tuned = tune_threshold(development, out_of_fold)
            predictions = aggregate(development, out_of_fold, tuned["threshold"])
            for fold in range(3):
                fold_truth = {source: truth[source] for source in development if assignments[source]["fold"] == fold}
                fold_metrics.append(evaluate(fold_truth, {source: predictions[source] for source in fold_truth},
                                             subset_pairs(fold_truth)))
            trial = {"parameters": parameters, **tuned, "fold_metrics": fold_metrics}
            trials.append(trial)
            # Fixed grid order resolves model ties reproducibly; tuning resolves threshold ties.
            if best is None or tuned["macro_f0_5"] > best[0]["macro_f0_5"] + 1e-12:
                best = (trial, out_of_fold)
    selected, oof_scores = best
    threshold = selected["threshold"]
    write_scores(output_dir / "oof_scored_pairs.tsv", sorted(
        oof_scores, key=lambda pair: (pair.source1_entity_id, pair.target_entity_id)))
    write_matches(output_dir / "oof_predictions.tsv", aggregate(development, oof_scores, threshold))

    development_model = new_model(selected["parameters"])
    development_model.fit(table.values[dev_indices], labels[dev_indices])
    holdout_scores = score_table(development_model, table, holdout_indices)
    holdout_predictions = aggregate(holdout, holdout_scores, threshold)
    write_scores(output_dir / "holdout_scored_pairs.tsv", holdout_scores)
    write_matches(output_dir / "holdout_predictions.tsv", holdout_predictions)
    report.update({
        "status": "trained", "trials": trials, "selected_parameters": selected["parameters"],
        "threshold": threshold, "predict_none": threshold is None,
        "oof_selection_macro_f0_5": selected["macro_f0_5"],
        "holdout_classifier": evaluate(holdout, holdout_predictions, subset_pairs(holdout)),
        "holdout_used_for_selection": False,
    })
    report["holdout_improvement_over_empty"] = (
        report["holdout_classifier"]["macro_f0_5"] - report["holdout_baselines"]["all_empty"]["macro_f0_5"])
    if baseline is not None:
        report["holdout_improvement_over_ml2"] = (
            report["holdout_classifier"]["macro_f0_5"] - report["holdout_baselines"]["ml2"]["macro_f0_5"])

    final_model = new_model(selected["parameters"])
    final_model.fit(table.values, labels)
    artifact = {"artifact_version": 1, "model": final_model, "feature_schema": table.names,
                "threshold": threshold, "parameters": selected["parameters"],
                "versions": report["versions"], "protocol": protocol, "feature_version": feature_version}
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output_dir / "model_metadata.json", {key: value for key, value in artifact.items() if key != "model"})
    write_json(output_dir / "metrics.json", report)
    return report


def load_model(path):
    with open(path, "rb") as handle:
        artifact = pickle.load(handle)
    if artifact.get("artifact_version") != 1:
        raise ValueError("Unsupported model artifact version")
    if artifact["versions"]["scikit-learn"] != importlib.metadata.version("scikit-learn"):
        raise ValueError("Use the scikit-learn version recorded in the model artifact")
    return artifact


def predict(artifact, source_ids, table):
    source_ids = validate_ids(source_ids)
    validate_feature_table(table)
    if list(table.names) != list(artifact["feature_schema"]):
        raise ValueError("Inference feature schema differs from the trained feature schema")
    validate_pairs(table.pairs, source_ids)
    scores = score_table(artifact["model"], table)
    return aggregate(source_ids, scores, artifact["threshold"]), scores


def predict_matches(feature_df, threshold=0.86, *, model_path=None, feature_version=None):
    """Preserve BE-1's DataFrame API, with optional trained-model inference.

    Without an artifact, retain the shared repository's original weighted
    similarity baseline. The caller's feature DataFrame is never mutated.
    """
    import pandas as pd

    pairs = validate_pairs(feature_df[list(PAIR_COLUMNS)].itertuples(index=False, name=None))
    if model_path is not None:
        artifact = load_model(model_path)
        if feature_version is not None and artifact.get("feature_version") != feature_version:
            raise ValueError("Model feature version differs from the pipeline; retrain with current features")
        names = artifact["feature_schema"]
        if set(feature_df.columns) != set(PAIR_COLUMNS) | set(names):
            raise ValueError("Inference feature schema differs from the trained feature schema")
        table = FeatureTable(pairs, names, feature_df[names].to_numpy(dtype=float))
        if not pairs:
            validate_feature_table(table)
            return pd.DataFrame(columns=list(PAIR_COLUMNS))
        predictions, _ = predict(artifact, list(dict.fromkeys(source for source, _ in pairs)), table)
        accepted = [(source, target) for source, target in pairs if target in predictions[source]]
    else:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Baseline threshold must be finite and within [0, 1]")
        if not pairs:
            return pd.DataFrame(columns=list(PAIR_COLUMNS))
        scores = 0.65 * feature_df["token_sort"] + 0.35 * feature_df["address_ratio"]
        accepted = [pair for pair, keep in zip(pairs, scores >= threshold) if keep]
    return pd.DataFrame(accepted, columns=list(PAIR_COLUMNS))
