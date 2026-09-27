"""Run `python -m src --help` for the BE-1 integration interface."""

import argparse
import json
import sys

from .aggregate import aggregate
from .evaluate import candidate_oracle, empty_predictions, evaluate
from .io_utils import (read_matches, read_pairs, read_scores, read_source_ids,
                 write_json, write_matches, write_scores)


def parser():
    root = argparse.ArgumentParser(description="ML-3 entity resolution evaluation and modeling")
    commands = root.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("evaluate", help="Score every truth entity using macro F0.5")
    scoring.add_argument("--truth", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--candidates", help="Optional long-form pairs; predictions must be a subset")
    scoring.add_argument("--report")

    baseline = commands.add_parser("baseline", help="Write and score the all-empty baseline")
    baseline.add_argument("--truth", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--report")

    oracle = commands.add_parser("oracle", help="Score the best possible candidate-constrained predictions")
    oracle.add_argument("--truth", required=True)
    oracle.add_argument("--candidates", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--report")

    aggregation = commands.add_parser("aggregate", help="Threshold scores and emit one row per Source 1 ID")
    aggregation.add_argument("--source1", required=True)
    aggregation.add_argument("--scores", required=True)
    policy = aggregation.add_mutually_exclusive_group(required=True)
    policy.add_argument("--threshold", type=float)
    policy.add_argument("--predict-none", action="store_true")
    aggregation.add_argument("--output", required=True)

    training = commands.add_parser("train", help="Grouped model selection, held-out evaluation, and full refit")
    for name in ("source1", "truth", "candidates", "features", "output-dir"):
        training.add_argument("--" + name, required=True)
    training.add_argument("--feature-columns", nargs="+", required=True)
    training.add_argument("--ml2-baseline", help="Optional candidate-constrained baseline predictions")
    protocol = training.add_mutually_exclusive_group(required=True)
    protocol.add_argument("--protocol-confirmed", action="store_true",
                          help="Declare official metric, ID/schema rules, and exhaustive labels verified")
    protocol.add_argument("--synthetic", action="store_true", help="Inputs are synthetic fixtures only")

    inference = commands.add_parser("predict", help="Use a saved model and its selected threshold")
    for name in ("model", "source1", "features", "output"):
        inference.add_argument("--" + name, required=True)
    inference.add_argument("--scores-output")
    return root


def run(args):
    if args.command in {"evaluate", "baseline", "oracle"}:
        truth = read_matches(args.truth)
        candidates = read_pairs(args.candidates) if getattr(args, "candidates", None) else None
        if args.command == "evaluate":
            predictions = read_matches(args.predictions)
        elif args.command == "baseline":
            predictions = empty_predictions(truth)
        else:
            predictions = candidate_oracle(truth, candidates)
        report = evaluate(truth, predictions, candidates)
        report["protocol_status"] = "metric_confirmed_against_official_brief; dataset_validation_external"
        if args.command != "evaluate":
            write_matches(args.output, predictions)
        if args.report:
            write_json(args.report, report)
        return report, 0
    if args.command == "aggregate":
        predictions = aggregate(read_source_ids(args.source1), read_scores(args.scores), args.threshold)
        write_matches(args.output, predictions)
        return {"output": args.output, "entities": len(predictions)}, 0
    # Lazy import keeps evaluation and aggregation standard-library-only.
    from .model import file_fingerprint, load_model, predict, read_features, train
    if args.command == "train":
        paths = {name: getattr(args, name) for name in ("source1", "truth", "candidates", "features", "ml2_baseline")}
        report = train(
            read_source_ids(args.source1), read_matches(args.truth), read_pairs(args.candidates),
            read_features(args.features, args.feature_columns), args.output_dir,
            protocol="synthetic" if args.synthetic else "official_confirmed",
            baseline=read_matches(args.ml2_baseline) if args.ml2_baseline else None,
            fingerprints={name: file_fingerprint(path) for name, path in paths.items() if path},
        )
        summary = {key: report[key] for key in (
            "status", "protocol", "limitation", "threshold", "predict_none", "holdout_classifier",
            "holdout_improvement_over_empty", "holdout_improvement_over_ml2") if key in report}
        summary["output_dir"] = args.output_dir
        return summary, 2 if report["status"] == "insufficient_data" else 0
    artifact = load_model(args.model)
    predictions, scores = predict(artifact, read_source_ids(args.source1),
                                  read_features(args.features, artifact["feature_schema"]))
    write_matches(args.output, predictions)
    if args.scores_output:
        write_scores(args.scores_output, scores)
    return {"output": args.output, "entities": len(predictions), "protocol": artifact["protocol"]}, 0


def main():
    args = parser().parse_args()
    try:
        result, code = run(args)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
