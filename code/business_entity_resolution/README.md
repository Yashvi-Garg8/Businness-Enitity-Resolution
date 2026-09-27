# Business Entity Resolution — ML-1 and ML-3

ML-1 provides normalization, candidate generation, and blocking diagnostics. ML-3
provides entity-level evaluation, supervised matching, threshold selection, and
prediction aggregation. See [ML1_HANDOFF.md](ML1_HANDOFF.md) for the implemented
ML-1 fixes and outstanding real-data checks.

The official brief confirms the implemented entity-macro F0.5 metric and singleton
convention. Training data is not included. Synthetic tests establish software
behavior, not challenge performance. Final model-license compliance remains to be
resolved: the brief requires MIT/Apache-2.0 models, while the current learner uses
scikit-learn (BSD-3-Clause). This branch does not claim submission readiness.

## Setup and tests

The pinned packages support Python 3.9–3.12; this setup was tested on Python 3.9.6.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

Evaluation and aggregation need only Python's standard library:

```bash
python3 -m src --help
python3 -m unittest discover -s tests -p 'test_evaluation.py' -v
```

Run commands from `code/business_entity_resolution/`. Modeling tests explicitly skip
when ML dependencies are absent. `OMP_NUM_THREADS=2` limits CPU use during model tests.

## Shared interfaces

Files are UTF-8 TSVs with headers. IDs remain strings, including leading zeros.
Core ML-3 scoring helpers keep IDs opaque. The official source prefixes are
`S1-`, `S2-`, and `S3-`; final outputs also need the organizer validator. Extra source-record columns are
allowed; the other interfaces reject missing or extra columns.

| File | Columns |
| --- | --- |
| Source 1 records | `entity_id`, plus optional record fields |
| Truth or predictions | `source1_entity_id`, `matched_entity_ids` |
| Long-form candidates | `source1_entity_id`, `target_entity_id` |
| ML-2 features | Both pair IDs plus explicitly registered numeric feature columns |
| Scored pairs | Both pair IDs plus `match_probability` |

Match lists are comma-separated IDs; an empty cell denotes an empty set. Whitespace
around IDs is stripped and repeated match-list IDs are deduplicated. Duplicate
source/pair rows, malformed TSV rows, empty IDs, and missing/extra prediction rows
are errors. Header-only candidate/feature files are valid. Empty evaluation
universes are rejected.

**ML-1:** `generate_candidate_pairs()` returns the needed long-form DataFrame.
Save it directly using `pairs_df.to_csv(path, sep="\t", index=False)`.
The shared pipeline exports grouped `candidate_pairs.tsv` with the official
`candidate_entity_ids` column. This differs from the long-form training input.

**ML-2:** provide exactly one feature row per candidate pair, with only the pair
IDs and registered feature columns. Empty numeric cells and `NaN` represent missing
values; infinity and nonnumeric values are rejected. IDs, labels, and prediction
columns cannot be features. Inference aligns physical TSV columns to the saved
ordered schema and rejects missing/extra columns. Feature generation must be
label-free; any learned transformations must be fitted within training folds.
ML-3 cannot detect leakage already embedded in externally supplied features.

## Evaluate first

```bash
python3 -m src baseline --truth dataset/train/train_ground_truth.tsv \
  --output output/all_empty.tsv --report output/all_empty_metrics.json

python3 -m src evaluate --truth dataset/train/train_ground_truth.tsv \
  --predictions output/matching_results.tsv --report output/evaluation.json

python3 -m src oracle --truth dataset/train/train_ground_truth.tsv \
  --candidates output/candidate_pairs_long.tsv \
  --output output/candidate_oracle.tsv --report output/candidate_oracle_metrics.json
```

Per entity, the score is `5*TP / (5*TP + 4*FP + FN)`. Both match sets empty scores 1;
a wrong match on a true singleton scores 0. The final metric averages all Source 1
entity scores equally. The all-empty baseline equals the singleton fraction.
This is not sklearn's macro average across positive/negative pair labels. The
formula also does not imply a fixed monetary or decision cost ratio.

Reports include macro F0.5, pair precision/recall, and singleton false-match rate.
Diagnostic rates with zero denominators are JSON `null`. Optional `--candidates`
on `evaluate` adds blocking/classifier miss counts, zero-candidate counts, and
the candidate oracle score, and requires predictions to stay within that candidate
set. Pair blocking recall is distinct from the maximum attainable macro score.

Evaluation and oracle commands require inputs scoped to the truth IDs. If only
part of Source 1 is labeled, explicitly subset predictions and candidates to that
same universe. Training does that labeled-entity filtering itself and reports
excluded counts. A row absent from truth is unlabeled, not a singleton.

## Train and predict

Before real-data training, verify the official metric, singleton convention, ID
rules, submission schema, and that each labeled row lists **all** true matches.
`--protocol-confirmed` records the caller's verification; it does not perform it.
Use `--synthetic` only for synthetic fixtures.

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m src train \
  --source1 dataset/train/train_source1.tsv \
  --truth dataset/train/train_ground_truth.tsv \
  --candidates output/candidate_pairs_long.tsv \
  --features output/pair_features.tsv \
  --feature-columns token_sort address_ratio \
  --ml2-baseline output/ml2_predictions.tsv \
  --output-dir output/experiment_001 --protocol-confirmed

.venv/bin/python -m src predict \
  --model output/experiment_001/model.pkl \
  --source1 dataset/test/test_source1.tsv \
  --features output/test_pair_features.tsv \
  --output output/matching_results.tsv \
  --scores-output output/scored_pairs.tsv
```

Replace the feature names with ML-2's actual registered columns. The ML-2 baseline
is optional; its absence is explicit in reports. If supplied, it must cover every
labeled entity, have no unknown Source 1 IDs, and use the same candidates. Its
threshold must be fixed independently of holdout labels for a fair comparison.
Experiment output directories must be empty to avoid mixing stale artifacts.

Training follows this fixed procedure:

1. Group Source 1 entities connected through shared true target IDs; keep every
   entity's candidate rows together. Singletons receive their own groups.
2. Reserve 20% of groups (rounded up), seed 42, as holdout. Use three grouped
   development folds for out-of-fold probabilities. Keep zero-candidate entities
   in scoring, and do not downsample negative candidates.
3. Compare histogram-gradient-boosting leaf limits 15/31 and L2 values 0/1. Use
   learning rate 0.05, 200 iterations, minimum leaf size 20, seed 42, and disabled
   internal early stopping.
4. Sweep distinct out-of-fold probability boundaries, plus predict-none, to maximize
   entity-macro F0.5. Accept probabilities `>= threshold`. Higher thresholds win ties;
   predict-none wins tied numeric policies. Equal model scores keep the first grid
   configuration. This is decision-threshold tuning, not probability calibration.
5. Evaluate the selected configuration once on holdout against all-empty, candidate
   oracle, and available ML-2 baselines. Then refit on all labeled candidates and
   retain the development threshold. The final model includes holdout records;
   only saved pre-refit holdout predictions/metrics measure held-out performance.

If there are too few independent groups, or the holdout or any training/validation
fold lacks either class, write the baselines, available split assignments, and an
`insufficient_data` report. No model is created; the command exits with status 2.

Artifacts include `metrics.json`, `splits.tsv`, out-of-fold/holdout scores and
predictions, `all_empty.tsv`, `model.pkl`, and `model_metadata.json`. They record
the feature schema, threshold (`null` means predict-none), protocol declaration,
dependency versions, and input SHA-256 hashes from CLI runs. Use trusted model
pickle files and the recorded sklearn version. Selection and holdout scores are
labeled separately; improved real-data performance is not assumed.

Aggregation includes every Source 1 ID in its original order, sorts unique target
IDs, and leaves the cell empty when no pair passes. Multiple targets are allowed;
there is no one-to-one constraint or transitive expansion.

To aggregate an external scorer's probabilities:

```bash
python3 -m src aggregate --source1 dataset/test/test_source1.tsv \
  --scores output/scored_pairs.tsv --threshold 0.9 --output output/matching_results.tsv
```

Use `--predict-none` instead of `--threshold` for an explicit all-empty policy.
Commands print JSON; evaluation `--report` also saves it. Exit codes: 0 success,
1 input/runtime validation failure, 2 argparse usage error or insufficient training
data (distinguished by the report).

Reusable `src` APIs: `evaluate`, `candidate_oracle`, `empty_predictions`, `aggregate`,
and `tune_threshold`. Match maps contain sets of target IDs; scored pairs use
`src.io_utils.ScoredPair`. Training/inference APIs live in `src.model` so scoring
does not import ML dependencies.

## Synthetic demonstration

```bash
python3 -m src.examples.synthetic_fixture --output-dir output/synthetic_data
OMP_NUM_THREADS=2 .venv/bin/python -m src train \
  --source1 output/synthetic_data/source1.tsv --truth output/synthetic_data/truth.tsv \
  --candidates output/synthetic_data/candidates.tsv \
  --features output/synthetic_data/features.tsv \
  --feature-columns name_similarity address_similarity \
  --ml2-baseline output/synthetic_data/ml2_baseline.tsv \
  --output-dir output/synthetic_experiment --synthetic
```

This fixture includes shared targets, singletons, missing feature values, blocking
misses, zero-candidate entities, and an unlabeled entity. It is solely an integration
exercise. Real-data evaluation and final submission verification require the official
training TSVs, feature validation, the organizer validator, and BE-1's packaging.


## Integration with the shared pipeline

Run from `code/business_entity_resolution/` after installing requirements:

```bash
python src/run_pipeline.py --s1 ../../dataset/test/test_source1.tsv \
  --s2 ../../dataset/test/test_source2.tsv --s3 ../../dataset/test/test_source3.tsv \
  --out ../../output
```

This preserves the repository's existing weighted-similarity baseline by default.
Add `--model output/experiment_001/model.pkl` to use a trained ML-3 artifact. The
artifact must have been trained on the shared feature columns `token_sort` and
`address_ratio`; the synthetic demonstration's feature names are different.
The integration rejects missing/duplicated feature pairs so the exported candidate
set equals the set presented for scoring.

The existing `predict_matches`, `aggregate_to_tsv_format`, and `compute_macro_f05`
interfaces are preserved. The legacy `python src/evaluate.py predictions.tsv
truth.tsv [errors.tsv]` command also remains available and uses the strict evaluator.
The improved ML-1 blocker now uses multi-key generation and bounded bucket
refinement; the old first-15-candidates cap has been removed. Existing ML-2 feature
semantics and the shared validator are preserved. Use the official validator from
the student-resource bundle before submitting real outputs.


## ML-1 normalization and blocking

Legacy `norm_business_name`, `norm_business_address`, and `norm_country` retain
this repository's original normalization behavior. New `feature_business_name`
and `feature_business_address` fields apply Unicode NFKC, lowercase, ampersand-to-
`and` expansion before punctuation cleanup, and whitespace cleanup, while keeping
meaningful words such as `bank`, `hotel`, `services`, and `new`. ML-2 can explicitly
adopt these fields in a later feature revision; existing model features still use
legacy `norm_*` fields and are not silently changed.

Separate `block_business_name` and `block_business_address` fields apply aggressive
noise removal and address abbreviation expansion. Acronyms are extracted from the
conservative name, before aggressive removal, and must share a street number or
postal code. `block_country` canonicalizes known US, India, and France aliases while
preserving other labels. Different countries, including missing versus known
countries, do not share keys. Two records with missing countries may share keys;
missing-country counts and IDs are reported.

`postal_code`, `street_number`, and `street_token` are conservative components
extracted before punctuation is removed. Supported postal patterns are US five-digit
(with optional ZIP+4), France five-digit, and India six-digit. Postal-only addresses
never supply street numbers. House-number ranges, conflicting leading numbers,
non-leading house numbers, and ambiguous long numbers remain unavailable. This is
not a complete international address parser; unsupported countries still receive
name/address blocking. No external identity or geocoding service is used.

The generator unions name stems, name words, exact names, name/address combinations,
location keys, and compatible acronyms. Exact names with at least two alphanumeric
characters cover short/addressless names. Keys are tuples, so token concatenations
cannot collide. The final union has no arbitrary top-N cap.

`BlockingConfig(bucket_limit=500, bucket_policy="refine")` is the default:

- `refine`: split oversized buckets by postal code and, independently, by street
  number plus the first alphabetic street token. Query all applicable secondary
  buckets that meet the same size limit, then union their candidates.
- `drop`: discard oversized buckets, for a controlled comparison.
- `uncapped`: retain all shared-key candidates, for comparisons on manageable data.

There is no truncation to the first 500 targets. An oversized secondary bucket or
missing refinement information leads to an explicitly reported skipped route.
The 500 limit applies to each bucket, not the final candidate union. Total candidate
volume can still be large and must be measured on real data. Different key routes
may recover a pair skipped elsewhere; recall-loss diagnostics account for the union.

Run the standalone, import-safe CLI from `code/business_entity_resolution/`:

```bash
python -m src.blocking --s1 ../../dataset/train/train_source1.tsv \
  --s2 ../../dataset/train/train_source2.tsv \
  --s3 ../../dataset/train/train_source3.tsv \
  --truth ../../dataset/train/train_ground_truth.tsv \
  --bucket-limit 500 --bucket-policy refine --out ../../output/ml1_train
```

The CLI writes:

- `candidate_pairs_long.tsv`: unique pairs accepted directly by ML-3.
- `candidate_pairs.tsv`: official `source1_entity_id,candidate_entity_ids` grouped
  schema, including empty lists and sorted targets in Source 1 input order.
- `blocking_report.json`: configuration, record/pair counts, reduction ratio,
  oversized/refined/skipped bucket counts, affected/recovered/unresolved source IDs,
  missing-country IDs, zero-key/zero-candidate IDs, and min/median/p95/max/mean
  candidate counts. A skipped bucket is an oversized primary with no eligible
  sub-buckets; an unresolved source has at least one oversized route it cannot use.

With optional `--truth`, reports additionally include pair recall and candidate-
oracle macro F0.5 both before and after bucket filtering, and the exact number of
true pairs lost to filtering. These checks use truth-pair key intersections instead
of expanding uncapped candidate products. Missing keys limit recall before filtering;
refinement can only recover matches with an original shared key. Truth is diagnostic
only and cannot change generated candidates. Unlabeled Source 1 rows are excluded
from metrics; a labeled empty match set is a singleton. No-positive-pair recall and
zero-denominator reduction ratios are `null`.

The full `src/run_pipeline.py` command accepts the same `--bucket-limit`,
`--bucket-policy`, and optional `--truth` flags, saves the same diagnostics/long-form
pairs, and verifies that feature rows exactly cover the exported candidates. Omit
`--truth` for test-set inference. IDs must have the correct `S1-`, `S2-`, or `S3-`
prefix; source records are loaded as strings and duplicate/missing IDs are errors.
Empty target files (with headers) are supported. Source files require all four
standard columns, though individual name/address/country cells may be empty.

Reusable APIs are `src.normalize.normalize_dataset`,
`src.blocking.generate_candidate_pairs` (two arguments, DataFrame result),
`generate_candidate_pairs_with_report` (pairs/report tuple), and
`format_and_save_candidates` (official grouped output, including bare filenames).
The former optional `max_cands_per_s1` argument is retired; use `config=BlockingConfig(...)`
for bucket limits instead. Pass raw records or the unchanged output of
`normalize_dataset` to the blocker. Re-normalize after editing raw fields.

Run ML-1 tests independently with:

```bash
python -m unittest discover -s tests -p 'test_ml1.py' -v
```

Synthetic regression tests verify correctness and deliberately include both
recoverable and unrecoverable oversized buckets. The 500 default is an operating
limit, not a measured optimum. Dataset-wide recall/runtime tuning, final official
validation, and real-data trained-model evaluation remain pending supplied data.
