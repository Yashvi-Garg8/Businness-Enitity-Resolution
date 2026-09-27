# Baseline pipeline integration handoff

## Status

The baseline pipeline passed 45 targeted tests and an approved real-data pilot.
The subsequent full-target check stopped at the approved 4 GiB memory limit after
2.672 million of 10.320 million target rows (25.9%). Index finalization and scoring
did not run. Software integration passed the small pilot, but the current target
index does not fit the approved full-target memory budget. Full-dataset matching
quality and completion time remain unverified.
Changes are local and uncommitted on `integration/full-pipeline`, based on `b63df3e`.
No training, threshold tuning, dependency changes, commit, push, or changes to
`experimental` were performed for this pilot.

## What is integrated

- Source TSV ingestion, conservative normalization, blocking, ML-2 similarity
  features, baseline matching, aggregation, and entity-level evaluation.
- Source batches default to 1,000 and pair batches to 50,000. Targets are normalized
  once and use one global blocking index. Batch sizes change memory use, not results.
- Strict input/ID/truth checks, disk-backed validation and evaluation, deterministic
  TSVs, and atomic publication into a fresh/empty output directory.
- Blocking defaults: limit 500, policy `refine`, no per-entity candidate cap.
  Baseline: `0.65 * token_sort + 0.35 * address_ratio >= 0.86`.
- Prepared features use `conservative-v2`. Old model artifacts need retraining;
  this handoff's validated path uses the baseline without `--model`.

## Real-data pilot evidence

Selection: first 500 training-truth rows, their 1,723 distinct true targets, and
10,000 background rows from each target source. Source input order was preserved.
Total selected targets: 21,723. This deliberately reduced target universe changes
negative examples and bucket sizes; scores below are pilot observations only.

| Measure | Observed |
| --- | ---: |
| Source 1 output coverage | 500 / 500 |
| Candidate pairs | 9,592 |
| Predicted matches / true positives | 713 / 713 |
| False positives | 0 |
| Macro F0.5 | 0.636209 |
| Pair precision / recall | 1.000000 / 0.413813 |
| Blocking pair recall | 0.820662 |
| Candidate-oracle macro F0.5 | 0.915628 |
| Blocking misses / baseline matcher misses | 309 / 701 |
| True pairs lost to bucket filtering | 0 |
| Singleton false matches | 0 / 24 |
| Entities with zero candidates | 13 |
| Source scan and selection | 32.23 seconds |
| Selected-data pipeline | 0.73 seconds |
| Total scan, pilot, and evaluator check | 32.98 seconds |
| Process peak RSS | 198.4 MiB |
| Scratch/output footprint | 2.6 MiB |

Both evaluators returned the same score. Their 428 error rows have identical
content; row order differs because the pipeline follows Source 1 order and the
standalone evaluator follows truth order. Output coverage, grouped/long candidate
agreement, valid IDs, uniqueness, and match-subset constraints passed validation.
Zero false positives in this sample does not establish full-data precision.
The 309 blocking misses and 701 baseline rejections remain modeling-quality work,
not runtime integration failures. The threshold was not tuned.

The approved source scan counted 2,206,821 Source 1 rows, 5,034,616 Source 2 rows,
and 5,285,603 Source 3 rows. It validated source schemas and ID prefixes while
selecting pilot rows, but did not establish uniqueness of every unselected ID or
validate the complete truth file. The full 10,320,219-target index was not built.
Do not extrapolate pilot RSS/time linearly to that index.

## Full-target feasibility result

The approved check attempted to ingest both complete target files, then score only
the same 500 pilot entities. Limits were 4 GiB process RSS, 15 minutes, and 512 MiB
scratch/output on a 16 GiB Mac. It stopped during target ingestion after about
102 seconds, with 2,672,000 target records added and peak RSS of 4,367,040,512 bytes
(4.07 GiB). The sampled guard detected the breach and exited with status 124;
it is not a kernel hard memory cap, so a small overshoot occurred.

The check did not finish indexing Source 2, begin indexing Source 3, finalize the index, or score the
pilot against all targets. No completed output directory was published. Scratch
usage at the stop was about 71 KiB; the time and disk limits were not reached.
This demonstrates failure under the approved 4 GiB budget, not the exact memory
required to finish. Do not infer feasibility on a larger machine from this result.
See `full_target_check.json` in the transfer ZIP for the measured evidence.

## Teammate commands and output contract

Use the existing pinned requirements with Python 3.9–3.12 (tested here on 3.9.6).
Run from the repository root. The commands below assume the teammate's data is at
`dataset/train`; replace that prefix with the actual absolute path when needed.
On this machine it is `/Users/amulyakundalia/Desktop/dataset/train`.
Use `python3` if that is the configured interpreter.

Before executing these full-data commands, address the target-index memory failure,
validate the new index implementation, and obtain the required heavy-usage approval.

```bash
python code/business_entity_resolution/src/run_pipeline.py --s1 dataset/train/train_source1.tsv --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv --truth dataset/train/train_ground_truth.tsv --out output/baseline-handoff --source-batch-size 1000 --pair-batch-size 50000
python code/business_entity_resolution/src/evaluate.py output/baseline-handoff/matching_results.tsv dataset/train/train_ground_truth.tsv output/baseline-handoff/errors-standalone.tsv
```

The standalone evaluator requires identical prediction/truth entity coverage.
For partially labeled data, use `evaluation.json` and `errors.tsv` from the pipeline;
these exclude and count unlabeled entities. Do not infer negatives from missing truth.

- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`.
- `candidate_pairs_long.tsv`: `source1_entity_id`, `target_entity_id`.
- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`.
- Match lists are sorted unique comma-separated IDs; unmatched entities have empty cells.
- `blocking_report.json` records blocking losses and candidate counts.
- `evaluation.json` and `errors.tsv` are produced when `--truth` is supplied.
- `run_manifest.json` records `status=complete`, configuration, counts, timings,
  process peak RSS and input size/mtime metadata. Only completed runs are published.
- Existing nonempty output directories are rejected; use a new run directory.

## Remaining gate and transfer

The target index and each source batch's candidate unions remain memory-resident.
Recommended next change: disk-backed target records and blocking indexes, with
bounded queries, global bucket counts, unchanged refinement semantics, and exact
candidate-equivalence tests against the current in-memory reference. This change
has not been implemented or benchmarked. Increasing source/pair batch limits will
not solve global target-index memory growth. Do not silently drop candidates.
After that change, repeat the approved full-target check and 500-entity validation;
full inference still requires a separate resource gate. No trained-model quality
or submission readiness is claimed by this handoff.

The transfer ZIP contains `baseline-integration.patch`, this handoff, and
`pilot_summary.json`, and `full_target_check.json`. It contains no dataset, raw
pilot records, or experimental code.
The patch includes the preceding normalization/integration fixes and the new batch
pipeline, tests, and documentation. Apply it to a clean compatible checkout based
on `b63df3e` (or resolve differences with the teammate's branch):

```bash
git apply --check baseline-integration.patch
git apply baseline-integration.patch
```

Do not apply the patch again to the working tree where these changes already exist.
The source files and test fixtures remain the authoritative implementation.
