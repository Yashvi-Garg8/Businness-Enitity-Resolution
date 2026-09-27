# ML-1 implementation handoff

The agreed ML-1 fixes are implemented locally in this branch. Synthetic checks and
a reduced-universe real-data pilot passed. See [PIPELINE_HANDOFF.md](PIPELINE_HANDOFF.md)
for measured pilot results, teammate commands, and the full-target memory failure
(4 GiB exceeded during ingestion). A disk-backed index is the recommended follow-up.
Pilot scores do not establish full-dataset recall or runtime.

## Implemented fixes

| Original issue | Current behavior |
| --- | --- |
| Shared blocker dropped buckets over 300 and kept only the first 15 candidates | Configurable per-bucket limit (default 500); `refine`, `drop`, and `uncapped` policies; no arbitrary top-N truncation |
| Original received blocker silently dropped buckets over 500 | Refines by postal code or street number plus street token; reports all skipped routes and labeled-match losses |
| Acronyms were incompatible across expanded and single-token names | Builds compatible variants from conservative names, including connector-word variants, with shared location evidence |
| Country aliases failed to meet; France is unseen during training | Canonicalizes supported aliases, preserves arbitrary labels, and reports missing countries |
| Aggressive normalization could erase modeling signals | Adds conservative `feature_*` and separate `block_*` fields; preserves the shared repository's legacy `norm_*` semantics |
| Postal codes were used as house numbers | Separates postal spans; leaves ambiguous/ranged/non-leading house numbers unavailable |
| Short addressless names produced no keys | Exact conservative-name key within country for names with at least two alphanumeric characters, using normal bucket limits |
| Output was nondeterministic or failed for a bare filename | Source-order rows, sorted targets, unique pairs, safe directory creation, and official candidate header |
| Manual recall script read/wrote files on import | Import-safe `python -m src.blocking` CLI with explicit input/output paths |
| Recall was presented as the macro-score ceiling | Reports pair recall and candidate-oracle macro F0.5 separately, before and after bucket filtering |

The original received files and the GitHub baseline were different implementations.
Legacy compatibility here refers to the shared GitHub repository's norm_* behavior;
the separately received files in the parent workspace remain a historical reference.

## Interfaces and ownership

- Source records require `entity_id`, `business_name`, `business_address`, and
  `country`. The loader preserves strings and leading zeros and rejects duplicate,
  missing, or incorrectly prefixed IDs. Empty field values are allowed.
- `generate_candidate_pairs(s1_df, target_df, config=...)` returns long-form pairs.
  `generate_candidate_pairs_with_report(..., truth=...)` returns pairs plus a JSON-
  ready report. Truth is used only after generation for diagnostics.
- The old third positional `max_cands_per_s1` parameter is removed. Use
  `BlockingConfig(bucket_limit=500, bucket_policy="refine")` instead.
- `candidate_pairs_long.tsv` uses `source1_entity_id,target_entity_id` internally.
  Official `candidate_pairs.tsv` uses `source1_entity_id,candidate_entity_ids`.
- Final candidates equal the pairs presented for feature extraction/scoring. The
  integrated pipeline rejects missing/duplicate feature pairs. Final matches must
  remain a subset; both grouped outputs cover every Source 1 entity.
- ML-2 uses conservative `feature_*` text, canonical `block_country`, and validated
  `street_number`. This is feature version `conservative-v2`; existing models must
  be retrained. Legacy `norm_*` fields remain available for other consumers.
- BE-1 retains official validator integration, full-scale performance tuning,
  submission history, final packaging, and methodology-template ownership.

## Diagnostics and remaining checks

`blocking_report.json` records candidate counts/distribution, reduction ratio,
oversized/refined/skipped buckets, and affected/recovered/unresolved source IDs.
It also records missing countries, sources with no keys, and sources with no
candidates. A source may use one refined route while another remains unresolved.

With training truth, compare before/after-filtering pair recall and candidate-oracle
macro F0.5. Their difference isolates bucket-filtering losses; matches lacking any
shared key are already absent before filtering. Ground-truth rows missing from a
partially labeled dataset are excluded from evaluation, not labeled negative.

On real data, run `drop`, `refine`, and `uncapped` where feasible, measuring recall,
oracle score, candidate volume, and runtime. The union can exceed 500 candidates per
source even though individual buckets are bounded. The current extractor is
conservative and may omit valid international or reordered house numbers; assess
those misses before broadening parsing/fallbacks. No external identity/geocoding
lookup has been introduced.

Run the organizer's validator before submission. The final-model license question
(MIT/Apache-2.0 requirement versus the current sklearn learner) remains separate
from these ML-1 fixes. See README.md for commands and the full data contracts.
