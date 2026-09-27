# ML-1 review and integration handoff

> Scope: these findings refer to the original teammate files received locally,
> not the newer ML-1 implementation in this GitHub repository. In the current
> shared blocker, dense buckets use a limit of 300 and candidates are truncated
> to 15 per Source 1 entity. Neither behavior is changed by this ML-3 branch.


ML-3 leaves the teammate's modules unchanged. These are findings from the received
files, with synthetic checks where indicated. Real-data recall impact is unmeasured.

| Priority | Location | Finding and suggested action |
| --- | --- | --- |
| High | `blocking.py:94–98` | Buckets over 500 records are silently deleted. Configure the limit, report affected records, and measure recall before/after pruning. Consider secondary keys within large buckets. |
| High | `blocking.py:56–60` | Acronym keys are generated only for multiword names. Emit compatible expanded-name and single-token acronym keys. Normalization also removes acronym-bearing words such as `bank` and `of`. |
| High | `blocking.py:23–79` | All keys require identical country strings. `us` and `united states` cannot share keys even with identical remaining fields. Canonicalize aliases and measure missing-country cases. |
| High | `normalize.py:4–7,23–32` | Removal of tokens such as `bank`, `hotel`, `services`, and `new` may erase distinctions. Preserve conservative normalized fields for ML-2; separate aggressive blocking variants. |
| Medium | `blocking.py:14–21` | The first standalone number is assumed to be a street number. A postal-only address such as `110001` becomes its street number. Distinguish postal/street components and uncertainty. |
| Medium | `blocking.py:23–79` | Identical short one-word names without addresses can yield zero keys. Report zero-candidate entities and evaluate targeted fallback keys. |
| Medium | `blocking.py:112–114,126–128` | Set traversal makes output ordering nondeterministic. Sort IDs before writing. |
| Medium | `blocking.py:123` | `os.makedirs('')` fails for a bare filename. Use `Path(output_path).parent.mkdir(...)`. |
| Medium | `test_ml1.py:11–43` | Importing the file reads/writes data. Add a guarded CLI and explicit paths; strip/deduplicate truth IDs before recall calculation. |
| Low | `normalize.py:5,12` | Punctuation stripping removes `&` before expansion; the word-boundary pattern also misses an ordinary spaced ampersand. Expand meaningful symbols before removing punctuation. |

Isolated helper checks reproduced zero shared keys for expanded-name/acronym pairs
without a shared postal key, country alias variants, and short addressless names.
They also reproduced postal-as-street extraction and the bare-filename failure.
These examples do not establish the frequency or impact on challenge data.

The received README referred to a missing dependency file and had an unfinished
code fence. ML-3 adds working dependencies and setup documentation for BE-1 to
incorporate. Final dependency integration and packaging remain with BE-1.

## Contracts to preserve

- Send ML-3 long-form pairs from `generate_candidate_pairs()` with columns
  `source1_entity_id,target_entity_id`. The grouped `format_and_save_candidates()`
  file has a different schema.
- Keep IDs as strings at ingestion; numeric inference may lose leading zeros.
- ML-2 must provide exactly one numeric feature row per candidate, with explicit
  feature names. IDs and labels must stay out of the model's feature matrix.
- Pair blocking recall is not the numerical macro-F0.5 ceiling. ML-3's candidate
  oracle measures the attainable macro score including zero-candidate records.
- Missing truth rows are unlabeled. Only an explicitly empty list on a completely
  labeled entity establishes a singleton.
- The F0.5 denominator weights FP and FN by 1 and 0.25, not a fixed two-to-one
  decision cost. Tune the implemented metric directly.

BE-1 still owns official prefix/schema validation, the full pipeline orchestrator,
error-dump tooling, full-scale performance profiling, and submission packaging.
