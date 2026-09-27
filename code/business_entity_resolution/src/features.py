import re

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

# NOTE: rapidfuzz does not expose `fuzz.jaro_winkler` -- Jaro-Winkler lives in
# rapidfuzz.distance.JaroWinkler, and `.normalized_similarity()` already returns
# a 0-1 score (no /100 needed), unlike the fuzz.* ratios below.

FEATURE_COLS = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "address_ratio",
    "address_token_sort",
    "address_jaro_winkler",
    "country_match",
    "street_number_match",
]

_STREET_NUMBER_RE = re.compile(r"\b(\d+[a-zA-Z]?)\b")


def extract_street_number(address: str):
    if not address:
        return None
    match = _STREET_NUMBER_RE.search(address)
    return match.group(1) if match else None


def _score_chunk(chunk, show_progress=False):
    """Score one slice of aligned (n1, n2, a1, a2, c1, c2) arrays. Runs in a
    worker process when multiprocessing is enabled, or directly in-process
    otherwise -- must stay a plain top-level function so it can be pickled
    on Windows (spawn-based multiprocessing)."""
    n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr = chunk

    # Local references avoid repeated global/attribute lookups -- a real
    # savings at tens of millions of iterations.
    _ratio = fuzz.ratio
    _tsort = fuzz.token_sort_ratio
    _tset = fuzz.token_set_ratio
    _jw = JaroWinkler.normalized_similarity
    _extract_num = extract_street_number

    name_ratio, name_token_sort, name_token_set, name_jw = [], [], [], []
    addr_ratio, addr_token_sort, addr_jw = [], [], []
    country_match, street_match = [], []

    iterator = zip(n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr)
    if show_progress:
        iterator = tqdm(iterator, total=len(n1_arr), desc="Scoring candidate pairs")

    for n1, n2, a1, a2, c1, c2 in iterator:
        name_ratio.append(_ratio(n1, n2) / 100.0)
        name_token_sort.append(_tsort(n1, n2) / 100.0)
        name_token_set.append(_tset(n1, n2) / 100.0)
        name_jw.append(_jw(n1, n2) if n1 and n2 else 0.0)

        addr_ratio.append(_ratio(a1, a2) / 100.0)
        addr_token_sort.append(_tsort(a1, a2) / 100.0)
        addr_jw.append(_jw(a1, a2) if a1 and a2 else 0.0)

        country_match.append(int(bool(c1) and bool(c2) and c1 == c2))
        num1, num2 = _extract_num(a1), _extract_num(a2)
        street_match.append(int(num1 is not None and num1 == num2))

    return {
        "name_ratio": name_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jaro_winkler": name_jw,
        "address_ratio": addr_ratio,
        "address_token_sort": addr_token_sort,
        "address_jaro_winkler": addr_jw,
        "country_match": country_match,
        "street_number_match": street_match,
    }


def compute_pairwise_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    n_jobs: int = 1,
    min_rows_for_parallel: int = 2_000_000,
) -> pd.DataFrame:
    """Score every (source1_entity_id, target_entity_id) candidate pair.

    Performance notes (this matters at multi-million-row scale):
      - Uses a vectorized pandas merge to attach name/address/country to each
        pair instead of per-row dict.get() lookups -- the previous version's
        main avoidable overhead.
      - Set n_jobs > 1 to split the scoring loop across worker processes
        (CPU-bound, so this scales close to linearly with cores). Only
        kicks in above min_rows_for_parallel rows, since process startup
        isn't worth it on small candidate sets. On Windows this requires
        being called from inside `if __name__ == "__main__":` (run_pipeline.py
        already does this).

    Expects s1_df / tgt_df to already carry norm_business_name,
    norm_business_address, and norm_country columns.
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"] + FEATURE_COLS)

    s1_small = s1_df[["entity_id", "norm_business_name", "norm_business_address", "norm_country"]].rename(
        columns={
            "entity_id": "source1_entity_id",
            "norm_business_name": "n1",
            "norm_business_address": "a1",
            "norm_country": "c1",
        }
    )
    tgt_small = tgt_df[["entity_id", "norm_business_name", "norm_business_address", "norm_country"]].rename(
        columns={
            "entity_id": "target_entity_id",
            "norm_business_name": "n2",
            "norm_business_address": "a2",
            "norm_country": "c2",
        }
    )

    merged = pairs_df.merge(s1_small, on="source1_entity_id", how="left").merge(
        tgt_small, on="target_entity_id", how="left"
    )
    for col in ("n1", "a1", "c1", "n2", "a2", "c2"):
        merged[col] = merged[col].fillna("")

    n1_arr = merged["n1"].to_numpy()
    n2_arr = merged["n2"].to_numpy()
    a1_arr = merged["a1"].to_numpy()
    a2_arr = merged["a2"].to_numpy()
    c1_arr = merged["c1"].to_numpy()
    c2_arr = merged["c2"].to_numpy()

    n = len(merged)

    if n_jobs is None or n_jobs <= 1 or n < min_rows_for_parallel:
        result = _score_chunk((n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr), show_progress=True)
    else:
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor

        idx_splits = np.array_split(np.arange(n), n_jobs)
        chunks = [
            (n1_arr[idx], n2_arr[idx], a1_arr[idx], a2_arr[idx], c1_arr[idx], c2_arr[idx])
            for idx in idx_splits
        ]
        result = {k: [] for k in FEATURE_COLS}
        print(f"      Scoring {n} pairs across {n_jobs} worker processes...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for chunk_result in tqdm(ex.map(_score_chunk, chunks), total=len(chunks), desc="Chunks"):
                for k in FEATURE_COLS:
                    result[k].extend(chunk_result[k])

    out = {
        "source1_entity_id": merged["source1_entity_id"].to_numpy(),
        "target_entity_id": merged["target_entity_id"].to_numpy(),
    }
    out.update(result)
    return pd.DataFrame(out)


def sanity_check_features_against_ground_truth(
    features_df: pd.DataFrame,
    ground_truth_path: str,
    feature_cols=None,
) -> pd.DataFrame:
    """Load train_ground_truth.tsv, label each scored pair as a true match or
    a look-alike, and compare each feature's mean between the two groups."""
    feature_cols = feature_cols or FEATURE_COLS

    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str)
    gt["matched_ids"] = gt["matched_entity_ids"].fillna("").apply(
        lambda x: set(x.split(",")) if x else set()
    )
    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_ids"]))

    labeled = features_df.copy()
    labeled["is_true_match"] = labeled.apply(
        lambda r: int(r["target_entity_id"] in gt_map.get(r["source1_entity_id"], set())),
        axis=1,
    )

    summary_rows = []
    print("\n=== Feature sanity check vs. ground truth ===")
    print(f"{'feature':<24} {'mean(match)':>12} {'mean(non-match)':>16} {'separation':>12}")
    for col in feature_cols:
        if col not in labeled.columns:
            continue
        mean_match = labeled.loc[labeled["is_true_match"] == 1, col].mean()
        mean_non_match = labeled.loc[labeled["is_true_match"] == 0, col].mean()
        separation = mean_match - mean_non_match
        flag = "" if separation > 0.05 else "  <-- WEAK, consider dropping/reworking"
        print(f"{col:<24} {mean_match:>12.3f} {mean_non_match:>16.3f} {separation:>12.3f}{flag}")
        summary_rows.append({
            "feature": col,
            "mean_true_match": mean_match,
            "mean_look_alike": mean_non_match,
            "separation": separation,
        })

    return pd.DataFrame(summary_rows)