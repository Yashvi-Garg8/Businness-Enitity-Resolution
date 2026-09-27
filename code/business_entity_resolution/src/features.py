import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm

from .normalize import normalize_dataset
from .io_utils import PAIR_COLUMNS, validate_pairs

FEATURE_VERSION = "conservative-v2"

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
    "token_sort",  # Added alias for ML-3 baseline compatibility
]

def _score_chunk(chunk, show_progress=False):
    n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr, h1_arr, h2_arr = chunk

    _ratio = fuzz.ratio
    _tsort = fuzz.token_sort_ratio
    _tset = fuzz.token_set_ratio
    _jw = JaroWinkler.normalized_similarity

    name_ratio, name_token_sort, name_token_set, name_jw = [], [], [], []
    addr_ratio, addr_token_sort, addr_jw = [], [], []
    country_match, street_match = [], []

    iterator = zip(n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr, h1_arr, h2_arr)
    if show_progress:
        iterator = tqdm(iterator, total=len(n1_arr), desc="Scoring candidate pairs")

    for n1, n2, a1, a2, c1, c2, num1, num2 in iterator:
        ts = _tsort(n1, n2) / 100.0 if n1 and n2 else 0.0
        ar = _ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

        name_ratio.append(_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0)
        name_token_sort.append(ts)
        name_token_set.append(_tset(n1, n2) / 100.0 if n1 and n2 else 0.0)
        name_jw.append(_jw(n1, n2) if n1 and n2 else 0.0)

        addr_ratio.append(ar)
        addr_token_sort.append(_tsort(a1, a2) / 100.0 if a1 and a2 else 0.0)
        addr_jw.append(_jw(a1, a2) if a1 and a2 else 0.0)

        country_match.append(int(bool(c1) and bool(c2) and c1 == c2))
        street_match.append(int(bool(num1) and num1 == num2))

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
        "token_sort": name_token_sort,
    }

def compute_pairwise_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    n_jobs: int = 1,
    min_rows_for_parallel: int = 2_000_000,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=list(PAIR_COLUMNS) + FEATURE_COLS)
    return _compute_prepared_pairwise_features(
        pairs_df, normalize_dataset(s1_df), normalize_dataset(tgt_df),
        n_jobs=n_jobs, min_rows_for_parallel=min_rows_for_parallel)


def _compute_prepared_pairwise_features(pairs_df, s1_df, tgt_df, n_jobs=1,
                                        min_rows_for_parallel=2_000_000, show_progress=False):
    """Internal path: callers supply current normalize_dataset output."""
    validate_pairs(pairs_df[list(PAIR_COLUMNS)].itertuples(index=False, name=None))
    if pairs_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"] + FEATURE_COLS)

    fields = ["entity_id", "feature_business_name", "feature_business_address",
              "block_country", "street_number"]
    s1_small = s1_df[fields].set_axis(
        ["source1_entity_id", "n1", "a1", "c1", "h1"], axis=1)
    tgt_small = tgt_df[fields].set_axis(
        ["target_entity_id", "n2", "a2", "c2", "h2"], axis=1)
    merged = pairs_df.merge(s1_small, on="source1_entity_id", how="left",
                           validate="many_to_one", indicator="source_join").merge(
        tgt_small, on="target_entity_id", how="left",
        validate="many_to_one", indicator="target_join")
    if not (merged["source_join"].eq("both").all() and merged["target_join"].eq("both").all()):
        raise ValueError("Candidate pairs reference unknown source or target IDs")

    n1_arr = merged["n1"].to_numpy()
    n2_arr = merged["n2"].to_numpy()
    a1_arr = merged["a1"].to_numpy()
    a2_arr = merged["a2"].to_numpy()
    c1_arr = merged["c1"].to_numpy()
    c2_arr = merged["c2"].to_numpy()

    h1_arr = merged["h1"].to_numpy()
    h2_arr = merged["h2"].to_numpy()

    n = len(merged)

    if n_jobs is None or n_jobs <= 1 or n < min_rows_for_parallel:
        result = _score_chunk((n1_arr, n2_arr, a1_arr, a2_arr, c1_arr, c2_arr, h1_arr, h2_arr), show_progress=show_progress)
    else:
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor

        idx_splits = np.array_split(np.arange(n), n_jobs)
        chunks = [
            (n1_arr[idx], n2_arr[idx], a1_arr[idx], a2_arr[idx], c1_arr[idx], c2_arr[idx], h1_arr[idx], h2_arr[idx])
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
