import pandas as pd
import re
from collections import defaultdict
import gc

STOPWORDS = {
    "the", "and", "inc", "ltd", "pvt", "llc", "corp", "co", "store",
    "shop", "hotel", "near", "shree", "shri", "dr", "om", "new", "ms",
    "enterprises", "company", "services", "solutions", "traders"
}


def extract_tokens(text: str) -> list:
    if not isinstance(text, str):
        return []
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def extract_address_prefix(address: str, n_chars: int = 6) -> str:
    """A third blocking key based on location instead of name -- catches true
    matches whose names diverge too much for token blocking (DBA names,
    transliterated names) but that share the same address."""
    if not isinstance(address, str):
        return ""
    alnum = re.sub(r"[^a-z0-9]", "", address.lower())
    return alnum[:n_chars]


def _prune_dense_keys(index: dict, cap: int) -> dict:
    """Truncate oversized buckets instead of discarding them outright.
    Dropping a key entirely zeroes out candidates for every Source 1 entity
    that hashes to it -- a silent, hard recall loss. Truncating keeps a
    deterministic subset instead of nothing."""
    return {k: (v[:cap] if len(v) > cap else v) for k, v in index.items()}


def generate_candidate_pairs(s1_df: pd.DataFrame, target_df: pd.DataFrame, max_cands_per_s1: int = 20) -> pd.DataFrame:
    print("      Building inverted index from target dataset...")

    index_k1 = defaultdict(list)  # country + first significant name token
    index_k2 = defaultdict(list)  # country + longest name token
    index_k3 = defaultdict(list)  # country + address prefix (NEW)

    tgt_ids = target_df["entity_id"].values
    tgt_names = target_df["norm_business_name"].fillna("").values
    tgt_addrs = target_df["norm_business_address"].fillna("").values
    tgt_countries = target_df["norm_country"].fillna("").values

    for tid, name, addr, country in zip(tgt_ids, tgt_names, tgt_addrs, tgt_countries):
        c = str(country)

        tokens = extract_tokens(name)
        if tokens:
            k1 = f"{c}::{tokens[0]}"
            index_k1[k1].append(tid)
            k2 = f"{c}::{max(tokens, key=len)}"
            if k2 != k1:
                index_k2[k2].append(tid)

        addr_prefix = extract_address_prefix(addr)
        if addr_prefix:
            k3 = f"{c}::{addr_prefix}"
            index_k3[k3].append(tid)

    print("      Pruning oversized buckets (truncate, not drop)...")
    index_k1 = _prune_dense_keys(index_k1, cap=300)
    index_k2 = _prune_dense_keys(index_k2, cap=200)
    index_k3 = _prune_dense_keys(index_k3, cap=150)

    print("      Querying candidates for Source 1 records...")
    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["norm_business_name"].fillna("").values
    s1_addrs = s1_df["norm_business_address"].fillna("").values
    s1_countries = s1_df["norm_country"].fillna("").values

    out_s1 = []
    out_tgt = []

    for sid, name, addr, country in zip(s1_ids, s1_names, s1_addrs, s1_countries):
        c = str(country)
        tokens = extract_tokens(name)
        addr_prefix = extract_address_prefix(addr)

        seen_targets = set()

        if tokens:
            k1 = f"{c}::{tokens[0]}"
            for tid in index_k1.get(k1, []):
                seen_targets.add(tid)
                if len(seen_targets) >= max_cands_per_s1:
                    break

            if len(seen_targets) < max_cands_per_s1:
                k2 = f"{c}::{max(tokens, key=len)}"
                for tid in index_k2.get(k2, []):
                    seen_targets.add(tid)
                    if len(seen_targets) >= max_cands_per_s1:
                        break

        if len(seen_targets) < max_cands_per_s1 and addr_prefix:
            k3 = f"{c}::{addr_prefix}"
            for tid in index_k3.get(k3, []):
                seen_targets.add(tid)
                if len(seen_targets) >= max_cands_per_s1:
                    break

        for tid in seen_targets:
            out_s1.append(sid)
            out_tgt.append(tid)

    del index_k1, index_k2, index_k3
    gc.collect()

    print(f"      Constructing candidate pair DataFrame ({len(out_s1)} pairs)...")
    return pd.DataFrame({"source1_entity_id": out_s1, "target_entity_id": out_tgt})


def check_blocking_recall(candidate_pairs_df: pd.DataFrame, ground_truth_path: str) -> float:
    """Diagnostic: what fraction of TRUE matches survived blocking? This is
    the recall ceiling for the whole pipeline -- no matcher can recover a
    match that never became a candidate. Run this on a held-out split before
    spending time tuning thresholds or features."""
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    cand_by_s1 = candidate_pairs_df.groupby("source1_entity_id")["target_entity_id"].apply(set).to_dict()

    hits, total = 0, 0
    for _, row in gt.iterrows():
        true_ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        for tid in true_ids:
            total += 1
            if tid in cand_by_s1.get(row["source1_entity_id"], set()):
                hits += 1

    recall = hits / total if total else 1.0
    avg_cands = candidate_pairs_df.groupby("source1_entity_id").size().mean() if not candidate_pairs_df.empty else 0.0
    print(f"      [blocking] recall ceiling: {recall:.4f} ({hits}/{total} true matches survived)")
    print(f"      [blocking] avg candidates per Source 1 entity: {avg_cands:.1f}")
    return recall