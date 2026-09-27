import pandas as pd

from features import FEATURE_COLS


def _composite_score(df: pd.DataFrame) -> pd.Series:
    name_sim = (
        0.40 * df["name_jaro_winkler"] +
        0.35 * df["name_token_set"] +
        0.25 * df["name_token_sort"]
    )
    addr_sim = (
        0.50 * df["address_jaro_winkler"] +
        0.50 * df["address_token_sort"]
    )
    street_boost = df["street_number_match"] * 0.08
    return (0.60 * name_sim) + (0.32 * addr_sim) + street_boost


def predict_matches(feature_df: pd.DataFrame, threshold: float = 0.82, require_country_match: bool = True) -> pd.DataFrame:
    """
    Heuristic composite-score baseline. Calibrated with a precision bias for
    macro F0.5. Use tune_threshold.py to pick `threshold` from data instead
    of guessing.
    """
    if feature_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"])

    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        raise KeyError(f"predict_matches() is missing expected feature columns: {missing}")

    df = feature_df.copy()

    # country_match as a HARD filter is a strong precision lever (F0.5
    # weights precision 2x recall) -- but only if norm_country is reliably
    # normalized on both sides (see normalize.COUNTRY_ALIASES). On noisy or
    # inconsistent country labels, a hard filter silently destroys recall
    # with no error message. If you're not confident in country data
    # quality, set require_country_match=False -- country_match still helps
    # as a soft signal inside the score.
    if require_country_match and "country_match" in df.columns:
        df = df[df["country_match"] == 1]
        if df.empty:
            return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"])

    df["score"] = _composite_score(df)
    if not require_country_match and "country_match" in df.columns:
        df["score"] = df["score"] + 0.05 * df["country_match"]

    is_match = df["score"] >= threshold
    return df[is_match][["source1_entity_id", "target_entity_id"]]


def label_pairs(feature_df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
    """Attach a 0/1 label to each candidate pair from train_ground_truth.tsv --
    needed to train a classifier or to grid-search a threshold against the
    real metric instead of guessing."""
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map = {
        row["source1_entity_id"]: set(x.strip() for x in row["matched_entity_ids"].split(",") if x.strip())
        for _, row in gt.iterrows()
    }
    df = feature_df.copy()
    df["label"] = df.apply(
        lambda r: int(r["target_entity_id"] in gt_map.get(r["source1_entity_id"], set())),
        axis=1,
    )
    return df


def train_classifier(labeled_feature_df: pd.DataFrame, random_state: int = 42):
    """A learned combination of all 9 features usually beats hand-tuned
    linear weights, and gets easier to improve as you add more features.
    Logistic regression: small, fast, MIT-licensed, well-calibrated
    probabilities -- easy to threshold for F0.5."""
    from sklearn.linear_model import LogisticRegression

    X = labeled_feature_df[FEATURE_COLS]
    y = labeled_feature_df["label"]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    clf.fit(X, y)
    return clf


def predict_matches_ml(clf, feature_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"])
    df = feature_df.copy()
    df["match_proba"] = clf.predict_proba(df[FEATURE_COLS])[:, 1]
    is_match = df["match_proba"] >= threshold
    return df[is_match][["source1_entity_id", "target_entity_id"]]