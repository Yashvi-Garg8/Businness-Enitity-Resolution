"""Separate backward-compatible, conservative, and blocking representations."""

import re
import unicodedata

import pandas as pd

LEGAL_SUFFIX_WORDS = frozenset({
    "pvt", "private", "ltd", "limited", "corp", "corporation", "inc",
    "incorporated", "llc", "co", "company",
})
BLOCK_NOISE_WORDS = LEGAL_SUFFIX_WORDS | {
    "group", "holdings", "services", "solutions", "enterprises", "traders", "agency",
    "the", "a", "an", "of", "and", "in", "at", "on", "for", "new", "shree", "shri",
    "om", "dr", "near", "opp", "opposite", "shop", "no", "number", "hotel", "bank", "atm",
}
ADDRESS_NOISE_WORDS = {
    "the", "a", "an", "of", "and", "in", "at", "on", "for", "new", "shree", "shri",
    "om", "dr", "near", "opp", "opposite", "shop", "no", "number", "hotel", "bank", "atm",
}
ADDRESS_ABBREVIATIONS = {"st": "street", "rd": "road", "ave": "avenue"}
COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us",
    "united states": "us", "united states of america": "us",
    "india": "india", "in": "india", "ind": "india",
    "france": "france", "fr": "france", "fra": "france",
}
POSTAL_PATTERNS = {
    "us": re.compile(r"(?<![\w-])(\d{5})(?:-\d{4})?(?![\w-])"),
    "france": re.compile(r"(?<![\w-])(\d{5})(?![\w-])"),
    "india": re.compile(r"(?<![\w-])(\d{6})(?![\w-])"),
}
HOUSE_NUMBER = re.compile(
    r"^\s*(?:(?:(?:house|plot|door|building)\s+(?:(?:no|number)\.?\s*)?)"
    r"|(?:(?:no|number)\.?\s*))?[#:]?\s*(\d{1,6}[a-z]?)\b", re.IGNORECASE)
NORMALIZED_FIELDS = frozenset({
    "feature_business_name", "feature_business_address", "block_business_name",
    "block_business_address", "block_country", "street_number", "postal_code", "street_token",
})


def clean_text(text: str) -> str:
    """Legacy normalization: retained unchanged for existing norm_* consumers."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unicode_text(text):
    return "" if pd.isna(text) else unicodedata.normalize("NFKC", str(text)).lower()


def conservative_text(text):
    text = unicode_text(text).replace("&", " and ")
    # Underscores are punctuation too; preserve Unicode letters and accents.
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def canonical_country(text):
    normalized = conservative_text(text)
    return COUNTRY_ALIASES.get(normalized, normalized)


def address_components(address, country):
    """Return postal, unambiguous leading house number, and following street token.

    This is intentionally conservative, not an international address parser.
    Unsupported countries retain name/address blocking but no guessed postal code.
    """
    text = unicode_text(address).strip()
    pattern = POSTAL_PATTERNS.get(country)
    postal_matches = list(pattern.finditer(text)) if pattern else []
    postal_values = {match.group(1) for match in postal_matches}
    postal = next(iter(postal_values)) if len(postal_values) == 1 else ""
    house = HOUSE_NUMBER.match(text)
    street_number = street_token = ""
    if house:
        # Never reuse a recognized postal span as a house number, even if ambiguous.
        overlaps_postal = any(match.start() <= house.start(1) < match.end() for match in postal_matches)
        rest = text[house.end():]
        ambiguous = bool(re.match(r"\s*(?:[-/–—]|\d)", rest))
        # For unknown countries, long leading numbers could be postal codes.
        long_unknown = country not in POSTAL_PATTERNS and len(house.group(1)) >= 5
        if not overlaps_postal and not ambiguous and not long_unknown:
            street_number = house.group(1)
            tokens = conservative_text(rest).split()
            street_token = next((token for token in tokens if token.isalpha()), "")
            street_token = ADDRESS_ABBREVIATIONS.get(street_token, street_token)
    return postal, street_number, street_token


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return new columns on a copy without changing raw or legacy normalized data."""
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[f"norm_{col}"] = df[col].apply(clean_text)
        else:
            df[f"norm_{col}"] = ""
    for col in ("business_name", "business_address"):
        df[f"feature_{col}"] = df[col].apply(conservative_text) if col in df else ""
    df["block_country"] = df["country"].apply(canonical_country) if "country" in df else ""
    df["block_business_name"] = df["feature_business_name"].apply(
        lambda value: " ".join(word for word in value.split() if word not in BLOCK_NOISE_WORDS))
    df["block_business_address"] = df["feature_business_address"].apply(
        lambda value: " ".join(ADDRESS_ABBREVIATIONS.get(word, word) for word in value.split()
                              if word not in ADDRESS_NOISE_WORDS))
    addresses = df["business_address"] if "business_address" in df else [""] * len(df)
    components = [address_components(address, country) for address, country in zip(addresses, df["block_country"])]
    for index, col in enumerate(("postal_code", "street_number", "street_token")):
        df[col] = [values[index] for values in components]
    return df
