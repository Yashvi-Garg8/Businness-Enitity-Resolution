"""Ultra-fast vectorized normalization using native Pandas C-level regex engines."""

import re
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

NORMALIZED_FIELDS = frozenset({
    "feature_business_name", "feature_business_address", "block_business_name",
    "block_business_address", "block_country", "street_number", "postal_code", "street_token",
})

# Precompiled regex patterns
POSTAL_RE = re.compile(r"(?<![\w-])(\d{5,6})(?:-\d{4})?(?![\w-])")
HOUSE_NUMBER_RE = re.compile(r"^\s*(?:(?:house|plot|door|building|no|number)\.?\s*)*[#:]?\s*(\d{1,6}[a-z]?)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def address_components(address: str, country: str):
    """Fallback utility function to maintain ML-1 interface compatibility."""
    if not address or pd.isna(address):
        return "", "", ""
    text = str(address).strip().lower()
    postal_m = POSTAL_RE.search(text)
    postal = postal_m.group(1) if postal_m else ""

    house_m = HOUSE_NUMBER_RE.match(text)
    street_num = house_m.group(1) if house_m else ""
    return postal, street_num, ""


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Country Canonicalization
    if "country" in df.columns:
        c_series = df["country"].fillna("").astype(str).str.lower().str.strip()
        df["norm_country"] = c_series
        df["block_country"] = c_series.replace(COUNTRY_ALIASES)
    else:
        df["norm_country"] = ""
        df["block_country"] = ""

    # 2. Business Name Normalization
    if "business_name" in df.columns:
        names = df["business_name"].fillna("").astype(str).str.lower()
        cleaned = names.str.replace(r"[^\w\s]|_", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        df["norm_business_name"] = cleaned
        df["feature_business_name"] = cleaned
        
        # Strip block noise words
        pat_noise = r"\b(" + "|".join(re.escape(w) for w in BLOCK_NOISE_WORDS) + r")\b"
        df["block_business_name"] = cleaned.str.replace(pat_noise, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        df["norm_business_name"] = ""
        df["feature_business_name"] = ""
        df["block_business_name"] = ""

    # 3. Address Normalization
    if "business_address" in df.columns:
        addrs = df["business_address"].fillna("").astype(str).str.lower()
        cleaned_addr = addrs.str.replace(r"[^\w\s]|_", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        df["norm_business_address"] = cleaned_addr
        df["feature_business_address"] = cleaned_addr

        # Replace abbreviations
        abbr_pat = r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREVIATIONS.keys()) + r")\b"
        df["block_business_address"] = cleaned_addr.str.replace(
            abbr_pat, lambda m: ADDRESS_ABBREVIATIONS.get(m.group(0), m.group(0)), regex=True
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        # 4. Vectorized C-level Regex Extraction for Refinement Keys
        df["postal_code"] = addrs.str.extract(r"(?<![\w-])(\d{5,6})(?:-\d{4})?(?![\w-])", expand=False).fillna("")
        df["street_number"] = addrs.str.extract(r"^\s*(?:(?:house|plot|door|building|no|number)\.?\s*)*[#:]?\s*(\d{1,6}[a-z]?)\b", flags=re.IGNORECASE, expand=False).fillna("")
        df["street_token"] = ""
    else:
        df["norm_business_address"] = ""
        df["feature_business_address"] = ""
        df["block_business_address"] = ""
        df["postal_code"] = ""
        df["street_number"] = ""
        df["street_token"] = ""

    return df