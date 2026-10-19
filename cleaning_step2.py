"""
Step 2: Data Cleaning.

Reads data/raw/pubmed_100k_raw.csv and produces a cleaned version:
    - Removes rows with missing/empty title or content
    - Removes duplicate abstracts (same content)
    - Normalizes whitespace
    - Preserves medical symbols/notation (e.g. %, °, Greek letters,
      chemical formulas like Ca2+, mg/dL) — does NOT strip them out
    - Optional lowercasing (OFF by default — embeddings generally
      perform fine on mixed-case text, and keeping case helps
      preserve gene/drug name distinctions like "STAT3" vs "stat3")

Run from the project root:
    python cleaning_step2.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import pandas as pd


# ==========================
# Configuration
# ==========================

INPUT_FILE = "data/raw/pubmed_100k_raw.csv"

OUTPUT_DIR = "data/cleaned"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pubmed_cleaned.csv")

# Set to True if you want lowercased text. Default False to preserve
# case-sensitive medical/gene/drug terminology.
APPLY_LOWERCASE = False

MIN_CONTENT_WORDS = 10  # drop abstracts that are too short to be useful


# ==========================
# Cleaning helpers
# ==========================

def normalize_whitespace(text):

    # Collapse multiple spaces/newlines/tabs into a single space,
    # without touching symbols, punctuation, or unicode characters
    # (so medical notation like "37°C", "5-HT2A", "±0.5 mg/dL" stays
    # intact).
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(text):

    text = normalize_whitespace(text)

    if APPLY_LOWERCASE:
        text = text.lower()

    return text


# ==========================
# Main
# ==========================

def main():

    print("Loading:", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    original_count = len(df)

    print("Original records:", original_count)

    # --- Remove missing title/content ---
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")

    df = df[
        (df["title"].str.strip() != "") &
        (df["content"].str.strip() != "")
    ]

    after_missing_removal = len(df)

    print(
        "After missing-data removal:",
        after_missing_removal,
        f"(removed {original_count - after_missing_removal})"
    )

    # --- Clean text (whitespace normalize, optional lowercase) ---
    print("Cleaning text (normalizing whitespace)...")

    df["title"] = df["title"].apply(clean_text)
    df["content"] = df["content"].apply(clean_text)

    # --- Drop abstracts that are too short to be useful chunks ---
    df = df[df["content"].str.split().str.len() >= MIN_CONTENT_WORDS]

    after_short_removal = len(df)

    print(
        "After short-abstract removal:",
        after_short_removal,
        f"(removed {after_missing_removal - after_short_removal})"
    )

    # --- Remove duplicates (based on content text) ---
    before_dedup = len(df)

    df = df.drop_duplicates(subset=["content"], keep="first")

    after_dedup = len(df)

    print(
        "After duplicate removal:",
        after_dedup,
        f"(removed {before_dedup - after_dedup})"
    )

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("Cleaning Summary")
    print("==============================")
    print("Original records:", original_count)
    print("Final records:   ", len(df))
    print("Total removed:   ", original_count - len(df))
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()