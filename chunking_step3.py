"""
Step 3: Document Chunking.

Reads data/cleaned/pubmed_cleaned.csv and splits each abstract into
overlapping chunks suitable for embedding + retrieval.

Design choices:
    - CHUNK_SIZE_WORDS = 220   (medium size — balances context
      preservation vs retrieval granularity; standard choice in RAG
      literature for abstract-length documents)
    - CHUNK_OVERLAP_WORDS = 40 (prevents losing information that
      falls right on a chunk boundary)
    - Each chunk is prefixed with the paper's title, since the title
      often carries key context (e.g. the specific condition/drug
      being studied) that individual chunks from the middle of an
      abstract may not repeat.
    - MIN_CHUNK_WORDS filter drops tiny leftover chunks (e.g. the
      last sliver of a short abstract) that add noise without value.

Run from the project root:
    python chunking_step3.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd


# ==========================
# Configuration
# ==========================

INPUT_FILE = "data/cleaned/pubmed_cleaned.csv"

OUTPUT_DIR = "data/chunks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pubmed_chunks.csv")

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 20


# ==========================
# Chunking logic
# ==========================

def chunk_text(text, chunk_size, overlap):

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []

    step = chunk_size - overlap

    start = 0

    while start < len(words):

        chunk_words = words[start:start + chunk_size]

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

        start += step

    return chunks


# ==========================
# Main
# ==========================

def main():

    print("Loading:", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("Input documents:", len(df))

    rows = []

    chunk_id = 0

    for _, row in df.iterrows():

        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        pmid = row.get("PMID", "")
        doc_id = row.get("id", "")

        raw_chunks = chunk_text(content, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

        for piece in raw_chunks:

            # Prefix with title so each chunk carries paper-level
            # context, not just the local sentence window.
            full_chunk = f"{title}. {piece}" if title else piece

            if len(full_chunk.split()) < MIN_CHUNK_WORDS:
                continue

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "PMID": pmid,
                    "title": title,
                    "chunk": full_chunk,
                }
            )

            chunk_id += 1

    chunks_df = pd.DataFrame(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunks_df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("Chunking Summary")
    print("==============================")
    print("Input documents:", len(df))
    print("Output chunks:  ", len(chunks_df))
    print(
        "Avg chunks/doc: ",
        round(len(chunks_df) / len(df), 2) if len(df) else 0
    )
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()