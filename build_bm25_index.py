"""
Build the BM25 sparse index over the new 111,045-chunk corpus using
the `bm25s` library.

Why bm25s instead of rank_bm25:
    rank_bm25's BM25Okapi keeps a Python dict of term frequencies for
    EVERY document in memory as plain Python objects. On this 16GB
    RAM machine, loading that pickle back (in retriever.py, together
    with an already-loaded embedding model) caused a MemoryError.

    bm25s stores everything as compact scipy sparse matrices and
    supports memory-mapped loading (mmap=True), which keeps RAM usage
    far lower — this is exactly the scenario bm25s was built for.

Requires: pip install bm25s --break-system-packages

Run from the project root:
    python build_bm25_index.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import bm25s


CHUNKS_FILE = "data/chunks/pubmed_chunks.csv"
VECTOR_DB_DIR = "vector_db"

# bm25s saves to a DIRECTORY (multiple files), not a single .pkl
BM25S_INDEX_DIR = os.path.join(VECTOR_DB_DIR, "bm25s_index")


def main():

    print("Loading chunks:", CHUNKS_FILE)

    df = pd.read_csv(CHUNKS_FILE)

    chunks = df["chunk"].astype(str).tolist()

    print("Total chunks:", len(chunks))

    print("Tokenizing corpus (stopwords removed for smaller index)...")

    corpus_tokens = bm25s.tokenize(chunks, stopwords="en")

    print("Building BM25 index...")

    retriever = bm25s.BM25()

    retriever.index(corpus_tokens)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    # Do NOT pass corpus= here — we don't need bm25s to store the raw
    # text itself, since row i always maps back to row i of
    # data/chunks/pubmed_chunks.csv (same alignment as the FAISS
    # indices). Skipping this keeps the saved index smaller.
    retriever.save(BM25S_INDEX_DIR)

    print("Saved BM25 index to:", BM25S_INDEX_DIR)
    print("Indexed chunks:", len(chunks))


if __name__ == "__main__":
    main()