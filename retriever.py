"""
Retriever: model-selectable dense retrieval, sparse (BM25) retrieval,
and hybrid retrieval with Reciprocal Rank Fusion (RRF).

Set ACTIVE_MODEL below to any model_key that has completed embedding
(check by running vector_db.py). Set RETRIEVAL_MODE to control the
default strategy used by retrieve_context().

Row alignment: row i of data/chunks/pubmed_chunks.csv corresponds to
vector i in every model's FAISS index (embeddings were generated in
that exact row order in embedding_step4.py), and to document i in the
BM25 index (same row order in build_bm25_index.py). So a single
integer position looks up consistently across all three.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import faiss
import bm25s

from sentence_transformers import SentenceTransformer


# ==========================
# Configuration
# ==========================

CHUNKS_FILE = "data/chunks/pubmed_chunks.csv"
VECTOR_DB_DIR = "vector_db"

# bm25s saves/loads a DIRECTORY (not a single .pkl like the old
# rank_bm25-based index). Loaded with mmap=True to keep RAM usage low
# on this 16GB machine — rank_bm25's plain-dict-per-document format
# caused a MemoryError here on the full 111K-chunk corpus.
BM25S_INDEX_DIR = os.path.join(VECTOR_DB_DIR, "bm25s_index")

# Which embedding model to use for dense retrieval.
# Must match one of the keys below AND have completed embedding_step4.py.
ACTIVE_MODEL = "bge"

# "dense" | "sparse" | "hybrid"
RETRIEVAL_MODE = "hybrid"

TOP_K = 8

# RRF fusion settings (used when RETRIEVAL_MODE == "hybrid")
RRF_K = 60
CANDIDATE_POOL = 30

# model_key -> (huggingface_id, needs_query_prefix)
# needs_query_prefix=True means queries must be prefixed with "query: "
# at retrieval time (E5-style models — passages were prefixed with
# "passage: " during embedding_step4.py).
MODEL_HF_IDS = {
    "minilm":     ("sentence-transformers/all-MiniLM-L6-v2", False),
    "bge":        ("BAAI/bge-base-en-v1.5", False),
    "pubmedbert": ("NeuML/pubmedbert-base-embeddings", False),
    "biobert":    ("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", False),
    "e5":         ("intfloat/e5-base-v2", True),
}


# ==========================
# Lazy-loaded globals
# ==========================

_chunks_df = None
_dense_index = None
_dense_model = None
_bm25 = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_chunks():

    global _chunks_df

    if _chunks_df is None:
        _chunks_df = pd.read_csv(CHUNKS_FILE)

    return _chunks_df


def _load_dense(model_key):

    global _dense_index, _dense_model

    if _dense_index is None or getattr(_load_dense, "_loaded_model_key", None) != model_key:

        index_path = os.path.join(VECTOR_DB_DIR, f"{model_key}_index.faiss")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No FAISS index for model '{model_key}' at {index_path}. "
                f"Run embedding_step4.py for this model first."
            )

        hf_id, _ = MODEL_HF_IDS[model_key]

        print(f"Loading dense index + model for '{model_key}'...")

        _dense_index = faiss.read_index(index_path)
        _dense_model = SentenceTransformer(hf_id, device=device)

        _load_dense._loaded_model_key = model_key

    return _dense_index, _dense_model


def _load_bm25():

    global _bm25

    if _bm25 is None:

        if not os.path.exists(BM25S_INDEX_DIR):
            raise FileNotFoundError(
                f"No BM25 index at {BM25S_INDEX_DIR}. Run build_bm25_index.py first."
            )

        # mmap=True: keeps the index memory-mapped on disk instead of
        # fully loading it into RAM — this is what avoids the
        # MemoryError we hit with the old rank_bm25-based index.
        _bm25 = bm25s.BM25.load(BM25S_INDEX_DIR, mmap=True)

    return _bm25


# ==========================
# Retrieval strategies
# ==========================

def _dense_search(query, k, model_key):

    index, model = _load_dense(model_key)

    _, needs_prefix = MODEL_HF_IDS[model_key]

    q = ("query: " + query) if needs_prefix else query

    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    q_emb = np.array(q_emb).astype("float32")

    scores, indices = index.search(q_emb, k)

    return list(indices[0]), list(scores[0])


def _sparse_search(query, k):

    bm25 = _load_bm25()

    # Cap k to the corpus size to avoid bm25s raising a ValueError on
    # very small corpora (e.g. quick local tests with a handful of rows).
    corpus_size = bm25.scores["num_docs"] if hasattr(bm25, "scores") else None

    effective_k = min(k, corpus_size) if corpus_size else k

    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)

    results, scores = bm25.retrieve(query_tokens, k=effective_k, show_progress=False)

    # results/scores have shape (1, k) for a single query
    ids = [int(i) for i in results[0]]
    doc_scores = [float(s) for s in scores[0]]

    return ids, doc_scores


def _hybrid_search(query, k, model_key):
    """Reciprocal Rank Fusion of dense + sparse rankings."""

    dense_ids, _ = _dense_search(query, CANDIDATE_POOL, model_key)
    sparse_ids, _ = _sparse_search(query, CANDIDATE_POOL)

    rrf_scores = {}

    for rank, idx in enumerate(dense_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(sparse_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    top = ranked[:k]

    return [idx for idx, _ in top], [score for _, score in top]


# ==========================
# Public API
# ==========================

def retrieve_context(query, k=None, mode=None, model_key=None):
    """
    Returns a list of dicts: [{chunk, title, doc_id, chunk_id, PMID, score}, ...]
    """

    k = k or TOP_K
    mode = mode or RETRIEVAL_MODE
    model_key = model_key or ACTIVE_MODEL

    chunks_df = _load_chunks()

    if mode == "dense":
        ids, scores = _dense_search(query, k, model_key)
    elif mode == "sparse":
        ids, scores = _sparse_search(query, k)
    elif mode == "hybrid":
        try:
            ids, scores = _hybrid_search(query, k, model_key)
        except FileNotFoundError as e:
            print(f"Hybrid retrieval fell back to dense-only: {e}")
            ids, scores = _dense_search(query, k, model_key)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    results = []

    for idx, score in zip(ids, scores):

        row = chunks_df.iloc[int(idx)]

        results.append({
            "chunk": row["chunk"],
            "title": row.get("title", ""),
            "doc_id": row.get("doc_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "PMID": row.get("PMID", ""),
            "score": float(score),
        })

    return results


if __name__ == "__main__":

    print(f"ACTIVE_MODEL = {ACTIVE_MODEL}")
    print(f"RETRIEVAL_MODE = {RETRIEVAL_MODE}")

    test_query = "What is cancer?"

    print(f"\nTest query: {test_query}")

    results = retrieve_context(test_query, k=3)

    for i, r in enumerate(results):
        print(f"\n[{i+1}] score={r['score']:.4f}  title={r['title'][:80]}")
        print(f"    {r['chunk'][:200]}...")