"""
Step 4: Embedding Generation (all 5 benchmark models).

Processes models ONE AT A TIME (not simultaneously) to stay within
16GB system RAM / 12GB VRAM. After each model finishes, it is fully
unloaded from memory before the next one loads.

Per-model resume: if a model's output files already exist, that
model is SKIPPED — so if the script is interrupted partway through
(e.g. after finishing MiniLM and BGE but crashing on PubMedBERT),
re-running just continues with the remaining models.

Models:
    1. MiniLM        sentence-transformers/all-MiniLM-L6-v2
    2. BGE-base       BAAI/bge-base-en-v1.5
    3. PubMedBERT     NeuML/pubmedbert-base-embeddings
    4. BioBERT        pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
    5. E5-base        intfloat/e5-base-v2  (needs "passage: " prefix)

Run from the project root:
    python embedding_step4.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time

import numpy as np
import pandas as pd
import torch
import faiss

from sentence_transformers import SentenceTransformer


# ==========================
# Configuration
# ==========================

CHUNKS_FILE = "data/chunks/pubmed_chunks.csv"

EMBEDDINGS_DIR = "embeddings"
VECTOR_DB_DIR = "vector_db"

# name -> (huggingface_id, needs_e5_passage_prefix, batch_size)
MODELS = {
    "minilm":     ("sentence-transformers/all-MiniLM-L6-v2", False, 128),
    "bge":        ("BAAI/bge-base-en-v1.5", False, 64),
    "pubmedbert": ("NeuML/pubmedbert-base-embeddings", False, 64),
    "biobert":    ("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", False, 64),
    "e5":         ("intfloat/e5-base-v2", True, 64),
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# Helpers
# ==========================

def model_output_paths(model_key):

    emb_path = os.path.join(EMBEDDINGS_DIR, f"{model_key}_embeddings.npy")
    index_path = os.path.join(VECTOR_DB_DIR, f"{model_key}_index.faiss")

    return emb_path, index_path


def is_model_done(model_key):

    emb_path, index_path = model_output_paths(model_key)

    return os.path.exists(emb_path) and os.path.exists(index_path)


def process_model(model_key, hf_id, needs_passage_prefix, batch_size, chunks):

    emb_path, index_path = model_output_paths(model_key)

    print(f"\n{'=' * 50}")
    print(f"Model: {model_key}  ({hf_id})")
    print(f"{'=' * 50}")

    print("Loading model...")

    model = SentenceTransformer(hf_id, device=device)

    texts = chunks

    if needs_passage_prefix:
        # E5 models require this prefix on documents (and "query: "
        # on queries at retrieval time — handled in rag_pipeline.py).
        texts = ["passage: " + t for t in texts]

    print(f"Encoding {len(texts)} chunks (batch_size={batch_size})...")

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start

    print(f"Encoding took {round(elapsed, 1)} seconds.")

    embeddings = np.array(embeddings).astype("float32")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    np.save(emb_path, embeddings)

    print("Saved embeddings to:", emb_path)

    # Build FAISS index (cosine similarity via inner product, since
    # embeddings are normalized above).
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    faiss.write_index(index, index_path)

    print("Saved FAISS index to:", index_path)

    # --- Explicit cleanup before loading the next model ---
    del model
    del embeddings
    del index

    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    print(f"Model '{model_key}' complete. Memory released.")


# ==========================
# Main
# ==========================

def main():

    print("Loading chunks:", CHUNKS_FILE)

    df = pd.read_csv(CHUNKS_FILE)

    chunks = df["chunk"].astype(str).tolist()

    print("Total chunks:", len(chunks))
    print("Device:", device)

    for model_key, (hf_id, needs_prefix, batch_size) in MODELS.items():

        if is_model_done(model_key):

            print(f"\nSkipping '{model_key}' — already completed "
                  f"(found existing output files).")

            continue

        try:

            process_model(model_key, hf_id, needs_prefix, batch_size, chunks)

        except Exception as e:

            print(f"\nERROR processing '{model_key}': {e}")
            print(
                "Skipping to next model. Re-run this script later to "
                "retry the failed model (it will not be marked done)."
            )

            # Clean up whatever was partially loaded before continuing
            gc.collect()

            if device == "cuda":
                torch.cuda.empty_cache()

    print("\nAll models processed (or skipped if already done).")

    print("\nSummary:")

    for model_key in MODELS:

        status = "DONE" if is_model_done(model_key) else "MISSING/FAILED"

        print(f"  {model_key}: {status}")


if __name__ == "__main__":
    main()