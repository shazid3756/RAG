"""
Step 5: Vector DB sanity check.

Dynamically discovers every embedding model that has BOTH:
  - embeddings/<model>_embeddings.npy
  - vector_db/<model>_index.faiss

No hardcoded model list — if you add a 6th model later (e.g. re-run
embedding_step4.py with a new entry in MODELS), this script picks it
up automatically.

Checks, for each discovered model:
  - FAISS index vector count matches embeddings row count
  - Both match the number of rows in data/chunks/pubmed_chunks.csv
    (since row i of the chunks file == vector i in every index)

Run from the project root:
    python vector_db.py
"""

import os
import glob

import numpy as np
import pandas as pd
import faiss


EMBEDDINGS_DIR = "embeddings"
VECTOR_DB_DIR = "vector_db"
CHUNKS_FILE = "data/chunks/pubmed_chunks.csv"


def discover_models():
    """Find every model_key that has both an embeddings .npy and a
    FAISS index file, by scanning the directories rather than
    relying on a hardcoded list."""

    models = []

    emb_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*_embeddings.npy"))

    for emb_path in emb_files:

        base = os.path.basename(emb_path)
        model_key = base.replace("_embeddings.npy", "")

        index_path = os.path.join(VECTOR_DB_DIR, f"{model_key}_index.faiss")

        if os.path.exists(index_path):
            models.append(model_key)
        else:
            print(f"Warning: found embeddings for '{model_key}' but no "
                  f"matching FAISS index — skipping.")

    return sorted(models)


def main():

    print("Discovering available embedding models...")

    models = discover_models()

    if not models:
        print("No complete (embeddings + index) models found. "
              "Run embedding_step4.py first.")
        return

    print("Found models:", models)

    print("\nLoading chunks file:", CHUNKS_FILE)

    chunks_df = pd.read_csv(CHUNKS_FILE)

    n_chunks = len(chunks_df)

    print("Chunks rows:", n_chunks)

    print(f"\n{'Model':<15}{'Embeddings':<15}{'FAISS vectors':<18}{'Status'}")
    print("-" * 60)

    all_ok = True

    for model_key in models:

        emb_path = os.path.join(EMBEDDINGS_DIR, f"{model_key}_embeddings.npy")
        index_path = os.path.join(VECTOR_DB_DIR, f"{model_key}_index.faiss")

        embeddings = np.load(emb_path)
        n_emb = embeddings.shape[0]

        index = faiss.read_index(index_path)
        n_vec = index.ntotal

        ok = (n_emb == n_vec == n_chunks)

        status = "OK" if ok else "MISMATCH"

        if not ok:
            all_ok = False

        print(f"{model_key:<15}{n_emb:<15}{n_vec:<18}{status}")

    print("-" * 60)

    if all_ok:
        print("\nAll models aligned with chunks file. Vector DB is consistent.")
    else:
        print("\nSome models have a mismatch — investigate before running "
              "retrieval on the affected model(s).")

    print("\nAvailable ACTIVE_MODEL values for retriever.py:")

    for model_key in models:
        print(" -", model_key)


if __name__ == "__main__":
    main()