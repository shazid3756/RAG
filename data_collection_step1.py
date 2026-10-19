"""
Step 1: Data Collection (fresh restart).

Source: MedRAG/pubmed on HuggingFace — 23.9M ready-to-use PubMed
abstract snippets (title + content), used for medical RAG.
https://huggingface.co/datasets/MedRAG/pubmed

Streams the dataset (does NOT download all 23.9M rows) and saves the
first TARGET_TOTAL valid (non-empty title+abstract) rows to CSV.

Resumable: if interrupted (crash, closed terminal, etc.), just run
this script again — it picks up exactly where it left off using a
checkpoint file, instead of starting over.

Requires: pip install datasets --break-system-packages

Run from the project root:
    python data_collection_step1.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import csv

from datasets import load_dataset


# ==========================
# Configuration
# ==========================

TARGET_TOTAL = 100_000

# How many valid rows to buffer before flushing to disk + updating
# the checkpoint. Smaller = safer (less lost on crash) but slightly
# slower; larger = faster but more lost on an unexpected crash.
BATCH_SIZE = 2000

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pubmed_100k_raw.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "collection_checkpoint.json")

FIELDNAMES = ["id", "title", "content", "PMID"]


# ==========================
# Checkpoint helpers
# ==========================

def load_checkpoint():
    """Returns (raw_rows_consumed, valid_rows_collected)."""

    if os.path.exists(CHECKPOINT_FILE):

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("raw_consumed", 0), data.get("collected", 0)

    return 0, 0


def save_checkpoint(raw_consumed, collected):

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"raw_consumed": raw_consumed, "collected": collected},
            f
        )


# ==========================
# Main collection
# ==========================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_consumed, collected = load_checkpoint()

    if collected >= TARGET_TOTAL:
        print(
            f"Target already reached: {collected} / {TARGET_TOTAL}. "
            "Nothing to do. Delete the checkpoint file to start over."
        )
        return

    if collected > 0:
        print(
            f"Resuming: {collected} / {TARGET_TOTAL} valid rows "
            f"already collected ({raw_consumed} raw rows consumed)."
        )
    else:
        print("Starting fresh collection.")

    print("Connecting to MedRAG/pubmed dataset (streaming mode)...")

    dataset = load_dataset("MedRAG/pubmed", split="train", streaming=True)

    if raw_consumed > 0:

        print(f"Skipping {raw_consumed} already-processed raw rows...")

        dataset = dataset.skip(raw_consumed)

    file_exists = os.path.exists(OUTPUT_FILE)

    mode = "a" if (file_exists and collected > 0) else "w"

    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if mode == "w":
            writer.writeheader()

        batch_count = 0

        try:

            for row in dataset:

                raw_consumed += 1

                title = (row.get("title") or "").strip()
                content = (row.get("content") or "").strip()

                # Skip rows with no usable title/abstract text.
                if not title or not content:
                    continue

                writer.writerow(
                    {
                        "id": row.get("id"),
                        "title": title,
                        "content": content,
                        "PMID": row.get("PMID"),
                    }
                )

                collected += 1
                batch_count += 1

                if batch_count >= BATCH_SIZE:

                    f.flush()

                    save_checkpoint(raw_consumed, collected)

                    print(f"Progress: {collected} / {TARGET_TOTAL}")

                    batch_count = 0

                if collected >= TARGET_TOTAL:
                    break

        finally:

            # Always save progress, even if interrupted mid-batch
            # (e.g. Ctrl+C, crash) so a re-run can resume cleanly.
            f.flush()

            save_checkpoint(raw_consumed, collected)

    print(f"\nDone. Collected {collected} / {TARGET_TOTAL} PubMed abstracts.")
    print("Saved to:", OUTPUT_FILE)

    if collected < TARGET_TOTAL:
        print(
            "\nTarget not yet reached (stream may have ended, or was "
            "interrupted). Run this script again to continue/resume."
        )


if __name__ == "__main__":
    main()