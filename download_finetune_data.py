"""
QLoRA Fine-Tuning — Stage 1: Training Data Preparation.

Downloads the TRAIN splits of MedQA and MedMCQA (distinct from the
test/validation splits already used throughout this project's
evaluation — so there is no data leakage between fine-tuning and the
before/after comparison later).

For each training question, retrieves context using the SAME
best-known retrieval configuration used everywhere else in this
project (embedding=minilm, mode=hybrid, k=8), then builds the exact
same prompt format used at inference time (mcqa_evaluation.build_mcqa_prompt)
paired with a target completion. This way, fine-tuning teaches the
model to answer WITHIN this specific RAG prompt structure, grounded in
retrieved context — not just generic question-answering.

Output: a single JSONL file, each line: {"prompt": ..., "completion": ...}
ready for QLoRA SFT training (Stage 2).

A held-out 10% slice is written separately for monitoring training
loss — this is NOT the same as the final before/after comparison,
which still uses the untouched evaluation/medqa_test.json and
evaluation/medmcqa_val.json files.

Run from the project root:
    python download_finetune_data.py
"""

import json
import os
import random

from datasets import load_dataset

import retriever
from mcqa_evaluation import build_mcqa_prompt


# ==========================
# Configuration
# ==========================

OUTPUT_DIR = "finetune_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.jsonl")

# Per-dataset cap. LoRA fine-tuning doesn't need the full train split
# (MedQA train alone is ~10K questions) to show a measurable effect,
# and each example requires a real retrieval call — capping keeps
# data-prep time reasonable while still giving several thousand
# examples total.
PER_DATASET_CAP = 1500

RETRIEVAL_MODEL = "minilm"
RETRIEVAL_MODE = "hybrid"
TOP_K = 8

VAL_FRACTION = 0.1
RANDOM_SEED = 42

COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


# ==========================
# Loading + normalizing (same schema as download_mcqa_datasets.py)
# ==========================

def load_medqa_train():

    print("Downloading MedQA train split...")

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    records = []

    for row in ds:

        options = row["options"]
        answer_idx = row["answer_idx"]

        if answer_idx not in options:
            continue

        records.append({
            "question": row["question"],
            "options": options,
            "answer_idx": answer_idx,
            "explanation": None,  # MedQA has no explanation field
            "source": "medqa",
        })

    print(f"  Loaded {len(records)} MedQA train questions.")

    return records


def load_medmcqa_train():

    print("Downloading MedMCQA train split...")

    ds = load_dataset("openlifescienceai/medmcqa", split="train")

    records = []

    for row in ds:

        cop = row["cop"]
        answer_idx = COP_TO_LETTER.get(cop)

        if answer_idx is None:
            continue

        options = {
            "A": row["opa"], "B": row["opb"],
            "C": row["opc"], "D": row["opd"],
        }

        explanation = (row.get("exp") or "").strip()

        records.append({
            "question": row["question"],
            "options": options,
            "answer_idx": answer_idx,
            "explanation": explanation if explanation else None,
            "source": "medmcqa",
        })

    print(f"  Loaded {len(records)} MedMCQA train questions.")

    return records


# ==========================
# Target completion
# ==========================

def first_sentence(text, max_words=25):
    """Trim an explanation down to roughly one sentence / max_words,
    matching the "one sentence, ~25 words" instruction given in the
    prompt at inference time."""

    text = text.replace("\n", " ").strip()

    # Cut at the first sentence-ending punctuation if present.
    for sep in [". ", "? ", "! "]:
        if sep in text:
            text = text.split(sep)[0] + sep.strip()
            break

    words = text.split()

    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text.strip()


def build_completion(item):

    letter = item["answer_idx"]
    correct_text = item["options"].get(letter, "")

    if item.get("explanation"):

        explanation = first_sentence(item["explanation"])

    else:

        explanation = f"The provided context and medical knowledge support {correct_text} as the correct choice."
        explanation = first_sentence(explanation)

    return f"{explanation}\nFINAL_ANSWER: {letter}"


# ==========================
# Main
# ==========================

def build_examples(records, dataset_label):

    examples = []

    for i, item in enumerate(records):

        print(f"  [{dataset_label}] retrieving context {i + 1} / {len(records)}", end="\r")

        try:

            results = retriever.retrieve_context(
                item["question"], k=TOP_K, mode=RETRIEVAL_MODE, model_key=RETRIEVAL_MODEL
            )

            chunks = [r["chunk"] for r in results]

            context_text = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"

            prompt = build_mcqa_prompt(context_text, item["question"], item["options"])

            completion = build_completion(item)

            examples.append({"prompt": prompt, "completion": completion})

        except Exception as e:

            print(f"\n  [{dataset_label}] skipping example {i + 1} due to error: {e}")

    print()

    return examples


def main():

    random.seed(RANDOM_SEED)

    medqa_records = load_medqa_train()
    medmcqa_records = load_medmcqa_train()

    random.Random(RANDOM_SEED).shuffle(medqa_records)
    random.Random(RANDOM_SEED).shuffle(medmcqa_records)

    medqa_records = medqa_records[:PER_DATASET_CAP]
    medmcqa_records = medmcqa_records[:PER_DATASET_CAP]

    print(f"\nUsing {len(medqa_records)} MedQA + {len(medmcqa_records)} MedMCQA "
          f"training questions ({len(medqa_records) + len(medmcqa_records)} total).")

    print("\nBuilding RAG-formatted training examples (this calls the "
          "retriever once per question)...")

    medqa_examples = build_examples(medqa_records, "medqa")
    medmcqa_examples = build_examples(medmcqa_records, "medmcqa")

    all_examples = medqa_examples + medmcqa_examples

    random.Random(RANDOM_SEED).shuffle(all_examples)

    val_size = int(len(all_examples) * VAL_FRACTION)

    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(train_examples)} training examples to {TRAIN_FILE}")
    print(f"Saved {len(val_examples)} validation examples to {VAL_FILE}")
    print(
        "\nNote: these are drawn from the TRAIN splits of MedQA/MedMCQA, "
        "distinct from evaluation/medqa_test.json and "
        "evaluation/medmcqa_val.json used for the before/after "
        "comparison — no data leakage."
    )


if __name__ == "__main__":
    main()