"""
Retrieval Diagnostic for PubMedQA evaluation.

PubMedQA questions are each derived from a SPECIFIC source abstract,
which is included in pubmedqa_test.json under item["context"]["contexts"].

Your RAG pipeline retrieves from a SEPARATE 100K PubMed abstract corpus
(FAISS index), not from that specific source abstract. This script
checks how much word-overlap exists between what your retriever
actually returns and the true source abstract for a sample of
questions, to see whether the source document is even present in
your corpus.

Run from the project root:
    python evaluation/retrieval_diagnostic.py
"""

import json
import os
import re
import sys


# rag_pipeline.py now lives at the project root (not under src/), so
# this script is meant to be run from the project root:
#     python retrieval_diagnostic.py
from rag_pipeline import retrieve_context


TEST_FILE = "evaluation/pubmedqa_test.json"

if not os.path.exists(TEST_FILE) and os.path.exists("pubmedqa_test.json"):
    TEST_FILE = "pubmedqa_test.json"

SAMPLE_SIZE = 30

# Below this overlap score, we consider the retrieved context to be
# essentially unrelated to the true source abstract.
LOW_OVERLAP_THRESHOLD = 0.08


def tokenize(text):

    words = re.findall(r"[a-z]{3,}", text.lower())

    return set(words)


def jaccard(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union if union else 0.0


def get_true_context_text(item):

    context = item.get("context")

    if isinstance(context, dict):

        contexts = context.get("contexts", [])

        return " ".join(contexts)

    if isinstance(context, list):

        return " ".join(context)

    if isinstance(context, str):

        return context

    return ""


def main():

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    sample = questions[:SAMPLE_SIZE]

    print(f"Checking retrieval overlap for {len(sample)} sample questions...\n")

    scores = []
    low_overlap_count = 0

    for index, item in enumerate(sample):

        question = item["question"]

        true_context_text = get_true_context_text(item)

        true_tokens = tokenize(true_context_text)

        retrieved_chunks = retrieve_context(question)

        best_overlap = 0.0

        for chunk in retrieved_chunks:

            chunk_tokens = tokenize(chunk)

            overlap = jaccard(true_tokens, chunk_tokens)

            best_overlap = max(best_overlap, overlap)

        scores.append(best_overlap)

        flag = ""

        if best_overlap < LOW_OVERLAP_THRESHOLD:
            low_overlap_count += 1
            flag = "  <-- LOW OVERLAP (likely source doc not in corpus)"

        print(
            f"[{index+1}/{len(sample)}] best_overlap={best_overlap:.3f}"
            f"  question: {question[:70]}...{flag}"
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print("\n==============================")
    print("Summary")
    print("==============================")
    print(f"Average best overlap score: {avg_score:.3f}")
    print(
        f"Questions with low overlap (<{LOW_OVERLAP_THRESHOLD}): "
        f"{low_overlap_count} / {len(sample)}"
        f" ({round(low_overlap_count / len(sample) * 100, 1)}%)"
    )

    print(
        "\nInterpretation:\n"
        "- Overlap scores here are naturally low even for GOOD retrieval\n"
        "  (different corpus, paraphrased text), so don't expect near 1.0.\n"
        "- But if the vast majority of questions show near-zero overlap,\n"
        "  it strongly suggests your 100K-abstract corpus rarely contains\n"
        "  the actual source paper PubMedQA questions were built from.\n"
        "  In that case, low yes/no/maybe accuracy is expected and is a\n"
        "  retrieval-corpus limitation, not a bug in your pipeline code.\n"
    )


if __name__ == "__main__":
    main()