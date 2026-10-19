"""
Retrieval Strategy Benchmark: dense vs sparse (BM25) vs hybrid (RRF
fusion), with the embedding model held FIXED at "minilm" — the
strongest performer from embedding_benchmark_mcqa.py (62.0% MedQA,
47.3% MedMCQA) — so only the retrieval STRATEGY varies, isolating its
effect independent of embedding choice.

Evaluated on MedQA + MedMCQA (same 300-question sample size as
embedding_benchmark_mcqa.py, for a like-for-like comparison).

Per-(mode, dataset) resume: results saved immediately, already-done
combinations skipped on re-run.

Run from the project root (after download_mcqa_datasets.py):
    python retrieval_strategy_benchmark.py
"""

import json
import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import retriever
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


# ==========================
# Configuration
# ==========================

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TABLE = os.path.join(RESULT_DIR, "retrieval_strategy_benchmark.csv")

# Best embedding model from embedding_benchmark_mcqa.py — held fixed
# so this benchmark isolates the effect of retrieval STRATEGY only.
FIXED_MODEL = "minilm"

MODES_TO_TEST = ["dense", "sparse", "hybrid"]

TOP_K = 8

# Matches embedding_benchmark_mcqa.py's sample size for comparability.
SAMPLE_LIMIT = 300


# ==========================
# Helpers
# ==========================

def load_questions(dataset_key):

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None:
        questions = questions[:SAMPLE_LIMIT]

    return questions


def load_completed():
    """Returns a set of (mode, dataset) tuples already in the CSV."""

    if not os.path.exists(OUTPUT_TABLE):
        return set()

    try:
        existing = pd.read_csv(OUTPUT_TABLE)
        return set(zip(existing["mode"], existing["dataset"]))
    except Exception:
        return set()


def append_result_to_csv(result):

    row_df = pd.DataFrame([result])

    if os.path.exists(OUTPUT_TABLE):
        existing = pd.read_csv(OUTPUT_TABLE)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(OUTPUT_TABLE, index=False)


# ==========================
# Per (mode, dataset) evaluation
# ==========================

def evaluate(mode, dataset_key, questions):

    y_true = []
    y_pred = []
    latencies = []
    fallback_count = 0

    for index, item in enumerate(questions):

        question = item["question"]
        options = item["options"]
        true_answer = item["answer_idx"]

        print(f"  [{mode} / {dataset_key}] {index + 1} / {len(questions)}", end="\r")

        start = time.time()

        try:

            results = retriever.retrieve_context(
                question, k=TOP_K, mode=mode, model_key=FIXED_MODEL
            )

            chunks = [r["chunk"] for r in results]

            context_text = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"

            prompt = build_mcqa_prompt(context_text, question, options)

            response = generate_answer(prompt)

            prediction = extract_choice(response)

            if prediction is None:
                fallback_count += 1
                prediction = "A"

        except Exception as e:

            print(f"\n  [{mode}/{dataset_key}] ERROR on question {index + 1}: {e}")

            prediction = "A"
            fallback_count += 1

        latencies.append(time.time() - start)

        y_true.append(true_answer)
        y_pred.append(prediction)

    print()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fallback_rate = fallback_count / len(questions) if questions else 0.0

    return {
        "mode": mode,
        "dataset": dataset_key,
        "embedding_model": FIXED_MODEL,
        "num_questions": len(questions),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_sec": round(avg_latency, 2),
        "fallback_rate": round(fallback_rate, 4),
    }


# ==========================
# Main
# ==========================

def main():

    for dataset_key, path in DATASET_FILES.items():
        if not os.path.exists(path):
            print(f"Missing {path} — run download_mcqa_datasets.py first.")
            return

    questions_by_dataset = {k: load_questions(k) for k in DATASET_FILES}

    completed = load_completed()

    if completed:
        print(f"Already completed (will skip): {sorted(completed)}\n")

    print(f"Fixed embedding model: {FIXED_MODEL}")
    print(f"Modes to test: {MODES_TO_TEST}")
    print(f"Datasets: {list(DATASET_FILES.keys())} (sample size {SAMPLE_LIMIT} each)\n")

    for mode in MODES_TO_TEST:

        for dataset_key, questions in questions_by_dataset.items():

            if (mode, dataset_key) in completed:
                print(f"Skipping ({mode}, {dataset_key}) — already done.")
                continue

            print(f"\n{'=' * 50}")
            print(f"Mode: {mode}  |  Dataset: {dataset_key}")
            print(f"{'=' * 50}")

            try:

                result = evaluate(mode, dataset_key, questions)

                append_result_to_csv(result)

                print(f"  -> accuracy={result['accuracy']}  f1={result['f1']}  "
                      f"avg_latency={result['avg_latency_sec']}s")

            except Exception as e:

                print(f"ERROR on ({mode}, {dataset_key}): {e}")
                print("Continuing — re-run later to retry this combination.")

    if not os.path.exists(OUTPUT_TABLE):
        print("\nNo results completed.")
        return

    df = pd.read_csv(OUTPUT_TABLE).sort_values(["dataset", "accuracy"], ascending=[True, False])

    print("\n" + "=" * 70)
    print("Retrieval Strategy Benchmark (embedding fixed: "
          f"{FIXED_MODEL}) — Summary")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nSaved to:", OUTPUT_TABLE)


if __name__ == "__main__":
    main()