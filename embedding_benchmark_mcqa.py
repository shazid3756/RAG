"""
Embedding Model Benchmark on MedQA / MedMCQA.

Runs all 5 embedding models, one at a time (GPU memory fully released
between models), through the SAME multiple-choice evaluation on
MedQA + MedMCQA — the open-domain benchmarks where this pipeline
showed real signal (59.6% / 50.6% accuracy with BGE, vs ~18% on
PubMedQA where corpus coverage dominates; see download_mcqa_datasets.py
for why).

Uses a SAMPLE_LIMIT per dataset (default 300) to keep the full
5-model x 2-dataset sweep tractable — running the complete dataset
(1273 + 4183 questions) x 5 models would be ~27,000 LLM calls. Your
existing full-dataset BGE result from mcqa_evaluation.py remains the
headline number; this script is for the comparative ranking across
embedding models on a representative sample.

Requires evaluation/medqa_test.json and evaluation/medmcqa_val.json
(run download_mcqa_datasets.py first if you haven't).

Per-(model, dataset) resume: results are saved to
evaluation/results/mcqa_embedding_benchmark.csv immediately after each
model+dataset combination finishes, and already-completed combinations
are skipped on re-run.

Run from the project root:
    python embedding_benchmark_mcqa.py
"""

import gc
import json
import os
import time

import pandas as pd
import torch
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

OUTPUT_TABLE = os.path.join(RESULT_DIR, "mcqa_embedding_benchmark.csv")

MODELS_TO_TEST = ["minilm", "bge", "pubmedbert", "biobert", "e5"]

RETRIEVAL_MODE = "hybrid"
TOP_K = 8

# Per-dataset sample size for this comparative sweep (see module
# docstring for why this isn't the full dataset). Increase later for
# a fuller comparison once you know which model(s) look promising.
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
    """Returns a set of (model, dataset) tuples already in the CSV."""

    if not os.path.exists(OUTPUT_TABLE):
        return set()

    try:
        existing = pd.read_csv(OUTPUT_TABLE)
        return set(zip(existing["model"], existing["dataset"]))
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


def release_model_memory():
    """Same pattern as embedding_step4.py / embedding_benchmark.py:
    fully release the currently-loaded dense model/index before the
    next model loads."""

    retriever._dense_model = None
    retriever._dense_index = None

    if hasattr(retriever._load_dense, "_loaded_model_key"):
        retriever._load_dense._loaded_model_key = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================
# Per (model, dataset) evaluation
# ==========================

def evaluate(model_key, dataset_key, questions):

    y_true = []
    y_pred = []
    latencies = []
    fallback_count = 0

    for index, item in enumerate(questions):

        question = item["question"]
        options = item["options"]
        true_answer = item["answer_idx"]

        print(f"  [{model_key} / {dataset_key}] {index + 1} / {len(questions)}", end="\r")

        start = time.time()

        try:

            results = retriever.retrieve_context(
                question, k=TOP_K, mode=RETRIEVAL_MODE, model_key=model_key
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

            print(f"\n  [{model_key}/{dataset_key}] ERROR on question {index + 1}: {e}")

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
        "model": model_key,
        "dataset": dataset_key,
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

    print(f"Models: {MODELS_TO_TEST}")
    print(f"Datasets: {list(DATASET_FILES.keys())} (sample size {SAMPLE_LIMIT} each)\n")

    for model_key in MODELS_TO_TEST:

        did_any_work = False

        for dataset_key, questions in questions_by_dataset.items():

            if (model_key, dataset_key) in completed:
                print(f"Skipping ({model_key}, {dataset_key}) — already done.")
                continue

            did_any_work = True

            print(f"\n{'=' * 50}")
            print(f"Model: {model_key}  |  Dataset: {dataset_key}")
            print(f"{'=' * 50}")

            try:

                result = evaluate(model_key, dataset_key, questions)

                append_result_to_csv(result)

                print(f"  -> accuracy={result['accuracy']}  f1={result['f1']}  "
                      f"avg_latency={result['avg_latency_sec']}s")

            except Exception as e:

                print(f"ERROR on ({model_key}, {dataset_key}): {e}")
                print("Continuing — re-run later to retry this combination.")

        if did_any_work:
            # Release memory once per model (after both its datasets),
            # not after every single question — that's the expensive
            # operation (loading a new SentenceTransformer + FAISS index).
            release_model_memory()

    if not os.path.exists(OUTPUT_TABLE):
        print("\nNo results completed.")
        return

    df = pd.read_csv(OUTPUT_TABLE).sort_values(["dataset", "accuracy"], ascending=[True, False])

    print("\n" + "=" * 70)
    print("Embedding Model Benchmark on MedQA/MedMCQA — Summary")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nSaved to:", OUTPUT_TABLE)


if __name__ == "__main__":
    main()