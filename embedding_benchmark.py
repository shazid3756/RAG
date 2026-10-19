"""
Embedding Model Benchmark (Step 11 of the research roadmap).

Runs the SAME evaluation (classification accuracy on PubMedQA +
retrieval-overlap diagnostic) across all 5 embedding models, one at a
time, with GPU memory fully released between models (same pattern as
embedding_step4.py) so a 12GB VRAM / 16GB RAM machine can run all 5
without restarting the script.

For each model, reports:
  - Accuracy / Precision / Recall / F1 (classification, via the
    normal RAG pipeline — retrieval + Qwen2.5 generation)
  - Average retrieval overlap score (proxy for retrieval quality:
    word-overlap between retrieved chunks and the TRUE PubMedQA
    source passage — same metric as retrieval_diagnostic.py)
  - Average latency, FINAL_ANSWER fallback rate

RETRIEVAL_MODE is fixed (hybrid) across all models so embedding
choice is the only thing that varies — a fair, controlled comparison.

Run from the project root (this can take a while — 5 models x
SAMPLE_LIMIT questions each):
    python embedding_benchmark.py
"""

import gc
import json
import os
import re
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import retriever
from rag_pipeline import build_prompt, generate_answer
from advanced_rag_evaluation import extract_label
import advanced_rag_evaluation as are


# ==========================
# Configuration
# ==========================

TEST_FILE = "evaluation/pubmedqa_test.json"

if not os.path.exists(TEST_FILE) and os.path.exists("pubmedqa_test.json"):
    TEST_FILE = "pubmedqa_test.json"

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TABLE = os.path.join(RESULT_DIR, "embedding_benchmark.csv")

MODELS_TO_TEST = ["minilm", "bge", "pubmedbert", "biobert", "e5"]

# Fixed across all models so ONLY the embedding model varies.
RETRIEVAL_MODE = "hybrid"

TOP_K = 8

# Full evaluation set (no cap). This will take a long time — 5
# models x however many questions are in pubmedqa_test.json. Combined
# with per-model resume below, it's safe to interrupt and re-run.
SAMPLE_LIMIT = None


# ==========================
# Helpers
# ==========================

def tokenize_words(text):
    return set(re.findall(r"[a-z]{3,}", str(text).lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def get_true_context_text(item):

    context = item.get("context")

    if isinstance(context, dict):
        return " ".join(context.get("contexts", []))
    if isinstance(context, list):
        return " ".join(context)
    if isinstance(context, str):
        return context
    return ""


def load_questions():

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None:
        questions = questions[:SAMPLE_LIMIT]

    return questions


def load_completed_models():
    """Per-model resume: if embedding_benchmark.csv already has a row
    for a model, that model is considered done and will be skipped.
    This mirrors embedding_step4.py's resume pattern — essential here
    since a full 5-model x ~1000-question run can take many hours and
    should be safely interruptible."""

    if not os.path.exists(OUTPUT_TABLE):
        return set()

    try:
        existing = pd.read_csv(OUTPUT_TABLE)
        return set(existing["model"].tolist())
    except Exception:
        return set()


def append_result_to_csv(result):
    """Append a single model's result row to the CSV immediately
    after it finishes, rather than waiting until all models are done
    — so a crash on model 4 doesn't lose models 1-3's results."""

    row_df = pd.DataFrame([result])

    if os.path.exists(OUTPUT_TABLE):
        existing = pd.read_csv(OUTPUT_TABLE)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(OUTPUT_TABLE, index=False)


def release_model_memory():
    """Same cleanup pattern as embedding_step4.py: fully release the
    currently-loaded dense model/index before the next model loads,
    so VRAM/RAM usage doesn't accumulate across 5 sequential runs."""

    retriever._dense_model = None
    retriever._dense_index = None

    if hasattr(retriever._load_dense, "_loaded_model_key"):
        retriever._load_dense._loaded_model_key = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================
# Per-model evaluation
# ==========================

def evaluate_model(model_key, questions):

    are.FALLBACK_COUNT = 0

    y_true = []
    y_pred = []
    latencies = []
    overlap_scores = []

    for index, item in enumerate(questions):

        question = item["question"]
        true_label = item["answer"]
        true_context_text = get_true_context_text(item)
        true_tokens = tokenize_words(true_context_text)

        print(f"  [{model_key}] {index + 1} / {len(questions)}", end="\r")

        start = time.time()

        try:

            results = retriever.retrieve_context(
                question, k=TOP_K, mode=RETRIEVAL_MODE, model_key=model_key
            )

            chunks = [r["chunk"] for r in results]

            # Retrieval-quality proxy: best word-overlap between any
            # retrieved chunk and the true source passage.
            best_overlap = 0.0

            for chunk in chunks:
                overlap = jaccard(true_tokens, tokenize_words(chunk))
                best_overlap = max(best_overlap, overlap)

            overlap_scores.append(best_overlap)

            if not chunks:
                prediction = "maybe"
                response = "No relevant context found."

            else:

                context_text = "\n\n".join(chunks)

                prompt = build_prompt(context_text, question, mode="classification")

                response = generate_answer(prompt)

                prediction = extract_label(response)

        except Exception as e:

            print(f"\n  [{model_key}] ERROR on question {index + 1}: {e}")

            prediction = "maybe"
            overlap_scores.append(0.0)

        latencies.append(time.time() - start)

        y_true.append(true_label)
        y_pred.append(prediction)

    print()  # newline after the \r progress line

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    fallback_rate = are.FALLBACK_COUNT / len(questions) if questions else 0.0

    return {
        "model": model_key,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_retrieval_overlap": round(avg_overlap, 4),
        "avg_latency_sec": round(avg_latency, 2),
        "final_answer_fallback_rate": round(fallback_rate, 4),
    }


# ==========================
# Main
# ==========================

def main():

    questions = load_questions()

    print(f"Total questions per model: {len(questions)}")
    print(f"Models to benchmark: {MODELS_TO_TEST}")
    print(f"Retrieval mode (fixed): {RETRIEVAL_MODE}\n")

    completed_models = load_completed_models()

    if completed_models:
        print(f"Already completed (will skip): {sorted(completed_models)}\n")

    all_results = []

    for model_key in MODELS_TO_TEST:

        if model_key in completed_models:
            print(f"Skipping '{model_key}' — already in {OUTPUT_TABLE}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Benchmarking embedding model: {model_key}")
        print(f"{'=' * 50}")

        try:

            result = evaluate_model(model_key, questions)

            all_results.append(result)

            # Save immediately — don't wait for all 5 models to finish.
            append_result_to_csv(result)

            print(f"  -> accuracy={result['accuracy']}  "
                  f"f1={result['f1']}  "
                  f"avg_overlap={result['avg_retrieval_overlap']}  "
                  f"avg_latency={result['avg_latency_sec']}s")

            print(f"  Saved to {OUTPUT_TABLE}")

        except Exception as e:

            print(f"ERROR benchmarking '{model_key}': {e}")
            print("Skipping to next model. Re-run this script later to "
                  "retry the failed model (it was not saved, so it "
                  "won't be skipped on the next run).")

        finally:

            # Always release memory, even if this model's run errored,
            # so a failure on one model doesn't cascade into OOM on
            # the next.
            release_model_memory()

    if not os.path.exists(OUTPUT_TABLE):
        print("\nNo models completed successfully.")
        return

    df = pd.read_csv(OUTPUT_TABLE).sort_values("accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("Embedding Model Benchmark — Summary (sorted by accuracy)")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nFull comparison table saved to:", OUTPUT_TABLE)


if __name__ == "__main__":
    main()