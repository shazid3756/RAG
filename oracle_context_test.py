"""
Oracle-Context Diagnostic Test.

Purpose:
    Your RAG pipeline retrieves context from a SEPARATE 100K PubMed
    abstract corpus, not from the exact source paper each PubMedQA
    question was derived from. Classification accuracy on the full
    RAG pipeline was ~24% in earlier testing — well below the ~55%
    majority-class ("always yes") baseline.

    This script removes retrieval from the equation entirely: for
    each question, it feeds the model the TRUE source passage
    (item["context"]["contexts"] from pubmedqa_test.json) directly,
    using the exact same prompt template and label-extraction logic
    as the real pipeline (via rag_pipeline.build_prompt /
    generate_answer, and advanced_rag_evaluation.extract_label).

    Compare this script's accuracy to advanced_rag_evaluation.py's
    accuracy on the SAME sample:

      - Oracle accuracy still low (~similar to RAG)
            -> bottleneck is the LLM/prompt itself (Qwen2.5 struggling
               with this classification task, or hedging), not retrieval.
               Retrieval quality won't fix this — focus on prompt
               engineering, few-shot examples, or a different LLM.

      - Oracle accuracy much higher than RAG accuracy
            -> bottleneck IS retrieval — the model can classify
               correctly when given the right passage, it just isn't
               getting the right passage from your 100K corpus. Focus
               on embedding model choice, hybrid retrieval tuning, or
               growing/adjusting the corpus.

Run from the project root:
    python oracle_context_test.py
"""

import json
import os
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from rag_pipeline import build_prompt, generate_answer
from advanced_rag_evaluation import extract_label


# ==========================
# Configuration
# ==========================

TEST_FILE = "evaluation/pubmedqa_test.json"

if not os.path.exists(TEST_FILE) and os.path.exists("pubmedqa_test.json"):
    TEST_FILE = "pubmedqa_test.json"

RESULT_DIR = "evaluation/results"

os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULT_DIR, "oracle_predictions.csv")

# Keep this the SAME as advanced_rag_evaluation.py's SAMPLE_LIMIT
# (100) when you want a like-for-like comparison on the same
# question subset. json files are read in the same fixed order both
# scripts use, so the first N questions match exactly.
SAMPLE_LIMIT = 100


# ==========================
# Data loading
# ==========================

def load_questions():

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_true_context_text(item):
    """Extract the TRUE source passage(s) this question was derived
    from — same logic as retrieval_diagnostic.py, kept in sync so
    both scripts treat the ground-truth context identically."""

    context = item.get("context")

    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        return "\n\n".join(contexts)

    if isinstance(context, list):
        return "\n\n".join(context)

    if isinstance(context, str):
        return context

    return ""


# ==========================
# Evaluation
# ==========================

def run_oracle_test():

    questions = load_questions()

    if SAMPLE_LIMIT is not None:
        questions = questions[:SAMPLE_LIMIT]

    print("Total samples (oracle-context):", len(questions))

    y_true = []
    y_pred = []
    results = []
    latency_list = []

    missing_context_count = 0

    for index, item in enumerate(questions):

        question = item["question"]
        true_label = item["answer"]

        true_context = get_true_context_text(item)

        print(f"\nProcessing: {index + 1} / {len(questions)}")

        if not true_context.strip():

            # Shouldn't normally happen for PubMedQA, but guard
            # against malformed entries rather than crashing the run.
            missing_context_count += 1

            y_true.append(true_label)
            y_pred.append("maybe")

            results.append({
                "question": question,
                "actual": true_label,
                "prediction": "maybe",
                "answer": "SKIPPED (no true context found in dataset)",
                "latency": 0.0,
            })

            continue

        start = time.time()

        try:

            prompt = build_prompt(true_context, question, mode="classification")

            response = generate_answer(prompt)

            prediction = extract_label(response)

        except Exception as e:

            print("Error:", e)

            response = str(e)
            prediction = "maybe"

        latency = round(time.time() - start, 2)

        latency_list.append(latency)

        y_true.append(true_label)
        y_pred.append(prediction)

        results.append({
            "question": question,
            "actual": true_label,
            "prediction": prediction,
            "answer": response,
            "latency": latency,
        })

    # ==========================
    # Save predictions
    # ==========================

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # ==========================
    # Metrics
    # ==========================

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    avg_latency = (sum(latency_list) / len(latency_list)) if latency_list else 0.0

    print("\n======================")
    print("Oracle-Context Results")
    print("======================")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Average Latency:", avg_latency)

    if missing_context_count:
        print(
            f"\nWARNING: {missing_context_count} question(s) had no "
            f"true context found in the dataset and were scored as "
            f"'maybe' by default — check the dataset format if this "
            f"count is large."
        )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Average Latency"],
        "Value": [accuracy, precision, recall, f1, avg_latency],
    })

    metrics.to_csv(os.path.join(RESULT_DIR, "oracle_metrics.csv"), index=False)

    print("\nCompleted!")
    print("Predictions saved to:", OUTPUT_FILE)
    print("Metrics saved to:", os.path.join(RESULT_DIR, "oracle_metrics.csv"))

    print(
        "\nNext step: compare this accuracy to advanced_rag_evaluation.py's "
        "accuracy on the SAME first "
        f"{SAMPLE_LIMIT if SAMPLE_LIMIT else 'N'} questions.\n"
        "  - Oracle >> RAG accuracy  -> retrieval is the bottleneck.\n"
        "  - Oracle ~= RAG accuracy  -> the LLM/prompt is the bottleneck, "
        "not retrieval."
    )


if __name__ == "__main__":
    run_oracle_test()