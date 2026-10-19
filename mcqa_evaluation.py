"""
RAG Evaluation on MedQA / MedMCQA (multiple-choice, general medical
knowledge — NOT tied to one specific source document, unlike
PubMedQA). See download_mcqa_datasets.py for why this benchmark is a
fairer test of THIS pipeline's open-corpus retrieval than PubMedQA.

Run download_mcqa_datasets.py first to produce:
    evaluation/medqa_test.json
    evaluation/medmcqa_val.json

Then run this script:
    python mcqa_evaluation.py

Per-dataset resume: results are saved to
evaluation/results/mcqa_benchmark.csv immediately after each dataset
finishes, and already-completed datasets are skipped on re-run — same
safety pattern as embedding_benchmark.py, since a full run can take a
long time.
"""

import json
import os
import re
import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from rag_pipeline import generate_answer
import retriever


# ==========================
# Configuration
# ==========================

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TABLE = os.path.join(RESULT_DIR, "mcqa_benchmark.csv")
PREDICTIONS_DIR = RESULT_DIR

# Which embedding model + retrieval strategy to use for context
# retrieval. Keep consistent with your best result from
# embedding_benchmark.py (default: bge, hybrid).
ACTIVE_MODEL = "bge"
RETRIEVAL_MODE = "hybrid"
TOP_K = 8

# None = full dataset. Set to e.g. 200 for a quick sanity check before
# committing to a full run.
SAMPLE_LIMIT = None


# ==========================
# Prompt + label extraction
# ==========================

def build_mcqa_prompt(context_text, question, options):

    options_text = "\n".join(f"{letter}) {text}" for letter, text in sorted(options.items()))

    prompt = f"""

You are a medical exam assistant.


========================

Reference Material (may or may not be directly relevant):

{context_text}


========================

Question:

{question}

Options:

{options_text}


Instructions:

Use the reference material if it helps, but rely on your own medical
knowledge as the primary basis for your answer — the reference
material is supplementary, not exhaustive.

First give a SHORT explanation in ONE sentence only (max ~25 words).
Then, on a new final line, output EXACTLY in this format (replace
<letter> with A, B, C, or D), with nothing else after it:

FINAL_ANSWER: <letter>

Answer:

"""

    return prompt


def extract_choice(response_text):
    """Robust A/B/C/D extractor, mirroring
    advanced_rag_evaluation.extract_label's approach: look for the
    explicit FINAL_ANSWER marker first (flexible spacing/punctuation),
    then fall back to the first standalone A-D letter found."""

    if not response_text:
        return None

    match = re.search(
        r"FINAL[_\s-]*ANSWER\s*[:\-]?\s*\(?\s*([ABCD])\b",
        response_text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    # Fallback: first standalone A-D letter (e.g. "B) ..." or "(B)")
    fallback = re.search(r"\b([ABCD])\)", response_text)

    if fallback:
        return fallback.group(1).upper()

    return None


def extract_bare_letter(response_text, max_len=15):
    """Stricter, separate extractor for use ONLY on short strict-retry
    responses (see build_strict_letter_prompt) — deliberately NOT
    merged into extract_choice's main logic, because a loose bare-
    letter search is unsafe on long free-text explanations: clinical
    vignette questions routinely start with phrases like "A patient
    presents with..." where "A" is just an article, not an answer.
    Restricting to short responses (the model was told to reply with
    ONLY a letter) keeps that false-positive risk low."""

    if not response_text:
        return None

    stripped = response_text.strip()

    if len(stripped) > max_len:
        return None

    match = re.search(r"\b([ABCD])\b", stripped, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return None


# ==========================
# Data + resume helpers
# ==========================

def load_questions(dataset_key):

    path = DATASET_FILES[dataset_key]

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None:
        questions = questions[:SAMPLE_LIMIT]

    return questions


def load_completed_datasets():

    if not os.path.exists(OUTPUT_TABLE):
        return set()

    try:
        existing = pd.read_csv(OUTPUT_TABLE)
        return set(existing["dataset"].tolist())
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
# Evaluation
# ==========================

def evaluate_dataset(dataset_key, questions):

    y_true = []
    y_pred = []
    latencies = []
    predictions_log = []

    fallback_count = 0

    for index, item in enumerate(questions):

        question = item["question"]
        options = item["options"]
        true_answer = item["answer_idx"]

        print(f"  [{dataset_key}] {index + 1} / {len(questions)}", end="\r")

        start = time.time()

        try:

            results = retriever.retrieve_context(
                question, k=TOP_K, mode=RETRIEVAL_MODE, model_key=ACTIVE_MODEL
            )

            chunks = [r["chunk"] for r in results]

            context_text = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"

            prompt = build_mcqa_prompt(context_text, question, options)

            response = generate_answer(prompt)

            prediction = extract_choice(response)

            if prediction is None:
                fallback_count += 1
                prediction = "A"  # neutral fallback, consistent default

        except Exception as e:

            print(f"\n  [{dataset_key}] ERROR on question {index + 1}: {e}")

            response = str(e)
            prediction = "A"
            fallback_count += 1

        latency = time.time() - start

        latencies.append(latency)

        y_true.append(true_answer)
        y_pred.append(prediction)

        predictions_log.append({
            "question": question,
            "true_answer": true_answer,
            "prediction": prediction,
            "response": response,
            "latency": round(latency, 2),
        })

    print()

    pd.DataFrame(predictions_log).to_csv(
        os.path.join(PREDICTIONS_DIR, f"{dataset_key}_predictions.csv"),
        index=False,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fallback_rate = fallback_count / len(questions) if questions else 0.0

    return {
        "dataset": dataset_key,
        "num_questions": len(questions),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_latency_sec": round(avg_latency, 2),
        "fallback_rate": round(fallback_rate, 4),
        "active_model": ACTIVE_MODEL,
        "retrieval_mode": RETRIEVAL_MODE,
    }


# ==========================
# Main
# ==========================

def main():

    for dataset_key, path in DATASET_FILES.items():
        if not os.path.exists(path):
            print(
                f"Missing {path} — run download_mcqa_datasets.py first "
                f"to generate it."
            )
            return

    completed = load_completed_datasets()

    if completed:
        print(f"Already completed (will skip): {sorted(completed)}\n")

    for dataset_key in DATASET_FILES:

        if dataset_key in completed:
            print(f"Skipping '{dataset_key}' — already in {OUTPUT_TABLE}")
            continue

        questions = load_questions(dataset_key)

        print(f"\n{'=' * 50}")
        print(f"Evaluating on: {dataset_key}  ({len(questions)} questions)")
        print(f"{'=' * 50}")

        try:

            result = evaluate_dataset(dataset_key, questions)

            append_result_to_csv(result)

            print(f"  -> accuracy={result['accuracy']}  f1={result['f1']}  "
                  f"avg_latency={result['avg_latency_sec']}s")

            print(f"  Saved to {OUTPUT_TABLE}")

        except Exception as e:

            print(f"ERROR evaluating '{dataset_key}': {e}")
            print("Re-run this script later to retry (it was not "
                  "saved, so it won't be skipped on the next run).")

    if not os.path.exists(OUTPUT_TABLE):
        print("\nNo datasets completed successfully.")
        return

    df = pd.read_csv(OUTPUT_TABLE)

    print("\n" + "=" * 70)
    print("MedQA / MedMCQA RAG Evaluation — Summary")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nFull results saved to:", OUTPUT_TABLE)


if __name__ == "__main__":
    main()