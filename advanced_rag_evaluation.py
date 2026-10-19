import json
import csv
import time
import os
import sys


# rag_pipeline.py now lives at the project root (not under src/), so
# this script is meant to be run from the project root:
#     python advanced_rag_evaluation.py
from rag_pipeline import rag_answer


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ==========================
# Configuration
# ==========================

TEST_FILE = "evaluation/pubmedqa_test.json"

# Fallback: if the file isn't under evaluation/, try current directory
# (handles the case where pubmedqa_test.json was uploaded directly)
if not os.path.exists(TEST_FILE) and os.path.exists("pubmedqa_test.json"):
    TEST_FILE = "pubmedqa_test.json"

RESULT_DIR = "evaluation/results"

# ==========================
# Subset size for testing
# ==========================
# Quick validation run (100) to confirm the new decisive prompt
# reduces "maybe" over-use before committing hours to the full
# ~1000 sample run again. Set to None for the full run.
SAMPLE_LIMIT = 100


os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


OUTPUT_FILE = (
    RESULT_DIR +
    "/predictions.csv"
)



# ==========================
# Load Dataset
# ==========================

def load_questions():

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)




# ==========================
# Extract Model Label
# ==========================

import re


# Counts how many times no marker/last-line match was found and we
# had to fall back to the neutral "maybe" default. High values here
# mean the LLM isn't following the FINAL_ANSWER format reliably.
FALLBACK_COUNT = 0


def extract_label(answer):

    global FALLBACK_COUNT

    if not answer:
        FALLBACK_COUNT += 1
        return "maybe"

    text = answer.strip().lower()

    # Primary: look for the explicit final-answer marker. Flexible on
    # spacing/underscore/hyphen between "final" and "answer", since
    # models often write "Final Answer:" with a space instead of the
    # literal "FINAL_ANSWER:" requested in the prompt.
    match = re.search(
        r"final[\s_-]*answer\s*:?\s*(?:is\s*)?[\"'*]*\s*(yes|no|maybe)",
        text
    )

    if match:
        return match.group(1)

    # Secondary: plain "Answer: yes" style, in case the model ignores
    # the "final" wording entirely.
    match = re.search(r"\banswer\s*:\s*[\"'*]*\s*(yes|no|maybe)", text)

    if match:
        return match.group(1)

    # Tertiary: some models still put a clean single-word verdict on
    # the last non-empty line without any marker at all.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    last_line = lines[-1] if lines else ""

    for label in ("yes", "no", "maybe"):
        if re.fullmatch(rf"\W*{label}\W*", last_line):
            return label

    # Neutral fallback: don't bias toward "no"/"maybe" by scanning the
    # whole explanation (those words are common in medical hedging
    # language regardless of the actual conclusion).
    FALLBACK_COUNT += 1

    return "maybe"





# ==========================
# Evaluation
# ==========================

def run_evaluation():

    global FALLBACK_COUNT
    FALLBACK_COUNT = 0

    questions = load_questions()

    # Test on a small subset first (SAMPLE_LIMIT) before running the
    # full corpus, to catch OOM / config issues early.
    if SAMPLE_LIMIT is not None:
        questions = questions[:SAMPLE_LIMIT]

    print(
        "Total Samples:",
        len(questions)
    )


    y_true = []
    y_pred = []


    results = []


    latency_list = []



    for index,item in enumerate(questions):


        question = item["question"]

        true_label = item["answer"]



        print(
            "\nProcessing:",
            index+1,
            "/",
            len(questions)
        )


        start = time.time()



        try:


            response = rag_answer(
                question,
                mode="classification"
            )


            prediction = extract_label(
                response
            )



        except Exception as e:

            error_text = str(e).lower().replace("-", " ")

            is_oom = (
                "out of memory" in error_text
                or "cudamalloc" in error_text
                or "cuda error" in error_text
                or "unable to allocate" in error_text
                or "failed to allocate" in error_text
                or "bad_alloc" in error_text
            )

            if is_oom:
                print(
                    "OOM detected, waiting 5s and retrying once "
                    "with the same question:",
                    e
                )

                time.sleep(5)

                try:
                    response = rag_answer(question)
                    prediction = extract_label(response)

                except Exception as e2:
                    print("Retry failed:", e2)
                    response = "ERROR (OOM, retry failed): " + str(e2)
                    prediction = "maybe"

            else:

                print(
                    "Error:",
                    e
                )

                response = str(e)

                prediction = "maybe"



        latency = round(
            time.time()-start,
            2
        )


        latency_list.append(
            latency
        )



        y_true.append(
            true_label
        )


        y_pred.append(
            prediction
        )



        results.append(
            {
                "question":
                question,

                "actual":
                true_label,

                "prediction":
                prediction,

                "answer":
                response,

                "latency":
                latency
            }
        )



    # ==========================
    # Save Predictions
    # ==========================


    pd.DataFrame(
        results
    ).to_csv(
        OUTPUT_FILE,
        index=False
    )



    # ==========================
    # Metrics
    # ==========================


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    avg_latency = sum(
        latency_list
    ) / len(latency_list)



    print("\n======================")
    print("Evaluation Results")
    print("======================")

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1 Score:",
        f1
    )

    print(
        "Average Latency:",
        avg_latency
    )



    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred
        )
    )



    # ==========================
    # Save Metrics
    # ==========================


    metrics = pd.DataFrame(
        {
            "Metric":
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Average Latency"
            ],

            "Value":
            [
                accuracy,
                precision,
                recall,
                f1,
                avg_latency
            ]
        }
    )


    metrics.to_csv(
        RESULT_DIR +
        "/metrics.csv",
        index=False
    )



    # ==========================
    # Confusion Matrix
    # ==========================


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "yes",
            "no",
            "maybe"
        ]
    )


    plt.figure(
        figsize=(6,5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "yes",
            "no",
            "maybe"
        ],
        yticklabels=[
            "yes",
            "no",
            "maybe"
        ]
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.title(
        "Medical RAG Confusion Matrix"
    )


    plt.savefig(
        RESULT_DIR +
        "/confusion_matrix.png"
    )


    print(
        "\nFallback (no FINAL_ANSWER marker found) rate:",
        FALLBACK_COUNT,
        "/",
        len(questions),
        "=",
        round(FALLBACK_COUNT / len(questions) * 100, 1),
        "%"
    )

    if FALLBACK_COUNT / len(questions) > 0.15:
        print(
            "WARNING: high fallback rate. The model is not reliably "
            "following the FINAL_ANSWER format. Check "
            "evaluation/results/predictions.csv 'answer' column to "
            "see raw model outputs and adjust the prompt/num_predict."
        )


    print(
        "\nCompleted!"
    )

    print(
        "Results saved in:",
        RESULT_DIR
    )




# ==========================
# Main
# ==========================

if __name__ == "__main__":

    run_evaluation()