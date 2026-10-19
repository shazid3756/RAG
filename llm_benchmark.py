"""
LLM Benchmark: compares generation-LLM choice on MedQA / MedMCQA,
with RETRIEVAL held fixed at the best combo found so far
(embedding=minilm, mode=hybrid — see embedding_benchmark_mcqa.py and
retrieval_strategy_benchmark.py) so only the generation LLM varies.

Models tested (Ollama tags):
    qwen2.5:7b                      baseline (already your default)
    cniongolo/biomistral            BioMistral-7B (community upload)
    meditron:7b                     Meditron-7B (official)
    medllama2:7b                    MedLlama2-7B — substitute for BioGPT
                                     (BioGPT has no Ollama-compatible
                                     instruction-following build)
    llama3.1:8b                     Llama-3.1-8B-Instruct (official)
    phi3.5                          Phi-3.5-mini (official)
    koesn/llama3-openbiollm-8b      OpenBioLLM-8B — substitute for
                                     ClinicalCamel (only a gated 70B
                                     HF checkpoint exists, too large
                                     for 12GB VRAM)

IMPORTANT: pull every model BEFORE running this script:
    ollama pull qwen2.5:7b
    ollama pull cniongolo/biomistral
    ollama pull meditron:7b
    ollama pull medllama2:7b
    ollama pull llama3.1:8b
    ollama pull phi3.5
    ollama pull koesn/llama3-openbiollm-8b

Per-(llm, dataset) resume: results saved immediately, already-done
combinations skipped on re-run. Ollama itself handles loading/
unloading each model from VRAM as the requested model changes between
calls — no manual memory management needed on our side for this one.

Run from the project root:
    python llm_benchmark.py
"""

import json
import os
import random
import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import retriever
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice, extract_bare_letter


# ==========================
# Configuration
# ==========================

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_TABLE = os.path.join(RESULT_DIR, "llm_benchmark.csv")

# Best retrieval combo found so far — held fixed.
FIXED_EMBEDDING_MODEL = "minilm"
RETRIEVAL_MODE = "hybrid"
TOP_K = 8

# BioMistral and Meditron were EXCLUDED after benchmarking: both
# showed ~93-95% fallback rate (model unable to produce a parseable
# answer) even after prompt refinement and a minimal-format retry.
# Research literature confirms why: both are primarily CONTINUED-
# PRETRAINING models on medical text, not heavily instruction-tuned
# for structured/chat-style output — this is an inherent property of
# these specific model families, not a fixable Ollama build issue.
# See docx report for the excluded-model note.
#
# Replaced with two well-instruction-tuned models instead of just
# dropping the "7 models" count:
#   - mistral:7b-instruct  — notably, this is literally the BASE
#     model BioMistral was continued-pretrained from. Comparing the
#     two head-to-head lets the paper discuss a real trade-off:
#     domain continued-pretraining vs. instruction-following ability
#     (a nice parallel to the "general embeddings > domain embeddings"
#     finding already in Section 5.5 of the report).
#   - gemma2:9b-instruct-q4_0 — strong, reliably instruction-tuned
#     general model from a different lineage than Qwen/Llama/Mistral.
# Phi-3.5-mini was EXCLUDED after benchmarking: despite being
# generally regarded as well instruction-tuned, it showed 82-87%
# fallback rate in this RAG-context pipeline (long retrieved passages
# + structured-output instructions). Combined with BioMistral and
# Meditron, this brings the total excluded-for-unreliability count to
# 3 of the original 9 candidate models — itself a notable finding on
# instruction-following reliability under long-context RAG prompting.
#
# Replaced with llama3.2:3b — Meta's own documentation reports it
# outperforming Phi-3.5-mini on several benchmarks, and it fills the
# same "small, efficient, low-latency" role in the comparison.
#
# FINE-TUNED MODEL (added for the before/after QLoRA comparison):
# qwen2.5-medqa-lora is qwen2.5:7b + our QLoRA adapter (r=16/alpha=32,
# trained on 2,700 RAG-formatted MedQA/MedMCQA *train*-split examples,
# best checkpoint eval_loss=0.6042 at epoch 1.04, early-stopped),
# merged into full weights and re-quantized to q4_0 to match the
# quantization of the other Ollama models in this table.
#
# It is listed LAST on purpose: the per-(llm, dataset) resume logic
# below skips the 7 combinations already in llm_benchmark.csv, so a
# re-run evaluates ONLY this new model — on the identical 500-question
# seed-42 sample, identical minilm+hybrid retrieval, and identical
# prompt/decoding settings as its own base model qwen2.5:7b
# (56.6% MedQA / 49.8% MedMCQA). That makes the delta attributable to
# fine-tuning alone, which is the entire point of the comparison.
LLMS_TO_TEST = [
    "qwen2.5:7b",
    "mistral:7b-instruct",
    "gemma2:9b-instruct-q4_0",
    "medllama2:7b",
    "llama3.1:8b",
    "llama3.2:3b",
    "koesn/llama3-openbiollm-8b",
    "qwen2.5-medqa-lora",
]

# Full-dataset runs take too long across 7 LLMs (~38K calls). Using a
# fixed-seed RANDOM sample instead of the full set keeps the run
# tractable while still giving a statistically meaningful comparison:
# n=500 gives a margin of error of roughly +/-4.4% at 95% confidence,
# which is tight enough to rank 7 models against each other.
#
# IMPORTANT: this is a random sample (seeded for reproducibility), NOT
# the first N questions in file order — MedMCQA in particular is
# grouped by subject in its original order, so naive slicing would
# heavily bias the sample toward whichever subjects happen to appear
# first in the file.
SAMPLE_LIMIT = 500
RANDOM_SEED = 42


# ==========================
# Helpers
# ==========================

def load_questions(dataset_key):

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(questions):
        # Fixed seed -> same 500 questions every run, for every model,
        # so all 7 LLMs are compared on IDENTICAL questions.
        rng = random.Random(RANDOM_SEED)
        questions = rng.sample(questions, SAMPLE_LIMIT)

    return questions


def load_completed():
    """Returns a set of (llm, dataset) tuples already in the CSV."""

    if not os.path.exists(OUTPUT_TABLE):
        return set()

    try:
        existing = pd.read_csv(OUTPUT_TABLE)
        return set(zip(existing["llm"], existing["dataset"]))
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
# Per (llm, dataset) evaluation
# ==========================

def build_strict_letter_prompt(question, options):
    """
    Minimal one-shot fallback prompt used ONLY when a model's first
    response doesn't contain a parseable FINAL_ANSWER marker. Some
    smaller/older models (observed: meditron:7b, cniongolo/biomistral)
    don't reliably follow the longer instruction-format prompt, but
    will often answer correctly if asked for ONLY the letter with no
    other instructions competing for attention. Kept deliberately
    short — num_predict is capped very low for this call since we
    need at most one character back.
    """

    options_text = "\n".join(f"{letter}) {text}" for letter, text in sorted(options.items()))

    return f"""{question}

{options_text}

Reply with ONLY one letter: A, B, C, or D. No explanation, no other text.

Answer:"""


def evaluate(llm_model, dataset_key, questions):

    y_true = []
    y_pred = []
    latencies = []
    fallback_count = 0

    # Retrieval is IDENTICAL for every question regardless of which
    # LLM is being tested — compute it once per question, reuse the
    # retrieved context, only the generation call changes.
    for index, item in enumerate(questions):

        question = item["question"]
        options = item["options"]
        true_answer = item["answer_idx"]

        print(f"  [{llm_model} / {dataset_key}] {index + 1} / {len(questions)}", end="\r")

        start = time.time()

        try:

            results = retriever.retrieve_context(
                question, k=TOP_K, mode=RETRIEVAL_MODE, model_key=FIXED_EMBEDDING_MODEL
            )

            chunks = [r["chunk"] for r in results]

            context_text = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"

            prompt = build_mcqa_prompt(context_text, question, options)

            # Cap generation length — some models don't respect the
            # "one sentence" instruction and will otherwise ramble
            # through the full token budget every call, which was the
            # dominant cause of high per-question latency observed
            # with meditron:7b (~14s/question vs ~2.6s for qwen2.5).
            response = generate_answer(prompt, model=llm_model, num_predict=150)

            prediction = extract_choice(response)

            if prediction is None:

                # One cheap, minimal retry before giving up — often
                # recovers cases where the model answered correctly
                # but didn't follow the FINAL_ANSWER format the first
                # time (observed with meditron:7b, biomistral).
                strict_prompt = build_strict_letter_prompt(question, options)

                retry_response = generate_answer(
                    strict_prompt, model=llm_model, num_predict=20
                )

                prediction = extract_bare_letter(retry_response)

                if prediction is None:
                    fallback_count += 1
                    prediction = "A"

        except Exception as e:

            print(f"\n  [{llm_model}/{dataset_key}] ERROR on question {index + 1}: {e}")

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
        "llm": llm_model,
        "dataset": dataset_key,
        "embedding_model": FIXED_EMBEDDING_MODEL,
        "retrieval_mode": RETRIEVAL_MODE,
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

    print(f"LLMs to test: {LLMS_TO_TEST}")
    print(f"Fixed retrieval: embedding={FIXED_EMBEDDING_MODEL}, mode={RETRIEVAL_MODE}")
    print(f"Datasets: {list(DATASET_FILES.keys())} (sample size {SAMPLE_LIMIT} each)\n")

    for llm_model in LLMS_TO_TEST:

        for dataset_key, questions in questions_by_dataset.items():

            if (llm_model, dataset_key) in completed:
                print(f"Skipping ({llm_model}, {dataset_key}) — already done.")
                continue

            print(f"\n{'=' * 50}")
            print(f"LLM: {llm_model}  |  Dataset: {dataset_key}")
            print(f"{'=' * 50}")

            try:

                result = evaluate(llm_model, dataset_key, questions)

                append_result_to_csv(result)

                print(f"  -> accuracy={result['accuracy']}  f1={result['f1']}  "
                      f"avg_latency={result['avg_latency_sec']}s  "
                      f"fallback_rate={result['fallback_rate']}")

            except Exception as e:

                print(f"ERROR on ({llm_model}, {dataset_key}): {e}")
                print(
                    "This usually means the model wasn't pulled yet — "
                    f"run: ollama pull {llm_model}"
                )
                print("Continuing — re-run later to retry this combination.")

    if not os.path.exists(OUTPUT_TABLE):
        print("\nNo results completed.")
        return

    df = pd.read_csv(OUTPUT_TABLE).sort_values(["dataset", "accuracy"], ascending=[True, False])

    print("\n" + "=" * 70)
    print("LLM Benchmark on MedQA/MedMCQA — Summary")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nSaved to:", OUTPUT_TABLE)


if __name__ == "__main__":
    main()