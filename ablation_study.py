"""
Ablation study: how much does each component of the optimized pipeline
actually contribute?

THE LADDER (each stage adds one thing to the stage above it)

    A  LLM only          qwen2.5:7b, NO retrieval at all
    B  + naive RAG       qwen2.5:7b + bge embeddings + dense retrieval
    C  + optimized       qwen2.5:7b + minilm + hybrid RRF retrieval
                         (the best combo from the embedding and retrieval
                          strategy benchmarks)
    D  + fine-tuned LLM  qwen2.5-medqa-lora + minilm + hybrid   <- full system

Consecutive stages are compared with McNemar's exact test, which isolates
each component's contribution:

    A -> B   does retrieval help at all?
    B -> C   did optimizing the retriever matter?
    C -> D   did fine-tuning the generator matter?

REUSE: stages C and D were already measured by paired_comparison.py under
greedy decoding -- C is its base-model run and D is its fine-tuned run, on
the identical 500-question sample with identical decoding. This script reads
those CSVs instead of re-running them, so only A and B are new work
(2 x 1000 questions instead of 4 x 1000).

Decoding is greedy (temperature=0, seed=42) for every stage, matching
paired_comparison.py, so all four stages are directly comparable.

Run from the project root:
    python ablation_study.py
"""

import json
import os
import random
import time

import pandas as pd
from scipy.stats import binomtest

import retriever
import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


# ==========================
# Deterministic decoding -- must match paired_comparison.py exactly
# ==========================

rag_pipeline.OLLAMA_OPTIONS.update({
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "repeat_penalty": 1.0,
})

RUN_TAG = "greedy"


# ==========================
# Configuration
# ==========================

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

SAMPLE_LIMIT = 500
RANDOM_SEED = 42
TOP_K = 8
NUM_PREDICT = 150
CHECKPOINT_EVERY = 10

# stage_id -> (label, llm, embedding_model, retrieval_mode)
# embedding_model=None means NO retrieval (stage A).
STAGES = [
    ("A", "LLM only (no RAG)",          "qwen2.5:7b",          None,     None),
    ("B", "+ naive RAG (bge, dense)",   "qwen2.5:7b",          "bge",    "dense"),
    ("C", "+ optimized retrieval",      "qwen2.5:7b",          "minilm", "hybrid"),
    ("D", "+ fine-tuned LLM",           "qwen2.5-medqa-lora",  "minilm", "hybrid"),
]

# Stages already measured by paired_comparison.py (greedy). Reused as-is.
REUSE = {
    "C": "paired_{dataset}_qwen2.5_7b_greedy.csv",
    "D": "paired_{dataset}_qwen2.5-medqa-lora_greedy.csv",
}


def questions_for(dataset_key):

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(questions):
        rng = random.Random(RANDOM_SEED)
        questions = rng.sample(questions, SAMPLE_LIMIT)

    return questions


# ==========================
# Prompts
# ==========================

def build_no_context_prompt(question, options):
    """Stage A prompt: identical instruction wording and identical
    FINAL_ANSWER format to build_mcqa_prompt, with ONLY the reference-material
    block removed. Keeping the rest word-for-word means the A -> B comparison
    isolates the presence of retrieved context, not a change of prompt style.

    Note this is deliberately NOT 'RAG with an empty context string' -- that
    would leave a dangling 'Reference Material:' header and instructions to
    use material that isn't there, which confuses instruction-tuned models."""

    options_text = "\n".join(
        f"{letter}) {text}" for letter, text in sorted(options.items())
    )

    return f"""

You are a medical exam assistant.


========================

Question:

{question}

Options:

{options_text}


Instructions:

Rely on your own medical knowledge to answer.

First give a SHORT explanation in ONE sentence only (max ~25 words).
Then, on a new final line, output EXACTLY in this format (replace
<letter> with A, B, C, or D), with nothing else after it:

FINAL_ANSWER: <letter>

Answer:

"""


# ==========================
# Retrieval caching (per embedding+mode combination)
# ==========================

def build_contexts(dataset_key, questions, embedding_model, mode):

    if embedding_model is None:
        return None

    # paired_comparison.py already cached minilm+hybrid under this name.
    if embedding_model == "minilm" and mode == "hybrid":
        path = os.path.join(RESULT_DIR, f"paired_contexts_{dataset_key}.json")
    else:
        path = os.path.join(
            RESULT_DIR, f"contexts_{dataset_key}_{embedding_model}_{mode}.json"
        )

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        if len(contexts) == len(questions):
            print(f"    using cached contexts: {os.path.basename(path)}")
            return contexts
        print("    cached contexts stale -- rebuilding")

    contexts = []

    for i, item in enumerate(questions):
        print(f"    retrieving {i + 1}/{len(questions)}", end="\r")

        results = retriever.retrieve_context(
            item["question"], k=TOP_K, mode=mode, model_key=embedding_model
        )

        chunks = [r["chunk"] for r in results]
        contexts.append("\n\n".join(chunks) if chunks
                        else "(no reference material retrieved)")

    print()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(contexts, f)

    return contexts


# ==========================
# Run one stage
# ==========================

def run_stage(stage_id, label, llm, embedding_model, mode, dataset_key, questions):

    # Reuse an already-measured stage if its CSV is present.
    if stage_id in REUSE:
        reuse_path = os.path.join(
            RESULT_DIR, REUSE[stage_id].format(dataset=dataset_key)
        )
        if os.path.exists(reuse_path):
            df = pd.read_csv(reuse_path)
            if len(df) == len(questions):
                print(f"  [{stage_id}] reusing {os.path.basename(reuse_path)}")
                return df
            print(f"  [{stage_id}] {os.path.basename(reuse_path)} has "
                  f"{len(df)} rows, expected {len(questions)} -- re-running")

    path = os.path.join(
        RESULT_DIR, f"ablation_{dataset_key}_{stage_id}_{RUN_TAG}.csv"
    )

    rows = []
    start_at = 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start_at = len(rows)
        if start_at >= len(questions):
            print(f"  [{stage_id}] already complete ({start_at})")
            return pd.DataFrame(rows)
        print(f"  [{stage_id}] resuming at {start_at}")

    contexts = build_contexts(dataset_key, questions, embedding_model, mode)

    for i in range(start_at, len(questions)):

        item = questions[i]

        print(f"  [{stage_id} {label}] {i + 1}/{len(questions)}", end="\r")

        if contexts is None:
            prompt = build_no_context_prompt(item["question"], item["options"])
        else:
            prompt = build_mcqa_prompt(
                contexts[i], item["question"], item["options"]
            )

        t0 = time.time()

        try:
            response = generate_answer(prompt, model=llm, num_predict=NUM_PREDICT)
            prediction = extract_choice(response)
        except Exception as e:
            print(f"\n  ERROR on {i + 1}: {e}")
            prediction = None

        rows.append({
            "index": i,
            "true_answer": item["answer_idx"],
            "prediction": prediction if prediction is not None else "",
            "unparseable": prediction is None,
            "correct": (prediction == item["answer_idx"]),
            "latency": round(time.time() - t0, 3),
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(questions) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


# ==========================
# McNemar between consecutive stages
# ==========================

def compare(prev_df, curr_df, prev_id, curr_id, n):

    prev_ok = prev_df["correct"].astype(bool).values
    curr_ok = curr_df["correct"].astype(bool).values

    gained = int((curr_ok & ~prev_ok).sum())
    lost = int((~curr_ok & prev_ok).sum())

    p = binomtest(gained, gained + lost, 0.5).pvalue if (gained + lost) else 1.0

    return gained, lost, p


# ==========================
# Main
# ==========================

def main():

    print("=" * 78)
    print("ABLATION STUDY -- greedy decoding (temperature=0, seed=42)")
    print("=" * 78)

    all_rows = []

    for dataset_key in DATASET_FILES:

        print(f"\n{'=' * 78}")
        print(f"Dataset: {dataset_key}")
        print("=" * 78)

        questions = questions_for(dataset_key)

        results = {}

        for stage_id, label, llm, emb, mode in STAGES:
            results[stage_id] = run_stage(
                stage_id, label, llm, emb, mode, dataset_key, questions
            )

        print(f"\n{'-' * 78}")
        print(f"{dataset_key} -- stage results")
        print("-" * 78)

        prev_id = None

        for stage_id, label, llm, emb, mode in STAGES:

            df = results[stage_id]
            acc = df["correct"].astype(bool).mean()
            lat = df["latency"].mean()

            row = {
                "dataset": dataset_key,
                "stage": stage_id,
                "config": label,
                "llm": llm,
                "embedding": emb or "-",
                "retrieval": mode or "-",
                "n": len(df),
                "accuracy": round(float(acc), 4),
                "latency_sec": round(float(lat), 3),
            }

            if prev_id is not None:

                prev_df = results[prev_id]
                prev_acc = prev_df["correct"].astype(bool).mean()

                gained, lost, p = compare(prev_df, df, prev_id, stage_id, len(df))

                row.update({
                    "vs_prev": f"{prev_id}->{stage_id}",
                    "delta_pp": round(float((acc - prev_acc) * 100), 2),
                    "gained": gained,
                    "lost": lost,
                    "mcnemar_p": round(float(p), 5),
                    "significant": bool(p < 0.05),
                })

                print(f"  {stage_id} {label:32s} acc={acc:.1%}  "
                      f"{prev_id}->{stage_id}: {(acc - prev_acc) * 100:+.1f} pp  "
                      f"(+{gained}/-{lost}, p={p:.4f})"
                      f"{'  SIGNIFICANT' if p < 0.05 else ''}")
            else:
                print(f"  {stage_id} {label:32s} acc={acc:.1%}  (baseline)")

            all_rows.append(row)
            prev_id = stage_id

        # Overall A -> D
        a_df, d_df = results["A"], results["D"]
        a_acc = a_df["correct"].astype(bool).mean()
        d_acc = d_df["correct"].astype(bool).mean()
        gained, lost, p = compare(a_df, d_df, "A", "D", len(d_df))

        print(f"\n  END TO END  A->D: {(d_acc - a_acc) * 100:+.1f} pp  "
              f"(+{gained}/-{lost}, p={p:.5f})"
              f"{'  SIGNIFICANT' if p < 0.05 else ''}")

        all_rows.append({
            "dataset": dataset_key, "stage": "A->D", "config": "END TO END",
            "llm": "-", "embedding": "-", "retrieval": "-", "n": len(d_df),
            "accuracy": round(float(d_acc), 4), "latency_sec": None,
            "vs_prev": "A->D",
            "delta_pp": round(float((d_acc - a_acc) * 100), 2),
            "gained": gained, "lost": lost,
            "mcnemar_p": round(float(p), 5), "significant": bool(p < 0.05),
        })

    out = os.path.join(RESULT_DIR, f"ablation_summary_{RUN_TAG}.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(pd.DataFrame(all_rows).to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
