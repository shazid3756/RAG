"""
Paired before/after comparison: base qwen2.5:7b vs fine-tuned
qwen2.5-medqa-lora, with per-question predictions saved so McNemar's test
can be run.

WHY THIS EXISTS
---------------
llm_benchmark.py reports only aggregate accuracy. On those aggregates the
deltas (-1.4 pp MedQA, +4.2 pp MedMCQA) are not significant on an UNPAIRED
two-proportion test (95% CI +/- 6.2 pp). But the design is PAIRED -- both
models answered the identical 500 questions -- and a base model and its own
fine-tune agree on most questions, so the paired test is far more powerful.
McNemar's test only looks at the questions where the two models DISAGREE,
which is exactly where the fine-tuning effect lives.

DESIGN NOTES
------------
1. Retrieval is computed ONCE per question and cached to disk, then reused
   for both models. This guarantees byte-identical context (a stronger
   pairing than merely "same retrieval settings") and halves the work.
2. Models are run in separate full passes, NOT alternated per question.
   Alternating would force Ollama to swap a 4.7GB model in and out of VRAM
   on every single question (~20-30s each) instead of twice total.
3. Everything checkpoints to disk after each question. Power outages have
   interrupted this project before; re-running resumes where it stopped.

Run from the project root:
    python paired_comparison.py

Then read the printed McNemar results. Re-running after completion just
re-prints the analysis without re-querying anything.
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
# Decoding mode
# ==========================
#
# The first run of this script used each model's OWN default sampling
# settings, because rag_pipeline.OLLAMA_OPTIONS sets only num_ctx /
# num_predict / num_gpu -- no temperature or seed. Two problems followed:
#
#   1. Results were not reproducible. Base MedMCQA measured 49.8% in
#      llm_benchmark.py and 47.6% here; the fine-tuned model 54.0% then
#      55.6%. The measured effect moved between +4.2 and +8.0 pp.
#   2. The two models were not necessarily decoding under the same rules.
#      Ollama's library qwen2.5:7b carries Qwen's own PARAMETER defaults,
#      while our hand-written Modelfile sets none, so the fine-tuned model
#      fell back to Ollama's built-in defaults instead.
#
# Options sent over the HTTP API override a model's Modelfile PARAMETERs,
# so setting them here makes decoding identical for both models whatever
# their Modelfile says. temperature=0 is greedy decoding: the same input
# always yields the same output, so run-to-run noise is eliminated and
# every discordant question reflects a real behavioural difference between
# the two models rather than a sampling accident.
#
# Set DETERMINISTIC = False to reproduce the original default-sampling run.

DETERMINISTIC = True

RUN_TAG = "greedy" if DETERMINISTIC else "default"

if DETERMINISTIC:
    rag_pipeline.OLLAMA_OPTIONS.update({
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 42,
        "repeat_penalty": 1.0,
    })


# ==========================
# Configuration -- mirrors llm_benchmark.py exactly
# ==========================

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

BASE_MODEL = "qwen2.5:7b"
FT_MODEL = "qwen2.5-medqa-lora"

FIXED_EMBEDDING_MODEL = "minilm"
RETRIEVAL_MODE = "hybrid"
TOP_K = 8

SAMPLE_LIMIT = 500
RANDOM_SEED = 42

NUM_PREDICT = 150          # same cap llm_benchmark.py uses
CHECKPOINT_EVERY = 10


def questions_for(dataset_key):
    """Identical sampling to llm_benchmark.load_questions."""

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(questions):
        rng = random.Random(RANDOM_SEED)
        questions = rng.sample(questions, SAMPLE_LIMIT)

    return questions


# ==========================
# Phase 1 -- cache retrieval
# ==========================

def build_contexts(dataset_key, questions):
    """Retrieve once per question; cache to disk. Reused by BOTH models so
    the pairing is on identical context, not just identical settings."""

    path = os.path.join(RESULT_DIR, f"paired_contexts_{dataset_key}.json")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        if len(contexts) == len(questions):
            print(f"  [{dataset_key}] using cached contexts ({len(contexts)})")
            return contexts
        print(f"  [{dataset_key}] cached contexts stale -- rebuilding")

    contexts = []

    for i, item in enumerate(questions):
        print(f"  [{dataset_key}] retrieving {i + 1}/{len(questions)}", end="\r")

        results = retriever.retrieve_context(
            item["question"], k=TOP_K, mode=RETRIEVAL_MODE,
            model_key=FIXED_EMBEDDING_MODEL,
        )

        chunks = [r["chunk"] for r in results]
        contexts.append("\n\n".join(chunks) if chunks
                        else "(no reference material retrieved)")

    print()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(contexts, f)

    return contexts


# ==========================
# Phase 2 -- one full pass per model
# ==========================

def run_model(model, dataset_key, questions, contexts):
    """One uninterrupted pass so Ollama loads the model once. Checkpoints
    to CSV so a power cut costs at most CHECKPOINT_EVERY questions."""

    safe_model = model.replace(":", "_").replace("/", "_")

    path = os.path.join(
        RESULT_DIR, f"paired_{dataset_key}_{safe_model}_{RUN_TAG}.csv"
    )

    rows = []
    start_at = 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start_at = len(rows)
        if start_at >= len(questions):
            print(f"  [{model} / {dataset_key}] already complete ({start_at})")
            return pd.DataFrame(rows)
        print(f"  [{model} / {dataset_key}] resuming at {start_at}")

    for i in range(start_at, len(questions)):

        item = questions[i]

        print(f"  [{model} / {dataset_key}] {i + 1}/{len(questions)}", end="\r")

        prompt = build_mcqa_prompt(contexts[i], item["question"], item["options"])

        t0 = time.time()

        try:
            response = generate_answer(prompt, model=model, num_predict=NUM_PREDICT)
            prediction = extract_choice(response)
        except Exception as e:
            print(f"\n  ERROR on {i + 1}: {e}")
            response = str(e)
            prediction = None

        latency = time.time() - t0

        rows.append({
            "index": i,
            "true_answer": item["answer_idx"],
            "prediction": prediction if prediction is not None else "",
            "unparseable": prediction is None,
            "correct": (prediction == item["answer_idx"]),
            "latency": round(latency, 3),
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(questions) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


# ==========================
# Phase 3 -- McNemar
# ==========================

def mcnemar(base_df, ft_df, dataset_key):

    n = len(base_df)

    base_ok = base_df["correct"].astype(bool).values
    ft_ok = ft_df["correct"].astype(bool).values

    # b: fine-tune fixed one the base got wrong
    # c: fine-tune broke one the base got right
    b = int((ft_ok & ~base_ok).sum())
    c = int((~ft_ok & base_ok).sum())
    both_right = int((ft_ok & base_ok).sum())
    both_wrong = int((~ft_ok & ~base_ok).sum())

    acc_base = base_ok.mean()
    acc_ft = ft_ok.mean()
    diff = acc_ft - acc_base

    # Exact McNemar: under H0 the discordant pairs split 50/50.
    if b + c > 0:
        p = binomtest(b, b + c, 0.5).pvalue
    else:
        p = 1.0

    # Paired 95% CI on the difference of proportions.
    se = ((b + c) ** 0.5) / n
    ci = 1.96 * se

    print("\n" + "=" * 70)
    print(f"McNemar's test -- {dataset_key}  (n={n}, paired)")
    print("=" * 70)
    print(f"  base {BASE_MODEL:24s} accuracy = {acc_base:.1%}")
    print(f"  fine-tuned {FT_MODEL:18s} accuracy = {acc_ft:.1%}")
    print(f"  difference = {diff * 100:+.1f} pp   (paired 95% CI +/- {ci * 100:.1f} pp)")
    print()
    print(f"  both correct        : {both_right}")
    print(f"  both wrong          : {both_wrong}")
    print(f"  fine-tune FIXED     : {b}   (base wrong -> fine-tune right)")
    print(f"  fine-tune BROKE     : {c}   (base right -> fine-tune wrong)")
    print(f"  discordant total    : {b + c}")
    print()
    print(f"  exact McNemar p = {p:.4f}")

    if p < 0.05:
        direction = "IMPROVED" if b > c else "DEGRADED"
        print(f"  -> SIGNIFICANT at alpha=0.05. Fine-tuning {direction} accuracy.")
    else:
        print(f"  -> NOT significant at alpha=0.05. Report as 'no significant")
        print(f"     change in accuracy' -- do not claim an improvement.")

    print(f"\n  mean latency: base {base_df['latency'].mean():.2f}s  "
          f"fine-tuned {ft_df['latency'].mean():.2f}s")

    return {
        "dataset": dataset_key,
        "n": n,
        "acc_base": round(float(acc_base), 4),
        "acc_finetuned": round(float(acc_ft), 4),
        "diff_pp": round(float(diff * 100), 2),
        "paired_ci_pp": round(float(ci * 100), 2),
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "finetune_fixed": b,
        "finetune_broke": c,
        "mcnemar_p": round(float(p), 5),
        "significant_at_05": bool(p < 0.05),
        "latency_base": round(float(base_df["latency"].mean()), 3),
        "latency_finetuned": round(float(ft_df["latency"].mean()), 3),
    }


# ==========================
# Main
# ==========================

def main():

    print("=" * 70)
    print(f"Decoding mode: {RUN_TAG.upper()}")
    if DETERMINISTIC:
        print("  temperature=0, top_p=1.0, top_k=0, seed=42, repeat_penalty=1.0")
        print("  -> identical decoding for both models; output is reproducible")
    else:
        print("  each model's own Modelfile defaults (NOT reproducible)")
    print("=" * 70)

    summary = []

    for dataset_key in DATASET_FILES:

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_key}")
        print("=" * 70)

        questions = questions_for(dataset_key)

        contexts = build_contexts(dataset_key, questions)

        # Base first, then fine-tuned: two model loads total, not 1000.
        base_df = run_model(BASE_MODEL, dataset_key, questions, contexts)
        ft_df = run_model(FT_MODEL, dataset_key, questions, contexts)

        summary.append(mcnemar(base_df, ft_df, dataset_key))

    out = os.path.join(RESULT_DIR, f"paired_comparison_summary_{RUN_TAG}.csv")
    pd.DataFrame(summary).to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
