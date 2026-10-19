"""
Adaptive (selective) retrieval: inject retrieved context ONLY when the
retriever is confident the corpus actually has something relevant.

MOTIVATION
----------
ablation_study.py showed that adding retrieval REDUCES accuracy
(A->B pooled: 100 gained / 134 lost, p=0.031), because with 100K of PubMed's
23.9M abstracts the retrieved passages are usually irrelevant, and irrelevant
context distracts the model away from knowledge it already has.

If that diagnosis is right, then gating on retrieval confidence should
recover the loss: keep context when it is likely relevant, drop it otherwise.

WHY THIS IS ALMOST FREE
-----------------------
Both arms have already been generated under greedy decoding on the identical
500-question seed-42 sample:
    arm NO_CTX  = ablation stage A   (no retrieval, base LLM)
    arm CTX     = ablation stage C   (minilm+hybrid, base LLM)   [= paired base]
    arm CTX_FT  = ablation stage D   (minilm+hybrid, fine-tuned) [= paired FT]

A gating policy just picks, per question, which arm's answer to take. So no
new generation is needed -- only the retrieval confidence score per question,
which needs no LLM at all.

WHICH SCORE TO GATE ON
----------------------
NOT the hybrid RRF score. RRF is rank-based: its top result always scores
1/(60+1) whether or not it is relevant, so it carries no relevance
information. We use the DENSE cosine similarity instead. embedding_step4.py
normalized the embeddings, so FAISS inner product IS cosine, which is a
genuine relevance signal.

HONEST THRESHOLD SELECTION
--------------------------
Choosing the threshold on the same questions we report on would be
overfitting. Each dataset's 500 questions are split in half by index parity:
even indices tune the threshold, odd indices report it. The held-out numbers
are the ones to publish.

Run from the project root:
    python adaptive_retrieval.py
"""

import json
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import retriever


DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"

SAMPLE_LIMIT = 500
RANDOM_SEED = 42
TOP_K = 8

GATE_EMBEDDING = "minilm"     # same embedding the context arm used

# Per-question prediction files produced by earlier scripts.
ARM_FILES = {
    "NO_CTX": "ablation_{dataset}_A_greedy.csv",
    "CTX":    "paired_{dataset}_qwen2.5_7b_greedy.csv",
    "CTX_FT": "paired_{dataset}_qwen2.5-medqa-lora_greedy.csv",
}


def questions_for(dataset_key):

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        questions = json.load(f)

    if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(questions):
        rng = random.Random(RANDOM_SEED)
        questions = rng.sample(questions, SAMPLE_LIMIT)

    return questions


# ==========================
# Retrieval confidence (no LLM involved)
# ==========================

def gate_scores(dataset_key, questions):
    """Dense cosine similarity of the retrieved passages, per question.
    Cached -- this is the only new computation the whole script needs."""

    path = os.path.join(RESULT_DIR, f"gate_scores_{dataset_key}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) == len(questions):
            print(f"  using cached gate scores ({len(df)})")
            return df
        print("  cached gate scores stale -- rebuilding")

    rows = []

    for i, item in enumerate(questions):

        print(f"  scoring {i + 1}/{len(questions)}", end="\r")

        results = retriever.retrieve_context(
            item["question"], k=TOP_K, mode="dense", model_key=GATE_EMBEDDING
        )

        scores = [r["score"] for r in results] or [0.0]

        rows.append({
            "index": i,
            "top1": float(max(scores)),
            "mean_topk": float(np.mean(scores)),
        })

    print()

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    return df


# ==========================
# Gating simulation
# ==========================

def gated_correct(no_ctx_ok, ctx_ok, scores, threshold):
    """Take the context arm's answer where score >= threshold, else the
    no-context arm's answer."""

    use_ctx = scores >= threshold

    return np.where(use_ctx, ctx_ok, no_ctx_ok), use_ctx


def evaluate(dataset_key, ctx_arm):

    questions = questions_for(dataset_key)

    print(f"\n  gate signal: dense top-1 cosine ({GATE_EMBEDDING})")

    gates = gate_scores(dataset_key, questions)

    arms = {}

    for name, pattern in ARM_FILES.items():

        path = os.path.join(RESULT_DIR, pattern.format(dataset=dataset_key))

        if not os.path.exists(path):
            print(f"  MISSING: {os.path.basename(path)} -- run ablation_study.py "
                  f"and paired_comparison.py first")
            return None

        df = pd.read_csv(path)

        if len(df) != len(questions):
            print(f"  {os.path.basename(path)} has {len(df)} rows, "
                  f"expected {len(questions)}")
            return None

        arms[name] = df["correct"].astype(bool).values

    no_ctx_ok = arms["NO_CTX"]
    ctx_ok = arms[ctx_arm]

    scores = gates["top1"].values

    n = len(scores)

    # ---- headroom check: what would PERFECT gating achieve? ----
    oracle_ok = no_ctx_ok | ctx_ok

    print(f"\n  {'-' * 66}")
    print(f"  {dataset_key}  |  context arm = {ctx_arm}  (n={n})")
    print(f"  {'-' * 66}")
    print(f"    no context          : {no_ctx_ok.mean():.1%}")
    print(f"    always context      : {ctx_ok.mean():.1%}")
    print(f"    ORACLE gating       : {oracle_ok.mean():.1%}   "
          f"<- ceiling for ANY gating policy")

    headroom = oracle_ok.mean() - max(no_ctx_ok.mean(), ctx_ok.mean())
    print(f"    headroom over best fixed policy: {headroom * 100:+.1f} pp")

    if headroom < 0.01:
        print("    -> no room for gating to help. Stop here.")

    # ---- honest threshold selection: tune on even indices, report on odd ----
    idx = np.arange(n)
    tune = idx % 2 == 0
    test = ~tune

    candidates = np.quantile(scores, np.linspace(0.0, 1.0, 41))

    best_t, best_acc = None, -1.0

    for t in candidates:
        acc = gated_correct(no_ctx_ok, ctx_ok, scores, t)[0][tune].mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    gated_ok, use_ctx = gated_correct(no_ctx_ok, ctx_ok, scores, best_t)

    print(f"\n    threshold tuned on even indices: {best_t:.4f}")
    print(f"      (context used on {use_ctx.mean():.0%} of questions)")
    print(f"    tune-half accuracy  : {gated_ok[tune].mean():.1%}")

    print(f"\n    HELD-OUT (odd indices, n={test.sum()}):")
    print(f"      no context        : {no_ctx_ok[test].mean():.1%}")
    print(f"      always context    : {ctx_ok[test].mean():.1%}")
    print(f"      adaptive gating   : {gated_ok[test].mean():.1%}")

    # McNemar of gating vs the better fixed policy, on held-out only.
    if no_ctx_ok[test].mean() >= ctx_ok[test].mean():
        base_name, base_ok = "no context", no_ctx_ok
    else:
        base_name, base_ok = "always context", ctx_ok

    g = int((gated_ok[test] & ~base_ok[test]).sum())
    l = int((~gated_ok[test] & base_ok[test]).sum())
    p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0

    print(f"      vs {base_name}: {(gated_ok[test].mean() - base_ok[test].mean()) * 100:+.1f} pp"
          f"  (+{g}/-{l}, p={p:.4f})"
          f"{'  SIGNIFICANT' if p < 0.05 else '  not significant'}")

    return {
        "dataset": dataset_key,
        "context_arm": ctx_arm,
        "n": n,
        "acc_no_context": round(float(no_ctx_ok.mean()), 4),
        "acc_always_context": round(float(ctx_ok.mean()), 4),
        "acc_oracle_gating": round(float(oracle_ok.mean()), 4),
        "headroom_pp": round(float(headroom * 100), 2),
        "threshold": round(float(best_t), 4),
        "context_used_frac": round(float(use_ctx.mean()), 4),
        "heldout_n": int(test.sum()),
        "heldout_no_context": round(float(no_ctx_ok[test].mean()), 4),
        "heldout_always_context": round(float(ctx_ok[test].mean()), 4),
        "heldout_adaptive": round(float(gated_ok[test].mean()), 4),
        "heldout_vs_best_fixed_pp": round(
            float((gated_ok[test].mean() - base_ok[test].mean()) * 100), 2),
        "mcnemar_p": round(float(p), 5),
        "significant": bool(p < 0.05),
    }


def main():

    print("=" * 72)
    print("ADAPTIVE RETRIEVAL -- gate context on dense cosine confidence")
    print("=" * 72)

    rows = []

    for dataset_key in DATASET_FILES:
        for ctx_arm in ("CTX", "CTX_FT"):
            r = evaluate(dataset_key, ctx_arm)
            if r:
                rows.append(r)

    if not rows:
        return

    out = os.path.join(RESULT_DIR, "adaptive_retrieval_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved to: {out}")
    print("\nRead the ORACLE row first: if oracle gating is barely above the")
    print("best fixed policy, no threshold can help and this direction is dead.")


if __name__ == "__main__":
    main()
