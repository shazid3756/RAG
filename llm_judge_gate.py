"""
LLM-judged selective retrieval: ask the model itself whether the retrieved
passages are actually useful, and drop the context when it says no.

WHY
---
gate_signal_diagnostic.py showed retrieval confidence scores carry no
information about when context helps (all AUC ~ 0.5, best uncorrected
p=0.062 across 12 signals). But adaptive_retrieval.py showed the per-question
decision is worth 9-12 pp if made perfectly. So the decision matters; the
similarity score just cannot make it. This tests whether the LLM can.

METHOD
------
One extra, very short generation call per question:

    judge(context, question) -> YES / NO

Then the answer is taken from the CTX arm when the judge says YES, and from
the NO_CTX arm when it says NO. Both arms are already generated (greedy), so
this script only produces the judgements.

THE HONEST TEST
---------------
There is no threshold to tune here, so all 500 questions are reportable. But
the number that actually matters is the judge's skill on DISCORDANT
questions -- those where the two arms disagree and the gate can change the
outcome. On those, a useless judge scores 50%. That binomial test is the
result; overall accuracy can drift up or down for reasons unrelated to judge
quality (e.g. simply preferring the better fixed arm more often).

Run from the project root:
    python llm_judge_gate.py
"""

import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import rag_pipeline
from rag_pipeline import generate_answer


# Greedy, matching every other greedy run in this project.
rag_pipeline.OLLAMA_OPTIONS.update({
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "repeat_penalty": 1.0,
})


DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"

SAMPLE_LIMIT = 500
RANDOM_SEED = 42

JUDGE_MODEL = "qwen2.5:7b"     # the framework's own base LLM
JUDGE_NUM_PREDICT = 5          # one word is enough
CHECKPOINT_EVERY = 20

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


def build_judge_prompt(context_text, question):
    """Deliberately asks about SPECIFIC usefulness, not topical similarity --
    topical similarity is exactly what the cosine gate already measured and
    failed with. The 'same general topic is not enough' clause is the whole
    point of the prompt."""

    return f"""

You are assessing whether retrieved reference material is useful for
answering a specific medical exam question.


========================

Reference Material:

{context_text}


========================

Question:

{question}


========================

Does the reference material contain specific information that would help
answer THIS question correctly?

Say YES only if it addresses the actual clinical or biomedical content the
question turns on. Material that is merely about the same general topic,
or that is related but does not bear on what is being asked, is NOT useful.

Reply with exactly one word: YES or NO.

Answer:

"""


def parse_judgement(text):

    if not text:
        return None

    m = re.search(r"\b(YES|NO)\b", text.strip(), re.IGNORECASE)

    return m.group(1).upper() if m else None


def collect_judgements(dataset_key, questions, contexts):

    path = os.path.join(RESULT_DIR, f"judge_{dataset_key}.csv")

    rows = []
    start_at = 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start_at = len(rows)
        if start_at >= len(questions):
            print(f"  judgements already complete ({start_at})")
            return pd.DataFrame(rows)
        print(f"  resuming judgements at {start_at}")

    for i in range(start_at, len(questions)):

        print(f"  judging {i + 1}/{len(questions)}", end="\r")

        prompt = build_judge_prompt(contexts[i], questions[i]["question"])

        t0 = time.time()

        try:
            resp = generate_answer(
                prompt, model=JUDGE_MODEL, num_predict=JUDGE_NUM_PREDICT
            )
            verdict = parse_judgement(resp)
        except Exception as e:
            print(f"\n  ERROR on {i + 1}: {e}")
            verdict = None

        rows.append({
            "index": i,
            # Unparseable defaults to YES = keep context = the standard RAG
            # behaviour, so a judge failure degrades to the baseline system
            # rather than silently disabling retrieval.
            "verdict": verdict if verdict else "YES",
            "parsed": verdict is not None,
            "latency": round(time.time() - t0, 3),
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(questions) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


def analyse(dataset_key, judge_df, ctx_arm):

    arms = {}

    for name, pattern in ARM_FILES.items():
        p = os.path.join(RESULT_DIR, pattern.format(dataset=dataset_key))
        arms[name] = pd.read_csv(p)["correct"].astype(bool).values

    no_ctx = arms["NO_CTX"]
    ctx = arms[ctx_arm]

    use_ctx = (judge_df["verdict"].values == "YES")

    gated = np.where(use_ctx, ctx, no_ctx)

    n = len(gated)

    print(f"\n  {'-' * 66}")
    print(f"  {dataset_key}  |  context arm = {ctx_arm}  (n={n})")
    print(f"  {'-' * 66}")
    print(f"    judge said context useful on {use_ctx.mean():.0%} of questions")
    print(f"    judge parse failures        : {(~judge_df['parsed']).sum()}")
    print()
    print(f"    no context      : {no_ctx.mean():.1%}")
    print(f"    always context  : {ctx.mean():.1%}")
    print(f"    LLM-judged gate : {gated.mean():.1%}")
    print(f"    oracle ceiling  : {(no_ctx | ctx).mean():.1%}")

    # --- the number that matters: judge skill on discordant questions ---
    helpful = ctx & ~no_ctx
    harmful = no_ctx & ~ctx
    disc = helpful | harmful

    # Correct call = said YES where context helps, NO where it harms.
    correct_call = np.where(helpful, use_ctx, ~use_ctx)[disc]

    k = int(correct_call.sum())
    m = int(disc.sum())

    p_skill = binomtest(k, m, 0.5).pvalue if m else 1.0

    print()
    print(f"    JUDGE SKILL on the {m} discordant questions:")
    print(f"      correct calls: {k}/{m} = {k / m:.1%}   (chance = 50%)")
    print(f"      binomial p = {p_skill:.4f}"
          f"{'   -> REAL SKILL' if p_skill < 0.05 and k / m > 0.5 else '   -> no better than chance'}")

    # gate vs better fixed policy
    if no_ctx.mean() >= ctx.mean():
        base_name, base_ok = "no context", no_ctx
    else:
        base_name, base_ok = "always context", ctx

    g = int((gated & ~base_ok).sum())
    l = int((~gated & base_ok).sum())
    p_gate = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0

    print()
    print(f"    vs best fixed policy ({base_name}): "
          f"{(gated.mean() - base_ok.mean()) * 100:+.1f} pp  "
          f"(+{g}/-{l}, p={p_gate:.4f})"
          f"{'  SIGNIFICANT' if p_gate < 0.05 else ''}")

    return {
        "dataset": dataset_key,
        "context_arm": ctx_arm,
        "n": n,
        "judge_says_useful_frac": round(float(use_ctx.mean()), 4),
        "acc_no_context": round(float(no_ctx.mean()), 4),
        "acc_always_context": round(float(ctx.mean()), 4),
        "acc_llm_judged_gate": round(float(gated.mean()), 4),
        "acc_oracle": round(float((no_ctx | ctx).mean()), 4),
        "discordant": m,
        "judge_correct_calls": k,
        "judge_skill_pct": round(100.0 * k / m, 2) if m else None,
        "judge_skill_p": round(float(p_skill), 5),
        "gate_vs_best_fixed_pp": round(
            float((gated.mean() - base_ok.mean()) * 100), 2),
        "gate_p": round(float(p_gate), 5),
        "gate_significant": bool(p_gate < 0.05),
    }


def main():

    print("=" * 72)
    print("LLM-JUDGED SELECTIVE RETRIEVAL")
    print("=" * 72)

    rows = []

    for dataset_key in DATASET_FILES:

        print(f"\nDataset: {dataset_key}")

        questions = questions_for(dataset_key)

        ctx_path = os.path.join(RESULT_DIR, f"paired_contexts_{dataset_key}.json")

        if not os.path.exists(ctx_path):
            print(f"  missing {ctx_path} -- run paired_comparison.py first")
            return

        with open(ctx_path, "r", encoding="utf-8") as f:
            contexts = json.load(f)

        judge_df = collect_judgements(dataset_key, questions, contexts)

        for ctx_arm in ("CTX", "CTX_FT"):
            rows.append(analyse(dataset_key, judge_df, ctx_arm))

    out = os.path.join(RESULT_DIR, "llm_judge_gate_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved to: {out}")
    print("\nRead 'judge_skill_pct' first. If it is not clearly above 50% with")
    print("a small p, the judge has no skill and selective retrieval by self-")
    print("assessment does not work either -- which is itself the finding.")


if __name__ == "__main__":
    main()
