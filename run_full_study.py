"""
FULL-DATA run of the ablation, fine-tuning and selective-retrieval analyses.

Replaces the n=500 sampled runs of ablation_study.py, paired_comparison.py,
adaptive_retrieval.py, gate_signal_diagnostic.py and llm_judge_gate.py with a
single resumable pass over the COMPLETE test sets:

    MedQA   evaluation/medqa_test.json    1,273 questions
    MedMCQA evaluation/medmcqa_val.json   4,183 questions

Everything is greedy (temperature=0, seed=42) so results are reproducible and
both models decode under identical rules.

WHAT IT PRODUCES (all suffixed _full, nothing existing is overwritten)

    arms, one generation pass each:
      A      no retrieval,          qwen2.5:7b
      B      bge + dense,           qwen2.5:7b
      C      minilm + hybrid,       qwen2.5:7b
      D      minilm + hybrid,       qwen2.5-medqa-lora
      JUDGE  relevance judgement,   qwen2.5:7b

    analyses:
      ablation ladder A->B->C->D with McNemar between consecutive stages
      C vs D paired fine-tuning test
      oracle gating ceiling
      cosine-threshold gating (tuned on even indices, reported on odd)
      ROC AUC of retrieval-confidence signals
      LLM-judged gating + judge skill on discordant questions

RUNTIME ~19 h. It checkpoints every 20 questions and skips completed work on
restart, so run it in chunks -- Ctrl+C is safe, and so is a power cut.
Arms are ordered A, B, C, JUDGE (all base model) then D, so Ollama loads only
two models per dataset instead of swapping repeatedly.

Run from the project root:
    python run_full_study.py
"""

import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import retriever
import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


rag_pipeline.OLLAMA_OPTIONS.update({
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "repeat_penalty": 1.0,
})

TAG = "full"

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
os.makedirs(RESULT_DIR, exist_ok=True)

BASE_MODEL = "qwen2.5:7b"
FT_MODEL = "qwen2.5-medqa-lora"

TOP_K = 8
NUM_PREDICT = 150
JUDGE_NUM_PREDICT = 5
CHECKPOINT_EVERY = 20

# Ordered so the base model runs to completion before the fine-tuned one.
ARMS = [
    ("A", "LLM only (no RAG)",        BASE_MODEL, None,     None),
    ("B", "+ naive RAG (bge, dense)", BASE_MODEL, "bge",    "dense"),
    ("C", "+ optimized retrieval",    BASE_MODEL, "minilm", "hybrid"),
    ("D", "+ fine-tuned LLM",         FT_MODEL,   "minilm", "hybrid"),
]

GATE_EMBEDDING = "minilm"


def load_questions(dataset_key):

    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================
# Retrieval (no LLM)
# ==========================

def contexts_for(dataset_key, questions, embedding_model, mode):

    if embedding_model is None:
        return None

    path = os.path.join(
        RESULT_DIR, f"contexts_{TAG}_{dataset_key}_{embedding_model}_{mode}.json"
    )

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        if len(ctx) == len(questions):
            print(f"    contexts cached ({embedding_model}/{mode})")
            return ctx

    ctx = []

    for i, item in enumerate(questions):
        print(f"    retrieving {embedding_model}/{mode} "
              f"{i + 1}/{len(questions)}", end="\r")

        res = retriever.retrieve_context(
            item["question"], k=TOP_K, mode=mode, model_key=embedding_model
        )
        chunks = [r["chunk"] for r in res]
        ctx.append("\n\n".join(chunks) if chunks
                   else "(no reference material retrieved)")

        if (i + 1) % 500 == 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ctx, f)

    print()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx, f)

    return ctx


def gate_scores(dataset_key, questions):

    path = os.path.join(RESULT_DIR, f"gate_scores_{TAG}_{dataset_key}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) == len(questions):
            print("    gate scores cached")
            return df

    rows = []

    for i, item in enumerate(questions):
        print(f"    scoring {i + 1}/{len(questions)}", end="\r")

        res = retriever.retrieve_context(
            item["question"], k=TOP_K, mode="dense", model_key=GATE_EMBEDDING
        )
        s = [r["score"] for r in res] or [0.0]

        rows.append({"index": i, "top1": float(max(s)),
                     "mean_topk": float(np.mean(s))})

        if (i + 1) % 500 == 0:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    return df


# ==========================
# Generation passes
# ==========================

def build_no_context_prompt(question, options):
    """Stage A prompt: same wording as build_mcqa_prompt with only the
    reference-material block removed, so A vs B isolates context presence."""

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


def build_judge_prompt(context_text, question):

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


def run_pass(name, dataset_key, questions, contexts, model, judge=False):
    """One resumable generation pass. `name` is the arm id or 'JUDGE'."""

    path = os.path.join(RESULT_DIR, f"{TAG}_{dataset_key}_{name}.csv")

    rows, start_at = [], 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start_at = len(rows)
        if start_at >= len(questions):
            print(f"  [{name}] complete ({start_at})")
            return pd.DataFrame(rows)
        print(f"  [{name}] resuming at {start_at}/{len(questions)}")

    t_start = time.time()

    for i in range(start_at, len(questions)):

        item = questions[i]

        done = i - start_at + 1

        if done % 5 == 0 or i == len(questions) - 1:
            rate = done / max(time.time() - t_start, 1e-9)
            eta = (len(questions) - i - 1) / rate / 60 if rate else 0
            print(f"  [{name} {dataset_key}] {i + 1}/{len(questions)}  "
                  f"eta {eta:.0f} min   ", end="\r")

        if judge:
            prompt = build_judge_prompt(contexts[i], item["question"])
            npred = JUDGE_NUM_PREDICT
        elif contexts is None:
            prompt = build_no_context_prompt(item["question"], item["options"])
            npred = NUM_PREDICT
        else:
            prompt = build_mcqa_prompt(contexts[i], item["question"],
                                       item["options"])
            npred = NUM_PREDICT

        try:
            resp = generate_answer(prompt, model=model, num_predict=npred)
        except Exception as e:
            print(f"\n  ERROR {name} q{i + 1}: {e}")
            resp = ""

        if judge:
            m = re.search(r"\b(YES|NO)\b", (resp or "").strip(), re.IGNORECASE)
            verdict = m.group(1).upper() if m else None
            rows.append({
                "index": i,
                # Unparseable -> YES = keep context = standard RAG behaviour,
                # so a judge failure degrades to the baseline system.
                "verdict": verdict if verdict else "YES",
                "parsed": verdict is not None,
            })
        else:
            pred = extract_choice(resp)
            rows.append({
                "index": i,
                "true_answer": item["answer_idx"],
                "prediction": pred if pred is not None else "",
                "unparseable": pred is None,
                "correct": (pred == item["answer_idx"]),
            })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(questions) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


# ==========================
# Analysis helpers
# ==========================

def mcnemar(prev_ok, curr_ok):

    b = int((curr_ok & ~prev_ok).sum())
    c = int((~curr_ok & prev_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0

    return b, c, p


def auc(scores, labels):

    pos, neg = scores[labels == 1], scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)

    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()

    r_pos = ranks[:len(pos)].sum()

    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ==========================
# Main
# ==========================

def main():

    print("=" * 76)
    print("FULL-DATA STUDY -- greedy decoding, complete test sets")
    print("=" * 76)

    ladder_rows, gate_rows, auc_rows = [], [], []

    for dataset_key in DATASET_FILES:

        questions = load_questions(dataset_key)

        print(f"\n{'=' * 76}")
        print(f"{dataset_key}: {len(questions)} questions")
        print("=" * 76)

        # ---- retrieval first (no LLM, cheap) ----
        ctx_cache = {}

        for _, _, _, emb, mode in ARMS:
            if emb is not None and (emb, mode) not in ctx_cache:
                ctx_cache[(emb, mode)] = contexts_for(
                    dataset_key, questions, emb, mode
                )

        gates = gate_scores(dataset_key, questions)

        # ---- generation passes, base-model arms first ----
        results = {}

        for arm_id, label, model, emb, mode in ARMS:
            if arm_id == "D":
                continue
            ctx = ctx_cache.get((emb, mode)) if emb else None
            results[arm_id] = run_pass(arm_id, dataset_key, questions, ctx, model)

        judge_df = run_pass(
            "JUDGE", dataset_key, questions,
            ctx_cache[("minilm", "hybrid")], BASE_MODEL, judge=True,
        )

        for arm_id, label, model, emb, mode in ARMS:
            if arm_id != "D":
                continue
            results[arm_id] = run_pass(
                arm_id, dataset_key, questions,
                ctx_cache[(emb, mode)], model,
            )

        ok = {k: v["correct"].astype(bool).values for k, v in results.items()}

        # ---- ablation ladder ----
        print(f"\n  {'-' * 70}")
        print(f"  {dataset_key} -- ablation ladder (n={len(questions)})")
        print(f"  {'-' * 70}")

        prev = None

        for arm_id, label, model, emb, mode in ARMS:

            acc = ok[arm_id].mean()
            row = {"dataset": dataset_key, "stage": arm_id, "config": label,
                   "n": len(questions), "accuracy": round(float(acc), 4)}

            if prev:
                b, c, p = mcnemar(ok[prev], ok[arm_id])
                row.update({"vs_prev": f"{prev}->{arm_id}",
                            "delta_pp": round(float((acc - ok[prev].mean()) * 100), 2),
                            "gained": b, "lost": c,
                            "mcnemar_p": round(float(p), 6),
                            "significant": bool(p < 0.05)})
                print(f"    {arm_id} {label:30s} {acc:.1%}   "
                      f"{prev}->{arm_id} {(acc - ok[prev].mean()) * 100:+.1f} pp "
                      f"(+{b}/-{c}, p={p:.4f})"
                      f"{'  SIG' if p < 0.05 else ''}")
            else:
                print(f"    {arm_id} {label:30s} {acc:.1%}   (baseline)")

            ladder_rows.append(row)
            prev = arm_id

        b, c, p = mcnemar(ok["A"], ok["D"])
        print(f"\n    END TO END A->D: {(ok['D'].mean() - ok['A'].mean()) * 100:+.1f} pp "
              f"(+{b}/-{c}, p={p:.5f}){'  SIG' if p < 0.05 else ''}")

        ladder_rows.append({
            "dataset": dataset_key, "stage": "A->D", "config": "END TO END",
            "n": len(questions), "accuracy": round(float(ok["D"].mean()), 4),
            "vs_prev": "A->D",
            "delta_pp": round(float((ok["D"].mean() - ok["A"].mean()) * 100), 2),
            "gained": b, "lost": c, "mcnemar_p": round(float(p), 6),
            "significant": bool(p < 0.05),
        })

        # ---- gating: oracle, cosine threshold, LLM judge ----
        scores = gates["top1"].values
        use_judge = (judge_df["verdict"].values == "YES")

        for ctx_arm in ("C", "D"):

            no_ctx, ctx = ok["A"], ok[ctx_arm]
            oracle = (no_ctx | ctx)
            best_fixed = max(no_ctx.mean(), ctx.mean())

            n = len(no_ctx)
            idx = np.arange(n)
            tune, test = idx % 2 == 0, idx % 2 == 1

            best_t, best_acc = None, -1.0

            for t in np.quantile(scores, np.linspace(0, 1, 41)):
                a = np.where(scores >= t, ctx, no_ctx)[tune].mean()
                if a > best_acc:
                    best_acc, best_t = a, t

            cos_gate = np.where(scores >= best_t, ctx, no_ctx)
            judge_gate = np.where(use_judge, ctx, no_ctx)

            helpful, harmful = ctx & ~no_ctx, no_ctx & ~ctx
            disc = helpful | harmful
            correct_call = np.where(helpful, use_judge, ~use_judge)[disc]
            k, m = int(correct_call.sum()), int(disc.sum())
            p_skill = binomtest(k, m, 0.5).pvalue if m else 1.0

            print(f"\n    gating vs arm {ctx_arm}: no-ctx {no_ctx.mean():.1%}  "
                  f"ctx {ctx.mean():.1%}  ORACLE {oracle.mean():.1%}  "
                  f"(headroom {(oracle.mean() - best_fixed) * 100:+.1f} pp)")
            print(f"      cosine gate (held-out): {cos_gate[test].mean():.1%}")
            print(f"      LLM-judge gate        : {judge_gate.mean():.1%}")
            print(f"      judge skill on {m} discordant: {k}/{m} = "
                  f"{k / m:.1%}  p={p_skill:.4f}"
                  f"{'  REAL SKILL' if p_skill < 0.05 and k / m > 0.5 else ''}")

            gate_rows.append({
                "dataset": dataset_key, "context_arm": ctx_arm, "n": n,
                "acc_no_context": round(float(no_ctx.mean()), 4),
                "acc_always_context": round(float(ctx.mean()), 4),
                "acc_oracle": round(float(oracle.mean()), 4),
                "headroom_pp": round(float((oracle.mean() - best_fixed) * 100), 2),
                "cosine_threshold": round(float(best_t), 4),
                "cosine_gate_heldout": round(float(cos_gate[test].mean()), 4),
                "judge_says_useful_frac": round(float(use_judge.mean()), 4),
                "judge_gate_acc": round(float(judge_gate.mean()), 4),
                "discordant": m, "judge_correct": k,
                "judge_skill_pct": round(100.0 * k / m, 2) if m else None,
                "judge_skill_p": round(float(p_skill), 6),
            })

            labels = helpful[disc].astype(int)

            for sig_name, sig in [
                ("top1 cosine", gates["top1"].values),
                ("mean top-k cosine", gates["mean_topk"].values),
                ("top1 - mean (margin)", (gates["top1"] - gates["mean_topk"]).values),
            ]:
                auc_rows.append({
                    "dataset": dataset_key, "context_arm": ctx_arm,
                    "signal": sig_name,
                    "auc": round(float(auc(sig[disc], labels)), 4),
                    "discordant": m,
                })

    for rows, name in [(ladder_rows, "ablation_ladder"),
                       (gate_rows, "gating"),
                       (auc_rows, "gate_signal_auc")]:
        out = os.path.join(RESULT_DIR, f"{TAG}_{name}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")

    print("\n" + "=" * 76)
    print("POOLED TESTS ACROSS BOTH DATASETS")
    print("=" * 76)

    lad = pd.DataFrame(ladder_rows)

    for step in ["A->B", "B->C", "C->D", "A->D"]:
        sub = lad[lad["vs_prev"] == step]
        if len(sub) == 2:
            g, l = int(sub["gained"].sum()), int(sub["lost"].sum())
            p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
            print(f"  {step}: +{g}/-{l}  p={p:.5f}"
                  f"{'  SIGNIFICANT' if p < 0.05 else ''}")

    gt = pd.DataFrame(gate_rows)
    k, m = int(gt["judge_correct"].sum()), int(gt["discordant"].sum())
    print(f"\n  pooled judge skill: {k}/{m} = {k / m:.1%}  "
          f"p={binomtest(k, m, 0.5).pvalue:.5f}")


if __name__ == "__main__":
    main()
