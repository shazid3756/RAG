"""
Evaluate the Llama-3.2-3B fine-tune, and test whether Finding 3 generalises
beyond Qwen2.5.

ARMS (full test sets, greedy, minilm+hybrid context -- identical protocol to
run_full_study.py, reusing its cached contexts):

    L_C   llama3.2:3b                 + context     base, the control
    L_D   llama3.2-3b-medqa-lora      + context     fine-tuned
    L_A   llama3.2:3b                 no context    for the ablation ladder

Ordered so the decisive pair (L_C, L_D) finishes first. L_A only replicates
Finding 4 on a second model -- useful, not essential -- so stopping after
L_D still answers the main question. Latency is recorded this time; the
full Qwen run omitted it and the latency numbers had to be cited from the
smaller n=500 run.

QUESTIONS THIS ANSWERS

  1. Does QLoRA fine-tuning help a DIFFERENT model family the same way?
     L_D vs L_C, per dataset. If MedMCQA gains and MedQA does not, Finding 3
     is a property of the method, not of Qwen2.5.

  2. Does a fine-tuned 3B match a 7B?  L_D vs full_*_C (base Qwen2.5-7B with
     context) and vs full_*_D (fine-tuned Qwen2.5-7B). This is the
     efficiency claim -- the only result in the study that speaks to the
     word "Efficient" in the paper's title.

Run from the project root:
    python eval_llama3b.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
NUM_PREDICT = 150
CHECKPOINT_EVERY = 20

BASE_TAG = "llama3.2:3b"
FT_TAG = "llama3.2-3b-medqa-lora"

# (arm, model, uses_context)
ARMS = [
    ("L_C", BASE_TAG, True),
    ("L_D", FT_TAG, True),
    ("L_A", BASE_TAG, False),
]


def questions(ds):
    with open(DATASET_FILES[ds], "r", encoding="utf-8") as f:
        return json.load(f)


def contexts(ds):
    p = os.path.join(RESULT_DIR, f"contexts_full_{ds}_minilm_hybrid.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_arm(ds, name):
    p = os.path.join(RESULT_DIR, f"full_{ds}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def build_no_context_prompt(question, options):
    """Same wording as build_mcqa_prompt minus the reference block -- must
    match run_full_study.py exactly or arm A is not comparable."""

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


def mcnemar(a_ok, b_ok):
    g = int((b_ok & ~a_ok).sum())
    l = int((~b_ok & a_ok).sum())
    p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
    return g, l, p


def preflight():
    """Verify every model actually answers BEFORE spending hours on it.

    The first attempt at this evaluation produced 0.00% on all three arms
    because `ollama create` had failed with "timed out waiting for server to
    start" -- the Ollama server was down, so every request failed silently
    (generate_answer catches connection errors and returns an error STRING,
    which is not None, so nothing raised). Six hours of calls were wasted
    writing rows that all said unparseable. One call per model up front
    would have caught it in five seconds.
    """

    print("Preflight: checking each model responds and parses...\n")

    ds = list(DATASET_FILES)[0]
    q = questions(ds)[0]
    ctx = contexts(ds)[0]

    prompt = build_mcqa_prompt(ctx, q["question"], q["options"])

    ok = True

    for arm, model, _ in ARMS:

        try:
            resp = generate_answer(prompt, model=model, num_predict=NUM_PREDICT)
        except Exception as e:
            print(f"  {model:26s} EXCEPTION {e}")
            ok = False
            continue

        pred = extract_choice(resp)

        if pred is None:
            print(f"  {model:26s} FAILED -- no parseable answer")
            print(f"    raw response: {resp[:300]!r}")
            ok = False
        else:
            print(f"  {model:26s} OK (answered {pred})")

    if not ok:
        print("\n" + "=" * 72)
        print("ABORTING before the long run.")
        print("Most likely: the Ollama server is not running, or the model was")
        print("never created. Check with:")
        print("    ollama list")
        print("If llama3.2-3b-medqa-lora is missing, restart Ollama and run:")
        print("    ollama create llama3.2-3b-medqa-lora --quantize q4_K_M "
              "-f Modelfile_llama3b")
        print("=" * 72)
        sys.exit(1)

    print("\n  All models responding. Starting evaluation.\n")


def run(arm, model, use_ctx, ds):

    qs = questions(ds)
    ctx = contexts(ds) if use_ctx else None

    if use_ctx and len(ctx) != len(qs):
        print(f"  [{ds}] context cache mismatch -- run run_full_study.py first")
        return None

    path = os.path.join(RESULT_DIR, f"full_{ds}_{arm}.csv")

    rows, start = [], 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start = len(rows)
        if start >= len(qs):
            print(f"  [{arm}/{ds}] complete ({start})")
            return pd.DataFrame(rows)
        print(f"  [{arm}/{ds}] resuming at {start}/{len(qs)}")

    t0 = time.time()

    for i in range(start, len(qs)):

        done = i - start + 1

        if done % 5 == 0 or i == len(qs) - 1:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(qs) - i - 1) / rate / 60 if rate else 0
            print(f"  [{arm} {ds}] {i + 1}/{len(qs)}  eta {eta:.0f} min   ",
                  end="\r")

        if use_ctx:
            prompt = build_mcqa_prompt(ctx[i], qs[i]["question"], qs[i]["options"])
        else:
            prompt = build_no_context_prompt(qs[i]["question"], qs[i]["options"])

        t = time.time()

        try:
            resp = generate_answer(prompt, model=model, num_predict=NUM_PREDICT)
        except Exception as e:
            print(f"\n  ERROR {arm} q{i + 1}: {e}")
            resp = ""

        latency = time.time() - t
        pred = extract_choice(resp)

        rows.append({
            "index": i,
            "true_answer": qs[i]["answer_idx"],
            "prediction": pred if pred is not None else "",
            "unparseable": pred is None,
            "correct": (pred == qs[i]["answer_idx"]),
            "latency": round(latency, 3),
            # Kept ONLY on failure: enough to diagnose a bad run from the CSV
            # alone, without bloating the file when everything is fine.
            "raw_on_failure": (resp or "")[:200] if pred is None else "",
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(qs) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


def report(ds, arms):

    out = {"dataset": ds, "n": len(arms["L_C"])}

    ok = {k: v["correct"].astype(bool).values for k, v in arms.items()}
    lat = {k: v["latency"].mean() for k, v in arms.items() if "latency" in v}

    print(f"\n{'=' * 72}")
    print(f"{ds}  (n={out['n']})")
    print("=" * 72)

    for k in ("L_A", "L_C", "L_D"):
        if k in ok:
            label = {"L_A": "3B no context", "L_C": "3B + context",
                     "L_D": "3B fine-tuned + context"}[k]
            print(f"  {label:26s} {ok[k].mean():.2%}"
                  + (f"   {lat[k]:.2f}s" if k in lat else ""))
            out[f"acc_{k}"] = round(float(ok[k].mean()), 4)
            if k in lat:
                out[f"latency_{k}"] = round(float(lat[k]), 3)

    # 1. does fine-tuning help this family?
    g, l, p = mcnemar(ok["L_C"], ok["L_D"])
    d = (ok["L_D"].mean() - ok["L_C"].mean()) * 100
    print(f"\n  FINE-TUNING EFFECT (L_C -> L_D): {d:+.2f} pp  +{g}/-{l}  "
          f"p={p:.5f}{'  SIGNIFICANT' if p < 0.05 else ''}")
    out.update({"ft_effect_pp": round(float(d), 2), "ft_gained": g,
                "ft_lost": l, "ft_p": round(float(p), 6)})

    if "L_A" in ok:
        g2, l2, p2 = mcnemar(ok["L_A"], ok["L_C"])
        d2 = (ok["L_C"].mean() - ok["L_A"].mean()) * 100
        print(f"  RETRIEVAL EFFECT (L_A -> L_C):   {d2:+.2f} pp  +{g2}/-{l2}  "
              f"p={p2:.5f}{'  SIGNIFICANT' if p2 < 0.05 else ''}")
        out.update({"retrieval_effect_pp": round(float(d2), 2),
                    "retrieval_p": round(float(p2), 6)})

    # 2. efficiency: 3B fine-tuned vs the 7B arms
    for qwen_arm, label in [("C", "base Qwen2.5-7B + ctx"),
                            ("D", "fine-tuned Qwen2.5-7B + ctx")]:

        qdf = load_arm(ds, qwen_arm)

        if qdf is None or len(qdf) != out["n"]:
            continue

        q_ok = qdf["correct"].astype(bool).values
        g3, l3, p3 = mcnemar(q_ok, ok["L_D"])
        d3 = (ok["L_D"].mean() - q_ok.mean()) * 100

        verdict = ("3B WINS" if p3 < 0.05 and d3 > 0 else
                   "3B loses" if p3 < 0.05 else "no significant difference")

        print(f"  vs {label:30s} {d3:+.2f} pp  +{g3}/-{l3}  "
              f"p={p3:.5f}   {verdict}")

        out[f"vs_qwen_{qwen_arm}_pp"] = round(float(d3), 2)
        out[f"vs_qwen_{qwen_arm}_p"] = round(float(p3), 6)

    return out


def main():

    print("=" * 72)
    print("LLAMA-3.2-3B FINE-TUNE EVALUATION")
    print("=" * 72 + "\n")

    preflight()

    rows = []

    for ds in DATASET_FILES:

        print(f"\nDataset: {ds}")

        arms = {}

        for arm, model, use_ctx in ARMS:
            df = run(arm, model, use_ctx, ds)
            if df is None:
                return
            arms[arm] = df

        rows.append(report(ds, arms))

    out = os.path.join(RESULT_DIR, "llama3b_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\n{'=' * 72}")
    print("POOLED ACROSS DATASETS")
    print("=" * 72)

    df = pd.DataFrame(rows)

    g, l = int(df.ft_gained.sum()), int(df.ft_lost.sum())

    if g + l:
        print(f"  fine-tuning effect: +{g}/-{l}  "
              f"p={binomtest(g, g + l, 0.5).pvalue:.6f}")
    else:
        print("  fine-tuning effect: no discordant items -- "
              "the two arms answered identically everywhere, which almost "
              "always means the run failed rather than that they agree.")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
