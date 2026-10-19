"""
Does matching the inference prompt format to the TRAINING format improve the
fine-tuned model?

THE MISMATCH
------------
finetune_data/train.jsonl stores the RAW prompt text -- exactly what
build_mcqa_prompt() produces, with no chat markers. qlora_finetune.py then
tokenizes it with add_special_tokens=False, so during training the model only
ever saw:

    <raw prompt><completion><|im_end|>

But the deployed Modelfile wraps every request in ChatML:

    <|im_start|>user\\n<raw prompt><|im_end|>\\n<|im_start|>assistant\\n

So at inference the model is asked inside a wrapper it was never fine-tuned
in. It still gained +9.92 pp on MedMCQA, but MedQA showed nothing -- and part
of that null may be this mismatch rather than a limit of fine-tuning.

Modelfile_raw sets its TEMPLATE to a bare prompt placeholder, passing the
prompt through untouched, which reproduces the training format exactly. The
raw Modelfile is the authoritative reference for it. The stop token stays
<|im_end|> because qlora_finetune.py appended tokenizer.eos_token, which for
Qwen2.5 IS <|im_end|>.

BEFORE RUNNING, create the model:

    ollama create qwen2.5-medqa-lora-raw --quantize q4_K_M -f Modelfile_raw

WHAT THIS SCRIPT DOES
---------------------
1. Smoke test: 5 items, prints the raw responses so a broken template is
   obvious immediately (Ctrl+C if they look wrong).
2. Full evaluation, MedQA first (1,273 items, ~55 min) because that is where
   fine-tuning showed no effect and where the mismatch would bite hardest.
   Then MedMCQA (4,183 items, ~3 h). Resumable -- Ctrl+C after MedQA is fine.
3. McNemar of raw-template vs the ChatML arm D, and vs no-context arm A,
   reusing the per-question CSVs run_full_study.py already wrote.

Run from the project root:
    python eval_raw_template.py
"""

import json
import os
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

RAW_MODEL = "qwen2.5-medqa-lora-raw"

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",       # run first, smaller
    "medmcqa": "evaluation/medmcqa_val.json",
}

RESULT_DIR = "evaluation/results"
NUM_PREDICT = 150
CHECKPOINT_EVERY = 20
SMOKE_N = 5


def load(dataset, name):
    p = os.path.join(RESULT_DIR, f"full_{dataset}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def contexts(dataset):
    p = os.path.join(RESULT_DIR, f"contexts_full_{dataset}_minilm_hybrid.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def questions(dataset):
    with open(DATASET_FILES[dataset], "r", encoding="utf-8") as f:
        return json.load(f)


def mcnemar(a_ok, b_ok):
    g = int((b_ok & ~a_ok).sum())
    l = int((~b_ok & a_ok).sum())
    p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
    return g, l, p


def smoke(qs, ctx):

    print("\n" + "=" * 70)
    print(f"SMOKE TEST -- {SMOKE_N} items through {RAW_MODEL}")
    print("=" * 70)

    bad = 0

    for i in range(SMOKE_N):

        prompt = build_mcqa_prompt(ctx[i], qs[i]["question"], qs[i]["options"])
        resp = generate_answer(prompt, model=RAW_MODEL, num_predict=NUM_PREDICT)
        pred = extract_choice(resp)

        if pred is None:
            bad += 1

        print(f"\n  Q{i + 1}: parsed={pred}  true={qs[i]['answer_idx']}")
        print(f"    {resp[:220]!r}")

    print(f"\n  unparseable {bad}/{SMOKE_N}")

    if bad:
        print("  >>> Template looks wrong. Ctrl+C now and report this output.")
    else:
        print("  >>> Format looks correct. Continuing to full evaluation.")

    time.sleep(3)


def run(dataset):

    qs = questions(dataset)
    ctx = contexts(dataset)

    if len(ctx) != len(qs):
        print(f"[{dataset}] context cache has {len(ctx)}, expected {len(qs)}")
        return None

    path = os.path.join(RESULT_DIR, f"full_{dataset}_D_RAW.csv")

    rows, start = [], 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        start = len(rows)
        if start >= len(qs):
            print(f"  [{dataset}] complete ({start})")
            return pd.DataFrame(rows)
        print(f"  [{dataset}] resuming at {start}/{len(qs)}")

    t0 = time.time()

    for i in range(start, len(qs)):

        done = i - start + 1

        if done % 5 == 0 or i == len(qs) - 1:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(qs) - i - 1) / rate / 60 if rate else 0
            print(f"  [{dataset} raw] {i + 1}/{len(qs)}  eta {eta:.0f} min   ",
                  end="\r")

        prompt = build_mcqa_prompt(ctx[i], qs[i]["question"], qs[i]["options"])

        try:
            resp = generate_answer(prompt, model=RAW_MODEL,
                                   num_predict=NUM_PREDICT)
        except Exception as e:
            print(f"\n  ERROR q{i + 1}: {e}")
            resp = ""

        pred = extract_choice(resp)

        rows.append({
            "index": i,
            "true_answer": qs[i]["answer_idx"],
            "prediction": pred if pred is not None else "",
            "unparseable": pred is None,
            "correct": (pred == qs[i]["answer_idx"]),
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(qs) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)

    print()

    pd.DataFrame(rows).to_csv(path, index=False)

    return pd.DataFrame(rows)


def report(dataset, raw_df):

    a_df, d_df = load(dataset, "A"), load(dataset, "D")

    if a_df is None or d_df is None:
        print(f"  [{dataset}] missing arm A or D -- run run_full_study.py first")
        return None

    raw = raw_df["correct"].astype(bool).values
    a = a_df["correct"].astype(bool).values
    d = d_df["correct"].astype(bool).values

    print(f"\n{'=' * 70}")
    print(f"{dataset}  (n={len(raw)})")
    print("=" * 70)
    print(f"  A  no context                 : {a.mean():.2%}")
    print(f"  D  fine-tuned, ChatML template: {d.mean():.2%}")
    print(f"  D_RAW fine-tuned, raw template: {raw.mean():.2%}")
    print(f"  unparseable (raw)             : {raw_df['unparseable'].sum()}")

    g, l, p = mcnemar(d, raw)
    print(f"\n  raw vs ChatML : {(raw.mean() - d.mean()) * 100:+.2f} pp  "
          f"+{g}/-{l}  p={p:.4f}"
          f"{'  SIGNIFICANT' if p < 0.05 else ''}")

    g2, l2, p2 = mcnemar(a, raw)
    print(f"  raw vs no-ctx : {(raw.mean() - a.mean()) * 100:+.2f} pp  "
          f"+{g2}/-{l2}  p={p2:.4f}"
          f"{'  SIGNIFICANT' if p2 < 0.05 else ''}")

    return {
        "dataset": dataset, "n": len(raw),
        "acc_no_context": round(float(a.mean()), 4),
        "acc_chatml": round(float(d.mean()), 4),
        "acc_raw": round(float(raw.mean()), 4),
        "raw_unparseable": int(raw_df["unparseable"].sum()),
        "raw_vs_chatml_pp": round(float((raw.mean() - d.mean()) * 100), 2),
        "raw_vs_chatml_gained": g, "raw_vs_chatml_lost": l,
        "raw_vs_chatml_p": round(float(p), 5),
        "raw_vs_nocontext_pp": round(float((raw.mean() - a.mean()) * 100), 2),
        "raw_vs_nocontext_p": round(float(p2), 5),
    }


def main():

    print("=" * 70)
    print("RAW-TEMPLATE EVALUATION -- train/inference format matched")
    print("=" * 70)

    first = list(DATASET_FILES)[0]
    smoke(questions(first)[:SMOKE_N], contexts(first)[:SMOKE_N])

    rows = []

    for dataset in DATASET_FILES:
        df = run(dataset)
        if df is not None:
            r = report(dataset, df)
            if r:
                rows.append(r)

    if rows:
        out = os.path.join(RESULT_DIR, "raw_template_comparison.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")

        if len(rows) == 2:
            g = sum(r["raw_vs_chatml_gained"] for r in rows)
            l = sum(r["raw_vs_chatml_lost"] for r in rows)
            p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
            print(f"\nPOOLED raw vs ChatML: +{g}/-{l}  p={p:.5f}"
                  f"{'  SIGNIFICANT' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
