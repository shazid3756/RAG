"""
top_k control - does stage C change when the prompt actually fits in num_ctx?
============================================================================

THE PROBLEM THIS TESTS
----------------------
rag_pipeline sets num_ctx=2048 and the context arms retrieve TOP_K=8 chunks of
~220 words each. On MedQA, whose questions are long clinical vignettes, the
assembled prompt averages ~1,350 words. At any plausible tokens-per-word ratio
that is close to or past 2048, and once num_predict=150 is reserved for the
answer a large share of items do not fit at all:

    ratio 1.30  ->  31% of MedQA prompts overflow
    ratio 1.35  ->  40%
    ratio 1.40  ->  49%

When a prompt exceeds the context window the server silently drops tokens. The
reference material sits at the FRONT of this prompt, so it is what gets cut.
Stage A is unaffected (no context block, short prompt), which means the A->C
contrast may have been measured with the context partially truncated on a
large minority of MedQA items.

This does not invalidate A vs C as an engineering comparison - "the system as
built" is a legitimate thing to measure. But it does undermine any claim about
WHY retrieval hurts, and a referee who notices it before we do is a bad
outcome.

WHAT THIS SCRIPT DOES
---------------------
Re-runs stage C on MedQA with TOP_K=4 so every prompt fits, holding everything
else fixed - same embedding model, same retrieval mode, same greedy decoding,
same prompt template, same seed. Then it pairs the new outcomes against the
existing stage C, item by item.

It also records the EXACT prompt token count for every item, taken from
Ollama's own prompt_eval_count rather than estimated, and reports how many
prompts overflowed under each setting. That turns the overflow claim from an
estimate into a measurement.

HOW TO READ THE RESULT
----------------------
  no significant change  ->  truncation was not driving the result. The
                             retrieval penalty is real and the mechanism
                             discussion stands. Report this as a robustness
                             check; it closes the hole permanently.

  C(k=4) significantly better  ->  part of the penalty was truncation, not
                             retrieval. Report k=4 as the corrected context
                             arm and re-derive the ladder from it. This would
                             be a genuine correction and is much better found
                             here than in review.

Either way the paper gets a defensible number, and the token histogram goes
in the appendix.

COST
----
1,273 generations, MedQA only, about one hour. MedMCQA is not affected -
its questions are short, so its prompts already fit - and running it would
cost four hours to confirm a null. Use --both if you want it anyway.

USAGE
-----
    python run_topk_control.py
    python run_topk_control.py --topk 5 --both
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest

import retriever
import rag_pipeline
from mcqa_evaluation import build_mcqa_prompt, extract_choice


# ==========================
# Configuration
# ==========================

BASE_MODEL = "qwen2.5:7b"
RESULT_DIR = "evaluation/results"

# Stage C's configuration, held fixed except for top_k.
EMBEDDING = "minilm"
MODE = "hybrid"
BASELINE_TOPK = 8

NUM_PREDICT = 150
CHECKPOINT_EVERY = 20

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0,
          "seed": 42, "repeat_penalty": 1.0}

ANSWER_KEY = "answer_idx"


def check_schema(questions, name=""):
    if not questions:
        raise SystemExit(f"FAILED: {name} is empty")
    missing = [k for k in ("question", "options", ANSWER_KEY)
               if k not in questions[0]]
    if missing:
        raise SystemExit(
            f"FAILED: {name} items are missing {missing}. "
            f"Found keys: {sorted(questions[0].keys())}")


def show_decoding():
    keys = ("temperature", "top_p", "top_k", "seed", "repeat_penalty",
            "num_ctx", "num_gpu")
    active = {k: rag_pipeline.OLLAMA_OPTIONS.get(k) for k in keys
              if k in rag_pipeline.OLLAMA_OPTIONS}
    print("Decoding in force:", ", ".join(f"{k}={v}" for k, v in active.items()))
    bad = [k for k in GREEDY if rag_pipeline.OLLAMA_OPTIONS.get(k) != GREEDY[k]]
    if bad:
        raise SystemExit(f"FAILED: greedy settings did not take effect: {bad}")


# ==========================
# Generation with token accounting
# ==========================

def generate_counting_tokens(prompt, model, num_predict):
    """Call Ollama directly so prompt_eval_count can be captured.

    rag_pipeline.generate_answer returns only the text. The server also
    reports how many tokens it actually consumed, which is the one number
    that settles the truncation question without estimating a
    tokens-per-word ratio. Everything else here matches _call_ollama:
    same options dict, same keep_alive, same OOM fallback.

    Returns (text, prompt_tokens). prompt_tokens is None if the field is
    absent, which older Ollama builds do.
    """
    options = dict(rag_pipeline.OLLAMA_OPTIONS)
    options["num_predict"] = num_predict

    payload = {"model": model, "prompt": prompt, "stream": False,
               "keep_alive": rag_pipeline.OLLAMA_KEEP_ALIVE,
               "options": options}

    try:
        r = requests.post(rag_pipeline.OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip(), data.get("prompt_eval_count")
    except requests.exceptions.RequestException as e:
        body = getattr(getattr(e, "response", None), "text", "") or ""
        txt = (str(e) + " " + body).lower().replace("-", " ")
        if any(w in txt for w in ("out of memory", "cudamalloc",
                                  "unable to allocate", "failed to allocate")):
            time.sleep(5)
            options["num_gpu"] = 10
            payload["options"] = options
            try:
                r = requests.post(rag_pipeline.OLLAMA_URL, json=payload,
                                  timeout=180)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "").strip(), data.get("prompt_eval_count")
            except Exception as e2:
                return f"Ollama Error (after retry): {e2}", None
        return f"Ollama Error: {e} Body: {body}", None
    except Exception as e:
        return str(e), None


def preflight(model):
    print(f"Preflight: {model} ... ", end="", flush=True)
    txt, tok = generate_counting_tokens("Reply with exactly: OK", model, 8)
    if not isinstance(txt, str) or not txt.strip():
        sys.exit(f"\nFAILED: empty reply from {model}. Is ollama running?")
    if any(w in txt.lower() for w in ("error", "connection", "refused")):
        sys.exit(f"\nFAILED: {model} returned an error string: {txt[:200]}")
    print(f"ok ({txt.strip()[:30]!r}, prompt_eval_count={tok})")
    if tok is None:
        print("  NOTE: this Ollama build does not report prompt_eval_count; "
              "token columns will be empty and the overflow table skipped.")
    return tok is not None


# ==========================
# The run
# ==========================

def load_questions(key):
    with open(DATASET_FILES[key], "r", encoding="utf-8") as f:
        return json.load(f)


def run_arm(dataset_key, questions, topk):
    path = os.path.join(RESULT_DIR, f"topk{topk}_{dataset_key}_C.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): a list mixing Series with the
        # dicts appended below makes pd.DataFrame() raise
        # "'dict' object has no attribute 'dtype'". Only bites on resume.
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"  k={topk}/{dataset_key}: already complete ({len(done)})")
            return pd.DataFrame(list(done.values()))
        print(f"  k={topk}/{dataset_key}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    started = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue

        res = retriever.retrieve_context(item["question"], k=topk,
                                         mode=MODE, model_key=EMBEDDING)
        chunks = [r["chunk"] for r in res]
        ctx = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"
        prompt = build_mcqa_prompt(ctx, item["question"], item["options"])

        reply, ptok = generate_counting_tokens(prompt, BASE_MODEL, NUM_PREDICT)
        pred = extract_choice(reply)

        rows.append({"index": i, "true_answer": item[ANSWER_KEY],
                     "prediction": pred if pred is not None else "",
                     "unparseable": pred is None,
                     "correct": (pred == item[ANSWER_KEY]),
                     "prompt_tokens": ptok})

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"    {i + 1}/{len(questions)}  {rate:.2f} it/s  "
                  f"~{left:.0f} min left   ", end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n  k={topk}/{dataset_key}: accuracy {100 * df.correct.mean():.2f}%")
    return df


def mcnemar(a_ok, b_ok):
    g = int(((~a_ok) & b_ok).sum())
    l = int((a_ok & (~b_ok)).sum())
    d = 100.0 * (b_ok.mean() - a_ok.mean())
    p = 1.0 if g + l == 0 else float(binomtest(g, g + l, 0.5).pvalue)
    return round(d, 2), g, l, p


def token_report(df, topk, num_ctx, reserve):
    """Measured overflow, not estimated."""
    tok = pd.to_numeric(df.get("prompt_tokens"), errors="coerce").dropna()
    if not len(tok):
        return None
    budget = num_ctx - reserve
    return {
        "top_k": topk, "n": int(len(tok)),
        "mean_tokens": round(float(tok.mean()), 1),
        "median_tokens": float(tok.median()),
        "p95_tokens": round(float(tok.quantile(0.95)), 1),
        "max_tokens": int(tok.max()),
        "over_num_ctx_pct": round(100.0 * float((tok > num_ctx).mean()), 1),
        "over_budget_pct": round(100.0 * float((tok > budget).mean()), 1),
        "budget": budget,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--both", action="store_true",
                    help="also run MedMCQA (its prompts already fit; ~4 h)")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)
    num_ctx = rag_pipeline.OLLAMA_OPTIONS.get("num_ctx", 2048)

    print("=" * 72)
    print(f"TOP-K CONTROL - does stage C change when the prompt fits?")
    print(f"num_ctx={num_ctx}  num_predict={NUM_PREDICT}  "
          f"baseline k={BASELINE_TOPK}  control k={args.topk}")
    print("=" * 72)
    show_decoding()
    have_tokens = preflight(BASE_MODEL)

    datasets = ["medqa"] + (["medmcqa"] if args.both else [])
    summary, tokens = [], []

    for ds in datasets:
        questions = load_questions(ds)
        check_schema(questions, ds)
        print(f"\n{ds}: {len(questions)} items")

        new = run_arm(ds, questions, args.topk)

        base_path = os.path.join(RESULT_DIR, f"full_{ds}_C.csv")
        if not os.path.exists(base_path):
            print(f"  ! {base_path} missing - cannot pair against stage C")
            continue
        base = pd.read_csv(base_path).set_index("index")["correct"].astype(bool)
        cur = new.set_index("index")["correct"].astype(bool)

        j = pd.concat([base.rename("b"), cur.rename("n")], axis=1).dropna()
        d, g, l, p = mcnemar(j.b.astype(bool), j.n.astype(bool))

        summary.append({
            "dataset": ds, "n": len(j),
            f"acc_k{BASELINE_TOPK}": round(100 * j.b.mean(), 2),
            f"acc_k{args.topk}": round(100 * j.n.mean(), 2),
            "delta_pp": d, "gained": g, "lost": l,
            "mcnemar_p": round(p, 6), "significant": p < 0.05,
        })
        print(f"  k={BASELINE_TOPK} -> k={args.topk}: {d:+.2f} pp "
              f"(+{g}/-{l}, p={p:.4g})")

        if have_tokens:
            t = token_report(new, args.topk, num_ctx, NUM_PREDICT)
            if t:
                t["dataset"] = ds
                tokens.append(t)

    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(RESULT_DIR, "topk_control_summary.csv"), index=False)

    print("\n" + "=" * 72)
    print(out.to_string(index=False))

    if tokens:
        tk = pd.DataFrame(tokens)
        tk.to_csv(os.path.join(RESULT_DIR, "topk_token_report.csv"), index=False)
        print("\nMEASURED prompt tokens (from Ollama's prompt_eval_count):")
        print(tk[["dataset", "top_k", "n", "mean_tokens", "median_tokens",
                  "p95_tokens", "max_tokens", "budget",
                  "over_num_ctx_pct", "over_budget_pct"]].to_string(index=False))
        print("\n  over_budget_pct is the share of prompts leaving no room for")
        print("  the 150-token answer. At the control k this should be ~0.")

    print("=" * 72)
    print("""
HOW TO READ THIS

  not significant  ->  truncation was not driving the retrieval penalty.
                       Report as a robustness check and keep stage C as the
                       headline context arm. The hole is closed.

  significant      ->  part of the penalty was the context window, not
                       retrieval. Make k=%d the reported context arm, re-derive
                       the ladder against it, and say plainly that the original
                       configuration overflowed on a measured share of items.
""" % args.topk)


if __name__ == "__main__":
    main()
