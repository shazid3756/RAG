"""
Cross-encoder reranking - the direct test of the distractor hypothesis.
=======================================================================

WHY THIS IS THE RIGHT NEXT EXPERIMENT
-------------------------------------
The coverage sweep produced a clean monotone result in the direction opposite
to the paper's original explanation: the retrieval penalty GROWS as the corpus
grows (-3.77 pp at 10k abstracts to -5.50 pp at 100k, Spearman rho = -1.00,
R^2 = 0.966).

The reading that fits is that a larger corpus does not meaningfully raise the
chance the answer is present - coverage is still 0.4% - but it steadily raises
how PLAUSIBLE the top-ranked wrong passages look. A convincing distractor
costs more than an obviously irrelevant one.

If that is right, the fix is not more corpus and not a different bi-encoder.
It is a second scoring stage that reads query and passage TOGETHER. A
bi-encoder embeds them independently and can only measure topical proximity,
which is exactly the quantity that improves with corpus size while remaining
useless for answering. A cross-encoder attends across the pair and can tell
"about this topic" from "contains this answer".

So this experiment tests the mechanism, not just the metric.

WHAT IT DOES
------------
Two phases, deliberately separated because both stages want the GPU:

  --rerank   retrieve CANDIDATE_POOL passages per question with the existing
             hybrid retriever, score every (question, passage) pair with a
             cross-encoder, keep the top FINAL_K, and cache the resulting
             context strings. Ollama is idle throughout.

  --eval     generate answers from the cached contexts with the same model,
             prompt and greedy settings as stage C. The cross-encoder is not
             loaded, so VRAM is not contended.

Running both in one process on a 12GB card is what would OOM, hence the split.

MODEL CHOICE
------------
Default is NCBI's MedCPT cross-encoder, which was trained on real PubMed
search logs - the closest thing to a purpose-built reranker for this corpus.
If it cannot be fetched, the script falls back to the ms-marco MiniLM
cross-encoder, which is the standard general-domain baseline. Whichever is
used is recorded in the summary, because it matters for interpretation.

HOW TO READ THE RESULT
----------------------
  rerank significantly better than C, and at or above stage A
      -> the distractor hypothesis is supported AND you have a fix. The paper
         gains a positive method contribution: retrieval hurts because ranking
         by similarity surfaces plausible non-answers, and joint scoring
         repairs it.

  rerank better than C but still below A
      -> partial support. Reranking removes some of the damage but retrieval
         still is not worth its cost at this coverage. A more precise negative
         result than the paper currently has.

  rerank no better than C
      -> the damage is not about ranking quality at all. That is a strong
         statement too: it says the problem is the corpus, not the retriever,
         and it forecloses the most obvious objection a referee will raise.

All three outcomes are publishable. The experiment cannot fail, only inform.

COST
----
Reranking: 5,456 questions x 30 candidates = 163,680 pairs, a few minutes on
the 3060 for MiniLM, ~20 minutes for MedCPT.
Generation: 5,456 items at ~2.5 s = about 4 hours.

USAGE
-----
    python run_rerank.py --rerank            # phase 1, both datasets
    python run_rerank.py --eval              # phase 2
    python run_rerank.py --rerank --eval     # sequentially, one command
    python run_rerank.py --rerank --medqa-only
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import retriever
import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


# ==========================
# Configuration
# ==========================

BASE_MODEL = "qwen2.5:7b"
RESULT_DIR = "evaluation/results"

# Stage C's retrieval, used to produce the candidate pool.
EMBEDDING = "minilm"
MODE = "hybrid"

CANDIDATE_POOL = 30      # how many passages the reranker sees
FINAL_K = 8              # how many survive into the prompt

# Preferred first. The fallback is the standard general-domain reranker.
RERANKER_CANDIDATES = [
    "ncbi/MedCPT-Cross-Encoder",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

RERANK_BATCH = 64
NUM_PREDICT = 150
CHECKPOINT_EVERY = 20

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0,
          "seed": 42, "repeat_penalty": 1.0}

ANSWER_KEY = "answer_idx"

TAG = "rerank"


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


def preflight(model):
    print(f"Preflight: {model} ... ", end="", flush=True)
    reply = generate_answer("Reply with exactly: OK", model=model, num_predict=8)
    if not isinstance(reply, str) or not reply.strip():
        sys.exit(f"\nFAILED: empty reply from {model}. Is ollama running?")
    if any(w in reply.lower() for w in ("error", "connection", "refused")):
        sys.exit(f"\nFAILED: {model} returned an error string: {reply[:200]}")
    print("ok")


def load_questions(key):
    with open(DATASET_FILES[key], "r", encoding="utf-8") as f:
        return json.load(f)


def ctx_path(dataset_key):
    return os.path.join(RESULT_DIR, f"contexts_{TAG}_{dataset_key}.json")


def meta_path():
    return os.path.join(RESULT_DIR, f"{TAG}_meta.json")


# ==========================
# Phase 1 - rerank
# ==========================

def load_reranker():
    """Try each candidate in order; report which one was actually used.

    Which reranker ran changes how the result should be read - a biomedical
    cross-encoder succeeding says something different from a general-domain
    one succeeding - so this is recorded rather than assumed.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    errors = []
    for name in RERANKER_CANDIDATES:
        try:
            print(f"  loading reranker {name} on {device} ...", flush=True)
            model = CrossEncoder(name, max_length=512, device=device)
            print(f"  using {name}")
            return model, name, device
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"  could not load {name} ({type(exc).__name__})")
    raise SystemExit("No reranker could be loaded:\n  " + "\n  ".join(errors))


def rerank_dataset(dataset_key, questions, model, model_name, final_k=FINAL_K):
    """Retrieve a candidate pool, rescore it jointly, cache the top final_k.

    final_k is a parameter rather than a rebound module global: reassigning
    the global from main() is a SyntaxError once argparse has already read it
    for the flag's default, and passing it explicitly is clearer anyway.
    """
    path = ctx_path(dataset_key)

    cached = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(questions):
            print(f"  {dataset_key}: contexts already cached ({len(cached)})")
            return cached
        print(f"  {dataset_key}: resuming rerank at {len(cached)}/{len(questions)}")

    out = list(cached)
    stats = []
    started = time.time()

    for i in range(len(out), len(questions)):
        q = questions[i]["question"]

        res = retriever.retrieve_context(q, k=CANDIDATE_POOL, mode=MODE,
                                         model_key=EMBEDDING)
        passages = [str(r["chunk"]) for r in res]

        if not passages:
            out.append("(no reference material retrieved)")
            stats.append({"index": i, "n_candidates": 0, "overlap_at_k": 0})
            continue

        scores = model.predict([(q, p) for p in passages],
                               batch_size=RERANK_BATCH,
                               show_progress_bar=False)
        order = np.argsort(np.asarray(scores))[::-1][:final_k]

        # How much the reranker actually changed: overlap between what the
        # bi-encoder would have kept and what survives now. If this is ~FINAL_K
        # the reranker is a no-op and a null result means nothing.
        overlap = len(set(order.tolist()) & set(range(min(final_k, len(passages)))))

        out.append("\n\n".join(passages[j] for j in order))
        stats.append({"index": i, "n_candidates": len(passages),
                      "overlap_at_k": overlap})

        if (i + 1) % 100 == 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
            rate = (i + 1 - len(cached)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"    reranked {i + 1}/{len(questions)}  "
                  f"~{left:.0f} min left   ", end="\r")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    if stats:
        sdf = pd.DataFrame(stats)
        sdf.to_csv(os.path.join(RESULT_DIR, f"{TAG}_stats_{dataset_key}.csv"),
                   index=False)
        mean_ov = sdf.overlap_at_k.mean()
        print(f"\n  {dataset_key}: cached {len(out)} contexts. "
              f"mean overlap with the bi-encoder top-{final_k}: "
              f"{mean_ov:.2f}/{final_k} "
              f"({100 * mean_ov / final_k:.0f}% unchanged)")
        if mean_ov > final_k - 0.5:
            print("  WARNING: the reranker barely changed the ranking. A null "
                  "result here would say nothing about reranking.")
    return out


# ==========================
# Phase 2 - generate
# ==========================

def run_eval(dataset_key, questions, contexts):
    path = os.path.join(RESULT_DIR, f"{TAG}_{dataset_key}_C.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): mixing Series with dicts makes
        # pd.DataFrame() raise "'dict' object has no attribute 'dtype'".
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"  {dataset_key}: already complete ({len(done)})")
            return pd.DataFrame(list(done.values()))
        print(f"  {dataset_key}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    started = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue
        prompt = build_mcqa_prompt(contexts[i], item["question"], item["options"])
        reply = generate_answer(prompt, model=BASE_MODEL, num_predict=NUM_PREDICT)
        pred = extract_choice(reply)

        rows.append({"index": i, "true_answer": item[ANSWER_KEY],
                     "prediction": pred if pred is not None else "",
                     "unparseable": pred is None,
                     "correct": (pred == item[ANSWER_KEY])})

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"    {i + 1}/{len(questions)}  {rate:.2f} it/s  "
                  f"~{left:.0f} min left   ", end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n  {dataset_key}: accuracy {100 * df.correct.mean():.2f}%")
    return df


def mcnemar(a_ok, b_ok):
    g = int(((~a_ok) & b_ok).sum())
    l = int((a_ok & (~b_ok)).sum())
    d = 100.0 * (b_ok.mean() - a_ok.mean())
    p = 1.0 if g + l == 0 else float(binomtest(g, g + l, 0.5).pvalue)
    return round(d, 2), g, l, p


# ==========================
# Main
# ==========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="phase 1")
    ap.add_argument("--eval", action="store_true", help="phase 2")
    ap.add_argument("--medqa-only", action="store_true")
    ap.add_argument("--final-k", type=int, default=FINAL_K)
    args = ap.parse_args()

    if not (args.rerank or args.eval):
        ap.error("choose --rerank, --eval, or both")

    os.makedirs(RESULT_DIR, exist_ok=True)
    datasets = ["medqa"] if args.medqa_only else ["medqa", "medmcqa"]

    print("=" * 72)
    print("CROSS-ENCODER RERANKING")
    print(f"candidate pool {CANDIDATE_POOL} -> final {args.final_k}   "
          f"retriever {EMBEDDING}/{MODE}")
    print("=" * 72)

    # ---------------- phase 1
    if args.rerank:
        model, model_name, device = load_reranker()
        for ds in datasets:
            questions = load_questions(ds)
            check_schema(questions, ds)
            print(f"\n{ds}: {len(questions)} questions")
            rerank_dataset(ds, questions, model, model_name,
                           final_k=args.final_k)

        with open(meta_path(), "w", encoding="utf-8") as f:
            json.dump({"reranker": model_name, "device": device,
                       "candidate_pool": CANDIDATE_POOL, "final_k": args.final_k},
                      f, indent=2)

        # Free the cross-encoder before any generation starts. Both want the
        # same 12GB, and holding this while Ollama loads a 7B is how you OOM.
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        print("\nReranker unloaded, VRAM released.")

    # ---------------- phase 2
    if not args.eval:
        print("\nPhase 1 done. Now run:  python run_rerank.py --eval")
        return

    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)
    show_decoding()
    preflight(BASE_MODEL)

    meta = {}
    if os.path.exists(meta_path()):
        with open(meta_path(), "r", encoding="utf-8") as f:
            meta = json.load(f)
    print(f"reranker used: {meta.get('reranker', 'unknown')}")

    summary = []
    for ds in datasets:
        questions = load_questions(ds)
        check_schema(questions, ds)

        p = ctx_path(ds)
        if not os.path.exists(p):
            print(f"  ! {p} missing - run --rerank first")
            continue
        with open(p, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        if len(contexts) != len(questions):
            print(f"  ! {ds}: {len(contexts)} cached contexts for "
                  f"{len(questions)} questions - re-run --rerank")
            continue

        print(f"\n{ds}: {len(questions)} items")
        new = run_eval(ds, questions, contexts).set_index("index")["correct"].astype(bool)

        row = {"dataset": ds, "n": int(len(new)),
               "reranker": meta.get("reranker", "unknown"),
               "acc_rerank": round(100 * new.mean(), 2)}

        for label, fn in (("A", f"full_{ds}_A.csv"), ("C", f"full_{ds}_C.csv")):
            f_ = os.path.join(RESULT_DIR, fn)
            if not os.path.exists(f_):
                continue
            other = pd.read_csv(f_).set_index("index")["correct"].astype(bool)
            j = pd.concat([other.rename("o"), new.rename("r")], axis=1).dropna()
            d, g, l, pv = mcnemar(j.o.astype(bool), j.r.astype(bool))
            row[f"acc_{label}"] = round(100 * j.o.mean(), 2)
            row[f"vs_{label}_pp"] = d
            row[f"vs_{label}_p"] = round(pv, 6)
            row[f"vs_{label}_sig"] = pv < 0.05
            print(f"  {label} -> rerank: {d:+.2f} pp (+{g}/-{l}, p={pv:.4g})")

        summary.append(row)

    if summary:
        out = pd.DataFrame(summary)
        out.to_csv(os.path.join(RESULT_DIR, "rerank_summary.csv"), index=False)
        print("\n" + "=" * 72)
        print(out.to_string(index=False))
        print("=" * 72)
        print("""
HOW TO READ THIS

  vs_C positive and significant, and acc_rerank >= acc_A
      -> plausible distractors were the mechanism, and joint query-passage
         scoring fixes it. This is a positive method contribution: report it
         as the remedy for Finding 4.

  vs_C positive but acc_rerank still < acc_A
      -> reranking recovers part of the damage; retrieval still does not pay
         for itself at this coverage. Report as a sharper negative result.

  vs_C not significant
      -> ranking quality is not the problem. The corpus is. This forecloses
         the most obvious objection to Finding 4 and is worth a paragraph.

Check the mean overlap printed during --rerank before interpreting a null:
if the reranker kept nearly the same eight passages, it never had a chance
to matter.
""")


if __name__ == "__main__":
    main()
