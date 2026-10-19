"""
Phase 2 under the Phase 3-4 protocol — one decoding regime for the whole paper.
==============================================================================

WHY
---
The ten-model generator comparison was run on 500 items per dataset under each
model's own default sampling settings, while Phases 3 and 4 use complete test
sets under greedy decoding with a fixed seed. That is the only place where the
paper uses two protocols, and it is the one inconsistency a reviewer can point
at without reading carefully.

The ordering is almost certainly safe: the gaps it rests on are 62.8 against
43.6 per cent, and parse-failure rates of 0 against 93 per cent, which no
plausible temperature effect reorders. But "almost certainly" is not a
measurement, and greedy decoding raised accuracy for every model we tested,
so the correction may not be uniform across models.

RUNTIME - READ THIS BEFORE STARTING
-----------------------------------
Full sets, all ten models, both datasets is 54,560 generations. Two of the
models take 8-9 s per item, so the whole thing is roughly 40-60 hours. That is
a lot of GPU time for a consistency fix.

Three sensible ways to cut it, in order of what we would choose:

  --retained-only     the seven models that survived, both datasets, full sets.
                      ~24 hours. The three excluded models failed on 82-95% of
                      items, which is not a marginal call and does not need
                      re-measuring at higher precision.

  --medqa-only        all ten, MedQA only (1,273 items). ~13 hours. Enough to
                      show the ordering is stable under greedy decoding.

  --limit 500         all ten, both datasets, same 500-item sample as before
                      but greedy. ~5 hours. This isolates the DECODING change
                      from the SAMPLE change, which is the cleaner experiment
                      if you only want to defend the existing table.

We would run --limit 500 first. If the ordering holds, the existing table is
defensible as-is and you can say so with evidence. Only then consider the
full-set version.

USAGE
-----
    python rerun_phase2_greedy.py --limit 500
    python rerun_phase2_greedy.py --retained-only
    python rerun_phase2_greedy.py                 # everything, slow
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import random
import sys
import time

import pandas as pd

import retriever
import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


RESULT_DIR = "evaluation/results"
EMBEDDING = "minilm"
MODE = "hybrid"
TOP_K = 8
NUM_PREDICT = 150
CHECKPOINT_EVERY = 20
SEED = 42

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0,
          "seed": 42, "repeat_penalty": 1.0}

RETAINED = [
    "llama3.1:8b",
    "gemma2:9b-instruct-q4_0",
    "qwen2.5:7b",
    "llama3.2:3b",
    "mistral:7b-instruct",
    "koesn/llama3-openbiollm-8b",
    "medllama2:7b",
]
EXCLUDED = ["phi3.5", "cniongolo/biomistral", "meditron:7b"]



# Gold answers live under "answer_idx", not "answer". The eval JSON has exactly
# these keys: question, options, answer_idx, source.
ANSWER_KEY = "answer_idx"


def check_schema(questions, name=""):
    """Fail immediately on a schema mismatch instead of hours into a run."""
    if not questions:
        raise SystemExit(f"FAILED: {name} is empty")
    missing = [k for k in ("question", "options", ANSWER_KEY)
               if k not in questions[0]]
    if missing:
        raise SystemExit(
            f"FAILED: {name} items are missing {missing}. "
            f"Found keys: {sorted(questions[0].keys())}")


def show_decoding():
    """Print the decoding settings actually in force.

    This project once reported results from two arms decoding under different
    rules without anyone noticing, so the settings get printed rather than
    assumed.
    """
    import rag_pipeline
    keys = ("temperature", "top_p", "top_k", "seed", "repeat_penalty",
            "num_ctx", "num_gpu")
    active = {k: rag_pipeline.OLLAMA_OPTIONS.get(k) for k in keys
              if k in rag_pipeline.OLLAMA_OPTIONS}
    print("Decoding in force:", ", ".join(f"{k}={v}" for k, v in active.items()))
    missing = [k for k in GREEDY if rag_pipeline.OLLAMA_OPTIONS.get(k) != GREEDY[k]]
    if missing:
        raise SystemExit(f"FAILED: greedy settings did not take effect: {missing}")


def preflight(models):
    """Check every model answers before committing hours to the run."""
    ok = []
    for m in models:
        print(f"Preflight {m} ... ", end="", flush=True)
        try:
            reply = generate_answer("Reply with exactly: OK", model=m,
                                    num_predict=8)
        except Exception as exc:
            print(f"EXCEPTION {exc}")
            continue
        if not isinstance(reply, str) or not reply.strip():
            print("empty reply - skipping this model")
            continue
        if any(w in reply.lower() for w in ("error", "connection", "refused",
                                            "not found", "try pulling")):
            print(f"error string - skipping ({reply[:60]!r})")
            continue
        print("ok")
        ok.append(m)
    if not ok:
        sys.exit("No model answered. Is ollama running?")
    return ok


def contexts_for(dataset_key, questions):
    """Retrieve once per question and cache, so every model sees identical text."""
    path = os.path.join(RESULT_DIR, f"phase2ctx_{dataset_key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(questions):
            print(f"  contexts cached ({len(cached)})")
            return cached

    out = []
    for i, item in enumerate(questions):
        res = retriever.retrieve_context(item["question"], k=TOP_K,
                                         mode=MODE, model_key=EMBEDDING)
        chunks = [str(r["chunk"]) for r in res]
        out.append("\n\n".join(chunks) if chunks
                   else "(no reference material retrieved)")
        if (i + 1) % 200 == 0:
            print(f"  retrieving {i + 1}/{len(questions)}", end="\r")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n  contexts written ({len(out)})")
    return out


def run_model(model, dataset_key, questions, contexts):
    safe = model.replace("/", "_").replace(":", "-")
    path = os.path.join(RESULT_DIR, f"p2greedy_{dataset_key}_{safe}.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): iterrows yields Series, and a
        # list mixing Series with the dicts appended below makes
        # pd.DataFrame() raise "'dict' object has no attribute 'dtype'".
        # This only ever bites on resume, which is why a fresh run looked fine.
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"    {model}: complete")
            return pd.DataFrame(list(done.values()))
        print(f"    {model}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    started = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue
        prompt = build_mcqa_prompt(contexts[i], item["question"], item["options"])
        reply = generate_answer(prompt, model=model,
                                num_predict=NUM_PREDICT)
        pred = extract_choice(reply)
        rows.append({"index": i, "true_answer": item[ANSWER_KEY],
                     "prediction": pred, "unparseable": pred is None,
                     "correct": (pred == item[ANSWER_KEY]),
                     "raw_on_failure": reply[:300] if pred is None else ""})

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"      {i + 1}/{len(questions)}  {rate:.2f} it/s  "
                  f"~{left:.0f} min left", end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n    {model}: acc {100 * df.correct.mean():.2f}%  "
          f"unparseable {100 * df.unparseable.mean():.2f}%")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retained-only", action="store_true")
    ap.add_argument("--medqa-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="sample this many items per dataset (seed 42), 0 = all")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)

    models = RETAINED if args.retained_only else RETAINED + EXCLUDED
    datasets = ["medqa"] if args.medqa_only else ["medqa", "medmcqa"]

    print("=" * 68)
    print("PHASE 2 RE-RUN under greedy decoding")
    print(f"models: {len(models)}   datasets: {datasets}   "
          f"limit: {args.limit or 'all'}")
    print("=" * 68)
    show_decoding()
    models = preflight(models)

    summary = []
    for ds in datasets:
        with open(DATASET_FILES[ds], "r", encoding="utf-8") as f:
            questions = json.load(f)
        check_schema(questions, ds)
        if args.limit and args.limit < len(questions):
            random.Random(SEED).shuffle(questions)
            questions = questions[:args.limit]
        print(f"\n{ds}: {len(questions)} items")

        contexts = contexts_for(ds, questions)
        for m in models:
            df = run_model(m, ds, questions, contexts)
            summary.append({
                "model": m, "dataset": ds, "n": len(df),
                "accuracy_pct": round(100 * df.correct.mean(), 2),
                "unparseable_pct": round(100 * df.unparseable.mean(), 2),
                "decoding": "greedy",
            })
            pd.DataFrame(summary).to_csv(
                os.path.join(RESULT_DIR, "phase2_greedy_summary.csv"),
                index=False)

    out = pd.DataFrame(summary)
    print("\n" + "=" * 68)
    print(out.pivot_table(index="model", columns="dataset",
                          values=["accuracy_pct", "unparseable_pct"]).to_string())
    print("=" * 68)
    print("""
HOW TO READ THIS

Compare the ordering against Table II of the paper, not the absolute values.

  ordering unchanged  ->  say so in the Limitations section and the existing
                          table stands. This is the expected outcome and it
                          costs one sentence to report.

  ordering changed    ->  replace Table II with these numbers and note that
                          the default-sampling version differed. That would
                          be a genuine correction, and better found by you
                          than by a referee.
""")


if __name__ == "__main__":
    main()
