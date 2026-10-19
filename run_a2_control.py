"""
Stage A2 — the prompt control for the A -> B/C contrast.
=======================================================

WHY THIS EXISTS
---------------
Stage A (no retrieval) and stages B/C (with retrieval) do not use identical
prompts. They are word-for-word the same except for one instruction sentence,
which had to differ because one version refers to material that is not there:

    context arms : "Use the reference material if it helps, but rely on your
                    own medical knowledge as the primary basis for your
                    answer - the reference material is supplementary, not
                    exhaustive."
    stage A      : "Rely on your own medical knowledge to answer."

So A -> B measures (context present) + (that sentence changed), not context
alone. This script removes the ambiguity by running stage A2: the FULL context
prompt, verbatim, with the context string replaced by the placeholder
"(no reference material retrieved)" - which is exactly what the pipeline
already substitutes when retrieval returns nothing.

WHAT TO CONCLUDE
----------------
  A2 ~= A   -> the instruction sentence does not matter. Finding 4 is about
               retrieval, and the paper's central claim is clean.
  A2 != A   -> part of the A->B effect is prompt wording. Report A2->C as the
               corrected retrieval effect, because A2 and C differ ONLY in
               whether real passages replace the placeholder.

Either way the paper gets a defensible number. The second outcome is not a
disaster - it is a more precise result than we currently have.

COST
----
5,456 generations (1,273 MedQA + 4,183 MedMCQA), one pass, no retrieval
needed. About 3-4 hours on the RTX 3060 at ~2.5 s/item. Resumable: re-run it
after an interruption and it continues from the last checkpoint.

USAGE
-----
    Stop-Process -Name "OneDrive" -Force -ErrorAction SilentlyContinue
    python run_a2_control.py

Outputs, all under evaluation/results/:
    full_medqa_A2.csv, full_medmcqa_A2.csv   per-question outcomes
    a2_control_summary.csv                   the comparisons that matter
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
import time

import pandas as pd
from scipy.stats import binomtest

import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import extract_choice


# ==========================
# Configuration
# ==========================

BASE_MODEL = "qwen2.5:7b"
RESULT_DIR = "evaluation/results"
CHECKPOINT_EVERY = 20
NUM_PREDICT = 150

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

# The placeholder the pipeline already uses when retrieval returns nothing.
EMPTY_CONTEXT = "(no reference material retrieved)"

# Identical to Phases 3-4 so A2 is comparable with A, B, C and D.
GREEDY = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "repeat_penalty": 1.0,
}



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


def build_context_prompt(context_text, question, options):
    """The FULL context-arm prompt, byte-identical to mcqa_evaluation's.

    Kept here as a literal rather than imported so that this control cannot
    silently drift if the other file is edited. If you change the context
    prompt anywhere, change it here too - the whole point of this script is
    that A2 and C differ in nothing but the context string.
    """
    options_text = "\n".join(
        f"{letter}) {text}" for letter, text in sorted(options.items())
    )

    return f"""

You are a medical exam assistant.


========================

Reference Material (may or may not be directly relevant):

{context_text}


========================

Question:

{question}

Options:

{options_text}


Instructions:

Use the reference material if it helps, but rely on your own medical
knowledge as the primary basis for your answer — the reference
material is supplementary, not exhaustive.

First give a SHORT explanation in ONE sentence only (max ~25 words).
Then, on a new final line, output EXACTLY in this format (replace
<letter> with A, B, C, or D), with nothing else after it:

FINAL_ANSWER: <letter>

Answer:

"""


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


def preflight(model):
    """Fail in seconds rather than writing six hours of empty rows.

    An earlier evaluation in this project returned 0.00% on every arm because
    the model server was down and generate_answer returns an error STRING
    rather than raising. Never again.
    """
    print(f"Preflight: {model} ... ", end="", flush=True)
    reply = generate_answer(
        "Reply with exactly: OK", model=model, num_predict=8
    )
    if not isinstance(reply, str) or len(reply.strip()) == 0:
        sys.exit(f"\nFAILED: empty reply from {model}. Is ollama running?")
    low = reply.lower()
    if "error" in low or "connection" in low or "refused" in low:
        sys.exit(f"\nFAILED: {model} returned an error string: {reply[:200]}")
    print(f"ok ({reply.strip()[:40]!r})")


def load_questions(dataset_key):
    with open(DATASET_FILES[dataset_key], "r", encoding="utf-8") as f:
        return json.load(f)


def run_a2(dataset_key, questions):
    path = os.path.join(RESULT_DIR, f"full_{dataset_key}_A2.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): iterrows yields Series, and a
        # list mixing Series with the dicts appended below makes
        # pd.DataFrame() raise "'dict' object has no attribute 'dtype'".
        # This only ever bites on resume, which is why a fresh run looked fine.
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"  A2/{dataset_key}: already complete ({len(done)} items)")
            return pd.DataFrame(list(done.values()))
        print(f"  A2/{dataset_key}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    started = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue

        prompt = build_context_prompt(
            EMPTY_CONTEXT, item["question"], item["options"]
        )
        reply = generate_answer(
            prompt, model=BASE_MODEL, num_predict=NUM_PREDICT
        )
        pred = extract_choice(reply)
        unparseable = pred is None

        rows.append({
            "index": i,
            "true_answer": item[ANSWER_KEY],
            "prediction": pred,
            "unparseable": unparseable,
            "correct": (pred == item[ANSWER_KEY]),
            # Keep the raw text when parsing failed, so a bad run is
            # diagnosable afterwards instead of silently scored as wrong.
            "raw_on_failure": reply[:400] if unparseable else "",
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"    {i + 1}/{len(questions)}  {rate:.2f} it/s  "
                  f"~{left:.0f} min left", end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n  A2/{dataset_key}: done, "
          f"accuracy {100 * df.correct.mean():.2f}%, "
          f"unparseable {100 * df.unparseable.mean():.2f}%")
    return df


def mcnemar(a_ok, b_ok):
    """Exact paired test. Returns (delta_pp, gained, lost, p)."""
    gained = int(((~a_ok) & b_ok).sum())
    lost = int((a_ok & (~b_ok)).sum())
    n = len(a_ok)
    delta = 100.0 * (b_ok.mean() - a_ok.mean())
    if gained + lost == 0:
        return round(delta, 2), gained, lost, 1.0
    p = binomtest(gained, gained + lost, 0.5).pvalue
    return round(delta, 2), gained, lost, float(p)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)

    print("=" * 68)
    print("STAGE A2 - prompt control for the retrieval contrast")
    print("=" * 68)
    show_decoding()
    preflight(BASE_MODEL)

    summary = []

    for ds in ("medqa", "medmcqa"):
        questions = load_questions(ds)
        check_schema(questions, ds)
        print(f"\n{ds}: {len(questions)} items")
        a2 = run_a2(ds, questions).set_index("index")["correct"]

        for arm, label in [("A", "A2 vs A  (prompt wording alone)"),
                           ("C", "A2 vs C  (retrieval, cleanly isolated)")]:
            f = os.path.join(RESULT_DIR, f"full_{ds}_{arm}.csv")
            if not os.path.exists(f):
                print(f"  ! {f} missing, skipping {label}")
                continue
            other = pd.read_csv(f).set_index("index")["correct"]
            j = pd.concat([a2.rename("a2"), other.rename("o")],
                          axis=1).dropna()
            d, g, l, p = mcnemar(j.a2.astype(bool), j.o.astype(bool))
            summary.append({
                "dataset": ds, "comparison": f"A2->{arm}", "n": len(j),
                "acc_A2": round(100 * j.a2.mean(), 2),
                f"acc_{arm}": round(100 * j.o.mean(), 2),
                "delta_pp": d, "gained": g, "lost": l,
                "mcnemar_p": round(p, 6),
                "significant": p < 0.05,
            })
            print(f"  {label}: {d:+.2f} pp, +{g}/-{l}, p={p:.4g}")

    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(RESULT_DIR, "a2_control_summary.csv"), index=False)
    print("\n" + "=" * 68)
    print(out.to_string(index=False))
    print("=" * 68)
    print("""
HOW TO READ THIS

  A2->A not significant  ->  the instruction sentence changes nothing.
                             Report Finding 4 as it stands.
  A2->A significant      ->  part of the A->B effect was wording. Report
                             A2->C as the retrieval effect instead; it is
                             the clean contrast.

Either way, quote A2->C in the paper as the isolated effect of supplying
retrieved context, and say that A2 was the control that established it.
""")


if __name__ == "__main__":
    main()
