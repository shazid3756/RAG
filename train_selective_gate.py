"""
A learned selective-retrieval gate, evaluated against the break-even bar.
========================================================================

THE BAR
-------
From the paper: on the items where the two arms differ in correctness, let a
be the number the no-context arm alone gets right, c the number the context
arm alone gets right, and d = a + c. A gate is worth deploying only when its
skill exceeds

    s* = max(a, c) / d

which in our data is 56.81, 51.97, 62.13 and 58.64 per cent for the four
settings, and 59.01 per cent pooled. The LLM judge reached 53.00 per cent and
therefore lost. This script asks whether a supervised model can clear the bar
using features the judge never saw.

HONEST EVALUATION
-----------------
The trap here is obvious and worth naming: it is trivial to "beat" the bar by
fitting the discordant items and reporting training accuracy. This script
therefore uses stratified 5-fold cross-validation and reports out-of-fold
skill only. It also reports a permutation baseline, so you can see what score
random labels achieve on the same features - if the real model is close to
that, the features carry nothing regardless of how good the number looks.

Nothing here is reported unless it survives cross-validation.

STAGE 1 - DUMP FEATURES  (needs the GPU; ~40 min for both datasets)
    python train_selective_gate.py --dump

    Writes gate_features_{dataset}.csv with, per question:
      top1, mean_topk, margin   retrieval confidence
      ctx_chars, ctx_chunks     how much context was supplied
      q_chars, q_words          question length
      n_options                 answer-option count
    The retrieval scores are recomputed with the same embedding and mode as
    stage C, so they match what the deployed gate would see.

STAGE 2 - TRAIN AND EVALUATE  (CPU only, seconds)
    python train_selective_gate.py --train

    Joins the features to the per-question outcomes already on disk, keeps
    the discordant items, and cross-validates a gradient-boosted tree and a
    logistic regression against s*.

USAGE
-----
    python train_selective_gate.py --dump      # once, on the machine with the indexes
    python train_selective_gate.py --train     # anywhere
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys

import numpy as np
import pandas as pd

RESULT_DIR = "evaluation/results"
EMBEDDING = "minilm"
MODE = "hybrid"
TOP_K = 8
SEED = 42

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

FEATURES = ["top1", "mean_topk", "margin", "ctx_chars", "ctx_chunks",
            "q_chars", "q_words", "n_options"]


# ==========================================================  STAGE 1
def dump_features():
    import retriever

    for ds, path in DATASET_FILES.items():
        out = os.path.join(RESULT_DIR, f"gate_features_{ds}.csv")
        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        done = 0
        rows = []
        if os.path.exists(out):
            prev = pd.read_csv(out)
            rows = prev.to_dict("records")
            done = len(rows)
            if done == len(questions):
                print(f"{ds}: features already complete")
                continue
            print(f"{ds}: resuming at {done}/{len(questions)}")

        for i, item in enumerate(questions):
            if i < done:
                continue

            res = retriever.retrieve_context(
                item["question"], k=TOP_K, mode=MODE, model_key=EMBEDDING
            )
            scores = [r["score"] for r in res] or [0.0]
            chunks = [str(r["chunk"]) for r in res]
            ctx = "\n\n".join(chunks)

            rows.append({
                "index": i,
                "top1": float(max(scores)),
                "mean_topk": float(np.mean(scores)),
                "margin": float(max(scores) - np.mean(scores)),
                "ctx_chars": len(ctx),
                "ctx_chunks": len(chunks),
                "q_chars": len(item["question"]),
                "q_words": len(item["question"].split()),
                "n_options": len(item["options"]),
            })

            if (i + 1) % 200 == 0:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  {i + 1}/{len(questions)}", end="\r")

        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n{ds}: wrote {out}")


# ==========================================================  STAGE 2
def load_setting(ds, arm):
    """Return the discordant items for one (dataset, arm) with features.

    label = 1 means the CONTEXT arm is the one that is right, so a gate
    predicting 1 should supply context.
    """
    fa = os.path.join(RESULT_DIR, f"full_{ds}_A.csv")
    fx = os.path.join(RESULT_DIR, f"full_{ds}_{arm}.csv")
    ff = os.path.join(RESULT_DIR, f"gate_features_{ds}.csv")
    for p in (fa, fx, ff):
        if not os.path.exists(p):
            sys.exit(f"missing {p} - run --dump first, and the main study before that")

    a = pd.read_csv(fa).set_index("index")["correct"].astype(bool)
    x = pd.read_csv(fx).set_index("index")["correct"].astype(bool)
    feats = pd.read_csv(ff).set_index("index")

    j = pd.concat([a.rename("a"), x.rename("x")], axis=1).dropna()
    disc = j[(j.a != j.x)]
    y = disc.x.astype(int)                       # 1 = context arm is correct
    X = feats.loc[disc.index, FEATURES]
    a_only = int((~disc.x).sum())
    c_only = int(disc.x.sum())
    s_star = max(a_only, c_only) / len(disc)
    return X, y, s_star, len(j), a_only, c_only


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    models = {
        "gradient boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED),
        "logistic regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    }

    rows = []
    rng = np.random.default_rng(SEED)

    for ds in ("medqa", "medmcqa"):
        for arm in ("C", "D"):
            X, y, s_star, n_all, a_only, c_only = load_setting(ds, arm)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            for name, make in models.items():
                pred = cross_val_predict(make(), X, y, cv=cv)
                skill = float((pred == y).mean())

                # permutation baseline: same pipeline, shuffled labels
                y_perm = pd.Series(rng.permutation(y.values), index=y.index)
                perm = cross_val_predict(make(), X, y_perm, cv=cv)
                perm_skill = float((perm == y_perm).mean())

                # what that skill is worth, via the paper's identity
                delta = 100.0 * (skill * len(y) - max(a_only, c_only)) / n_all

                rows.append({
                    "dataset": ds, "arm": arm, "model": name,
                    "discordant": len(y),
                    "required_skill_pct": round(100 * s_star, 2),
                    "cv_skill_pct": round(100 * skill, 2),
                    "permuted_skill_pct": round(100 * perm_skill, 2),
                    "margin_pp": round(100 * (skill - s_star), 2),
                    "implied_delta_pp": round(delta, 2),
                    "clears_bar": skill > s_star,
                })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULT_DIR, "learned_gate_summary.csv"), index=False)
    print(out.to_string(index=False))

    print("""
HOW TO READ THIS

  cv_skill close to permuted_skill
      -> the features carry no usable signal. This is the same conclusion the
         cosine threshold reached, now with a stronger model, which STRENGTHENS
         the paper: two model classes and eight features, still nothing.

  cv_skill > required_skill in some settings
      -> a real positive result. Report implied_delta_pp as the expected gain
         and re-run the gate end to end to confirm it materialises.

  cv_skill high but permuted_skill also high
      -> class imbalance is being exploited, not signal. Check the discordant
         split before believing anything.

Report the cross-validated number. Never the training-set number.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="compute features (needs indexes)")
    ap.add_argument("--train", action="store_true", help="cross-validate the gate")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    if not (args.dump or args.train):
        ap.error("choose --dump or --train")
    if args.dump:
        dump_features()
    if args.train:
        evaluate()


if __name__ == "__main__":
    main()
