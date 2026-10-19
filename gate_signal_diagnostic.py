"""
Is ANY cheap retrieval statistic able to predict when context helps?

adaptive_retrieval.py found a large oracle headroom (9-12 pp) but no gain
from thresholding dense top-1 cosine. Before investing in a more expensive
gate (e.g. asking the LLM to judge context relevance, ~45 min per run), test
directly whether the signals we already have carry ANY information.

METHOD
------
Restrict to the DISCORDANT questions -- those where the context arm and the
no-context arm disagree. Those are the only questions a gate can affect.
Label each one:

    HELPFUL  context arm right, no-context arm wrong   -> we want context
    HARMFUL  no-context arm right, context arm wrong   -> we want no context

Then measure ROC AUC of each candidate signal at separating the two.

    AUC ~ 0.50  the signal is noise; no threshold on it can ever work
    AUC > 0.60  usable signal; a smarter gate on it is worth building
    AUC < 0.50  signal is inverted (higher similarity predicts HARM)

This is decisive and costs nothing -- no LLM calls, all inputs already on
disk.

Run from the project root:
    python gate_signal_diagnostic.py
"""

import os

import numpy as np
import pandas as pd

RESULT_DIR = "evaluation/results"

DATASETS = ["medqa", "medmcqa"]

ARM_FILES = {
    "NO_CTX": "ablation_{dataset}_A_greedy.csv",
    "CTX":    "paired_{dataset}_qwen2.5_7b_greedy.csv",
    "CTX_FT": "paired_{dataset}_qwen2.5-medqa-lora_greedy.csv",
}


def auc(scores, labels):
    """ROC AUC via the rank (Mann-Whitney) identity. labels: 1 = HELPFUL."""

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)

    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()

    r_pos = ranks[:len(pos)].sum()

    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run(dataset, ctx_arm):

    gate_path = os.path.join(RESULT_DIR, f"gate_scores_{dataset}.csv")

    if not os.path.exists(gate_path):
        print(f"  missing {gate_path} -- run adaptive_retrieval.py first")
        return None

    gates = pd.read_csv(gate_path)

    arms = {}

    for name, pattern in ARM_FILES.items():
        path = os.path.join(RESULT_DIR, pattern.format(dataset=dataset))
        if not os.path.exists(path):
            print(f"  missing {path}")
            return None
        arms[name] = pd.read_csv(path)["correct"].astype(bool).values

    no_ctx = arms["NO_CTX"]
    ctx = arms[ctx_arm]

    helpful = ctx & ~no_ctx
    harmful = no_ctx & ~ctx

    mask = helpful | harmful
    labels = helpful[mask].astype(int)

    signals = {
        "top1 cosine": gates["top1"].values,
        "mean top-k cosine": gates["mean_topk"].values,
        "top1 - mean (margin)": (gates["top1"] - gates["mean_topk"]).values,
    }

    print(f"\n  {dataset}  |  context arm = {ctx_arm}")
    print(f"    discordant questions: {mask.sum()}  "
          f"(HELPFUL {helpful.sum()}, HARMFUL {harmful.sum()})")

    out = []

    for name, s in signals.items():

        a = auc(s[mask], labels)

        verdict = ("NOISE" if abs(a - 0.5) < 0.05
                   else "usable" if a > 0.5 else "INVERTED")

        print(f"    AUC {name:22s} = {a:.3f}   {verdict}")

        out.append({
            "dataset": dataset, "context_arm": ctx_arm,
            "signal": name, "auc": round(float(a), 4),
            "discordant": int(mask.sum()),
            "helpful": int(helpful.sum()), "harmful": int(harmful.sum()),
            "verdict": verdict,
        })

    return out


def main():

    print("=" * 72)
    print("GATE SIGNAL DIAGNOSTIC -- can retrieval confidence predict")
    print("whether context will help on a given question?")
    print("=" * 72)

    rows = []

    for dataset in DATASETS:
        for ctx_arm in ("CTX", "CTX_FT"):
            r = run(dataset, ctx_arm)
            if r:
                rows.extend(r)

    if not rows:
        return

    df = pd.DataFrame(rows)

    out = os.path.join(RESULT_DIR, "gate_signal_diagnostic.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(df.to_string(index=False))

    best = df["auc"].max()

    print(f"\nBest AUC across all signals and settings: {best:.3f}")

    if best < 0.55:
        print("\n-> No cheap retrieval statistic predicts when context helps.")
        print("   Thresholding retrieval scores CANNOT work, and this is a")
        print("   reportable finding in its own right: retrieval confidence")
        print("   and context usefulness are largely independent.")
        print("   Remaining option for a positive method: an LLM-judged gate")
        print("   (ask the model whether the retrieved passages actually")
        print("   answer the question) -- costs one extra generation call")
        print("   per question.")
    else:
        print("\n-> There is exploitable signal. A better gate on it is worth")
        print("   building (calibrated threshold per dataset, or a small")
        print("   classifier over several signals).")

    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
