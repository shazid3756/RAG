"""
Complete the RCT-corpus study: everything that needs no further generation.
==========================================================================

WHAT THIS COMPUTES, FOR FREE
----------------------------
Once stages A, C and (optionally) D exist as per-question outcome files, a
surprising amount follows by arithmetic alone:

    the ablation ladder over the RCT corpus
    every paired McNemar between consecutive stages
    the oracle ceiling for per-question gating
    s* = max(a,c)/d, the skill a gate must exceed to be worth deploying
    how often the two corpora disagree, item by item

No model is called. This runs in seconds on any machine, including one with
no GPU, because every number comes from CSVs already on disk.

WHY THE DISAGREEMENT COUNT MATTERS
----------------------------------
The headline result from the first RCT run was that swapping a 100,000-
abstract PubMed corpus for a 20,000-abstract RCT corpus - five times less
coverage, completely different composition - changed accuracy by nothing
detectable. The natural objection is that the two corpora must be returning
similar passages.

They are not. Roughly a fifth of all answers change. Different context,
different answers, same accuracy is the signature of context acting as
noise rather than as signal, and it is a stronger statement than the
accuracy comparison alone.

WHAT YOU STILL NEED THE GPU FOR
-------------------------------
    python run_rct_corpus.py --eval --finetuned

That produces stage D over the RCT corpus, about four hours. Without it this
script reports everything it can and marks the D rows as pending. With it,
the RCT replication of the ablation ladder is complete and Finding 3 can be
checked against a corpus it was never tuned on.

USAGE
-----
    python run_rct_analysis.py
"""

import os
import sys

import pandas as pd
from scipy.stats import binomtest

RESULT_DIR = "evaluation/results"


# ==========================================================  helpers
def load(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("index")["correct"].astype(bool)


def mcnemar(a, b):
    """Paired exact test. Returns (delta_pp, gained, lost, p)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    a2, b2 = j.a.astype(bool), j.b.astype(bool)
    gained = int(((~a2) & b2).sum())
    lost = int((a2 & (~b2)).sum())
    delta = 100.0 * (b2.mean() - a2.mean())
    p = 1.0 if gained + lost == 0 else binomtest(gained, gained + lost, 0.5).pvalue
    return round(delta, 2), gained, lost, float(p), len(j)


def gate_numbers(no_ctx, ctx):
    """Oracle ceiling and the break-even gate skill, from the paper's identity.

    On items where the two arms differ in correctness, a is the count the
    no-context arm alone gets right and c the count the context arm alone
    gets right. The better fixed policy collects max(a,c) of those d items;
    an oracle collects all d. A gate is worth deploying only when its skill
    exceeds s* = max(a,c)/d.
    """
    j = pd.concat([no_ctx.rename("a"), ctx.rename("c")], axis=1).dropna()
    a_ok, c_ok = j.a.astype(bool), j.c.astype(bool)
    n = len(j)

    a_only = int((a_ok & (~c_ok)).sum())     # no-context right, context wrong
    c_only = int(((~a_ok) & c_ok).sum())     # context right, no-context wrong
    d = a_only + c_only

    acc_no = 100 * a_ok.mean()
    acc_ctx = 100 * c_ok.mean()
    both = int((a_ok & c_ok).sum())
    acc_oracle = 100 * (both + d) / n
    best_fixed = max(acc_no, acc_ctx)

    return {
        "n": n,
        "acc_no_ctx": round(acc_no, 2),
        "acc_ctx": round(acc_ctx, 2),
        "acc_oracle": round(acc_oracle, 2),
        "headroom_pp": round(acc_oracle - best_fixed, 2),
        "discordant": d,
        "noctx_only": a_only,
        "ctx_only": c_only,
        "required_skill_pct": round(100 * max(a_only, c_only) / d, 2) if d else None,
    }


def fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"


# ==========================================================  report
def main():
    if not os.path.isdir(RESULT_DIR):
        sys.exit(f"{RESULT_DIR} not found - run this from the project root")

    datasets = ("medqa", "medmcqa")
    have_rct_d = all(load(f"rct_{ds}_D.csv") is not None for ds in datasets)

    print("=" * 74)
    print("RCT CORPUS - COMPLETE ANALYSIS")
    print("=" * 74)
    if not have_rct_d:
        print("\nStage D over the RCT corpus is missing. Run this for the full picture:")
        print("    python run_rct_corpus.py --eval --finetuned")
        print("Everything below that does not need D is still reported.\n")

    # ---------------------------------------------- 1. ladder
    print("\n1. ABLATION LADDER - accuracy (%)\n")
    ladder = []
    for ds in datasets:
        row = {"dataset": ds}
        A = load(f"full_{ds}_A.csv")
        if A is None:
            sys.exit(f"missing full_{ds}_A.csv - the main study must be run first")
        row["A no retrieval"] = round(100 * A.mean(), 2)
        for label, f in (("C 100k PubMed", f"full_{ds}_C.csv"),
                         ("C 20k RCT",     f"rct_{ds}_C.csv"),
                         ("D 100k PubMed", f"full_{ds}_D.csv"),
                         ("D 20k RCT",     f"rct_{ds}_D.csv")):
            s = load(f)
            row[label] = round(100 * s.mean(), 2) if s is not None else None
        ladder.append(row)
    print(pd.DataFrame(ladder).to_string(index=False))

    # ---------------------------------------------- 2. paired tests
    print("\n\n2. PAIRED COMPARISONS (McNemar exact)\n")
    rows = []
    pooled = {}
    for ds in datasets:
        A = load(f"full_{ds}_A.csv")
        for corpus, ctag, dtag in (("100k PubMed", f"full_{ds}_C.csv", f"full_{ds}_D.csv"),
                                   ("20k RCT",     f"rct_{ds}_C.csv",  f"rct_{ds}_D.csv")):
            C, D = load(ctag), load(dtag)
            for step, x, y in (("A->C", A, C), ("C->D", C, D)):
                if x is None or y is None:
                    continue
                delta, g, l, p, n = mcnemar(x, y)
                rows.append({"dataset": ds, "corpus": corpus, "step": step,
                             "delta_pp": delta, "gained": g, "lost": l,
                             "p": fmt_p(p), "sig": p < 0.05})
                key = (corpus, step)
                pooled.setdefault(key, [0, 0])
                pooled[key][0] += g
                pooled[key][1] += l
    for (corpus, step), (g, l) in pooled.items():
        p = binomtest(g, g + l, 0.5).pvalue if g + l else 1.0
        rows.append({"dataset": "POOLED", "corpus": corpus, "step": step,
                     "delta_pp": None, "gained": g, "lost": l,
                     "p": fmt_p(p), "sig": p < 0.05})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------- 3. corpora head to head
    print("\n\n3. DO THE TWO CORPORA DIFFER? (paired, identical items)\n")
    head = []
    tg = tl = 0
    for ds in datasets:
        for arm in ("C", "D"):
            x, y = load(f"full_{ds}_{arm}.csv"), load(f"rct_{ds}_{arm}.csv")
            if x is None or y is None:
                continue
            delta, g, l, p, n = mcnemar(x, y)
            tg += g
            tl += l
            head.append({"dataset": ds, "arm": arm, "n": n,
                         "delta_pp": delta, "changed_answer": g + l,
                         "changed_pct": round(100 * (g + l) / n, 1),
                         "p": fmt_p(p),
                         "verdict": "DIFFERENT" if p < 0.05 else "indistinguishable"})
    if head:
        print(pd.DataFrame(head).to_string(index=False))
        p = binomtest(tg, tg + tl, 0.5).pvalue if tg + tl else 1.0
        tot = sum(h["n"] for h in head)
        print(f"\n  pooled: +{tg}/-{tl}, p={fmt_p(p)}   "
              f"answers changed: {tg + tl} of {tot} ({100 * (tg + tl) / tot:.1f}%)")
        print("\n  A large share of answers changing while accuracy does not is what")
        print("  retrieved context behaving as noise rather than signal looks like.")

    # ---------------------------------------------- 4. gating
    print("\n\n4. SELECTIVE RETRIEVAL - ceiling and the bar a gate must clear\n")
    gates = []
    for ds in datasets:
        A = load(f"full_{ds}_A.csv")
        for corpus, tag in (("100k PubMed", f"full_{ds}_{{}}.csv"),
                            ("20k RCT", f"rct_{ds}_{{}}.csv")):
            for arm in ("C", "D"):
                ctx = load(tag.format(arm))
                if ctx is None:
                    continue
                g = gate_numbers(A, ctx)
                g.update({"dataset": ds, "corpus": corpus, "arm": arm})
                gates.append(g)
    if gates:
        cols = ["dataset", "corpus", "arm", "n", "acc_no_ctx", "acc_ctx",
                "acc_oracle", "headroom_pp", "discordant", "required_skill_pct"]
        print(pd.DataFrame(gates)[cols].to_string(index=False))
        print("\n  required_skill_pct is s* = max(a,c)/d. The LLM judge reached 53.0%")
        print("  on the 100k corpus. Any gate scoring below s* loses accuracy.")

    # ---------------------------------------------- save
    out = os.path.join(RESULT_DIR, "rct_full_analysis.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(gates).to_csv(
        os.path.join(RESULT_DIR, "rct_gating_analysis.csv"), index=False)
    print(f"\n\nWrote {out}")
    print(f"Wrote {os.path.join(RESULT_DIR, 'rct_gating_analysis.csv')}")

    print("""
WHAT TO LOOK FOR

  C->D significant on MedMCQA for the RCT corpus too
      -> Finding 3 replicates on a corpus it was never tuned against. That
         makes the fine-tuning result independent of which corpus is used,
         which is a stronger claim than the paper currently makes.

  headroom similar across corpora
      -> the oracle ceiling is a property of the model and the questions,
         not of the corpus. Further evidence that the retrieved text is not
         carrying usable signal.

  required_skill_pct similar across corpora
      -> the bar a gate must clear does not move when the corpus changes,
         so a gate trained on one corpus is worth testing on the other.
""")


if __name__ == "__main__":
    main()
