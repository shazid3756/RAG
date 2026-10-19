"""
Corpus-draw variance: how much does the retrieval effect move when the only
thing that changes is WHICH passages landed in the index?

    python summarise_draws.py

WHY THIS EXISTS
---------------
Every sweep level in this project was measured from a single random subset of
the passage vectors. The bootstrap intervals around those levels resample
QUESTIONS, so they answer "would another sample of exam items give the same
number?" - they say nothing about "would another sample of PASSAGES give the
same number?"

Four draws at 2.09% coverage answered that second question and the answer was
uncomfortable: -5.11, -5.26, -3.46, -1.81, an SD of 1.62 pp and a range of
3.46 pp. That is wider than the entire coverage trend the paper reports across
a full decade of corpus growth, and wider than the +2.67 pp "recovery" at the
top level that the paper had started to read as the curve turning.

So the trend needs the right error model, which needs several draws at EVERY
level rather than four draws at one. This script reads whatever draws exist
and reports the trend with a between-draw error bar instead of a within-draw
one.

WHAT IT PRINTS
--------------
Per level: each draw's effect, the mean, the SD, the range.
Per corpus: the slope on the per-level means, and its standard error derived
from the pooled between-draw SD - which is the honest error bar, because a
fresh draw is exactly what a replication would produce.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

R = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "evaluation", "results")

# (label, corpus_total, levels, {seed: filename pattern})
# Seed 42 is the draw already reported; the rest come from RUN_MULTISEED.ps1.
CORPORA = [
    ("100k corpus (MedQA)", 100_000, [10_000, 25_000, 50_000, 100_000],
     "full_medqa_A.csv",
     {42: "sweep_medqa_{L}.csv", 7: "draw7_medqa_{L}.csv",
      13: "draw13_medqa_{L}.csv", 99: "draw99_medqa_{L}.csv"}),
    ("1M corpus (MedQA)", 1_000_000, [100_000, 250_000, 500_000, 1_000_000],
     "full_medqa_A.csv",
     # The 1M-corpus extra draws reuse the seedvar tags on purpose: the
     # 500k level already exists under them from the first probe, so it
     # resumes as a no-op instead of costing another three hours.
     {42: "upsweep_medqa_{L}.csv", 7: "seedvar7_medqa_{L}.csv",
      13: "seedvar13_medqa_{L}.csv", 99: "seedvar99_medqa_{L}.csv"}),
    ("100k corpus (MedMCQA)", 100_000, [10_000, 25_000, 50_000, 100_000],
     "full_medmcqa_A.csv",
     {42: "sweep_medmcqa_{L}.csv"}),
]

# The one-off probe that started this: four draws at a single level.
PROBE = ("1M corpus, 500k level only", "full_medqa_A.csv",
         {42: "upsweep_medqa_500000.csv", 7: "seedvar7_medqa_500000.csv",
          13: "seedvar13_medqa_500000.csv", 99: "seedvar99_medqa_500000.csv"})


def correct(path):
    d = pd.read_csv(path).sort_values("index").set_index("index")
    return d["correct"].astype(bool)


def effect(base, arm):
    i = base.index.intersection(arm.index)
    a, b = base.loc[i].values, arm.loc[i].values
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return 100 * (b.mean() - a.mean()), p


def main():
    if not os.path.isdir(R):
        sys.exit(f"no results directory at {R}")

    print("=" * 74)
    print("CORPUS-DRAW VARIANCE")
    print("=" * 74)

    # ------------------------------------------------------------ the probe
    label, base_f, files = PROBE
    base_p = os.path.join(R, base_f)
    if os.path.exists(base_p):
        base = correct(base_p)
        vals = {}
        for s, f in files.items():
            p = os.path.join(R, f)
            if os.path.exists(p):
                vals[s], _ = effect(base, correct(p))
        if len(vals) > 1:
            v = np.array(list(vals.values()))
            print(f"\n{label}  ({len(v)} draws)")
            print("  " + "   ".join(f"seed {s}: {e:+.2f}" for s, e in vals.items()))
            print(f"  mean {v.mean():+.2f}   SD {v.std(ddof=1):.2f}   "
                  f"range {v.max() - v.min():.2f} pp")

    # ------------------------------------------------------------- corpora
    for label, total, levels, base_f, pats in CORPORA:
        base_p = os.path.join(R, base_f)
        if not os.path.exists(base_p):
            continue
        base = correct(base_p)

        rows = []
        for L in levels:
            per = {}
            for s, pat in pats.items():
                p = os.path.join(R, pat.format(L=L))
                if os.path.exists(p):
                    per[s], _ = effect(base, correct(p))
            if per:
                rows.append((L, per))
        if not rows:
            continue

        n_draws = max(len(p) for _, p in rows)
        print(f"\n{'-' * 74}\n{label}   -   up to {n_draws} draw(s) per level\n")
        header = f"{'level':>10}  {'cov':>7}  " + \
                 "".join(f"{'seed ' + str(s):>11}" for s in pats) + \
                 f"{'mean':>9}{'SD':>7}{'range':>8}"
        print(header)

        means, sds = [], []
        for L, per in rows:
            cells = "".join(f"{per[s]:+11.2f}" if s in per else f"{'-':>11}"
                            for s in pats)
            v = np.array(list(per.values()))
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            rng = (v.max() - v.min()) if len(v) > 1 else np.nan
            means.append(v.mean())
            if len(v) > 1:
                sds.append(sd)
            print(f"{L:>10,}  {100*L/23.9e6:>6.3f}%  {cells}"
                  f"{v.mean():+9.2f}"
                  f"{sd:>7.2f}" .replace("nan", "  -") +
                  (f"{rng:>8.2f}" if len(v) > 1 else f"{'-':>8}"))

        if len(means) < 3:
            continue
        x = np.log10([L for L, _ in rows])
        slope, intercept = np.polyfit(x, means, 1)
        pred = intercept + slope * x
        ss = ((np.array(means) - pred) ** 2).sum()
        tot = ((np.array(means) - np.mean(means)) ** 2).sum()
        r2 = 1 - ss / tot if tot else float("nan")

        print(f"\n  slope on the per-level means: {slope:+.2f} pp per decade "
              f"(R2 = {r2:.3f})")

        if sds:
            # Pooled between-draw SD, then the SE of a mean of k draws, then
            # the SE of the slope. This is the error bar that matters: a
            # replication of this study would draw fresh subsets, not fresh
            # exam questions.
            pooled = float(np.sqrt(np.mean(np.square(sds))))
            k = np.mean([len(p) for _, p in rows])
            se_level = pooled / np.sqrt(k)
            sxx = ((x - x.mean()) ** 2).sum()
            se_slope = se_level / np.sqrt(sxx)
            t = slope / se_slope
            print(f"  pooled between-draw SD: {pooled:.2f} pp "
                  f"({k:.1f} draws per level -> SE of a level mean "
                  f"{se_level:.2f})")
            print(f"  slope SE {se_slope:.2f}  ->  t = {t:+.2f}  "
                  f"{'SIGNIFICANT at .05' if abs(t) > 2.5 else 'NOT significant'}")
        else:
            print("  only one draw per level - no between-draw error bar is "
                  "possible.\n  Run RUN_MULTISEED.ps1 before quoting this "
                  "slope as a result.")

        neg = sum(1 for _, per in rows for e in per.values() if e < 0)
        tot_n = sum(len(per) for _, per in rows)
        print(f"  every measurement negative: {neg}/{tot_n}")

    print("\n" + "=" * 74)
    print("HOW TO READ THE SLOPE TEST")
    print("=" * 74)
    print("""
  The t here uses the BETWEEN-DRAW spread, not the bootstrap interval. The
  bootstrap resamples questions, which is the wrong reference for a claim
  about corpus size: two labs repeating this study would index different
  passages, not quiz different students.

  slope significant
      -> the coverage trend survives the right error model, and "inverse
         corpus scaling" is a result rather than an ordering.

  slope not significant, all effects still negative
      -> the honest claim is the level one: retrieval hurts everywhere
         measured, and there is no detectable trend with coverage. That is
         still a strong finding and it needs no scaling law to stand up.
""")


if __name__ == "__main__":
    main()
