"""
Robustness statistics: bootstrap CIs, multiplicity correction, random-gate baseline.
====================================================================================

WHY THIS EXISTS
---------------
The paper currently reports McNemar exact p-values and nothing else. Three
things are missing that a referee will ask for, and all three are computable
from CSVs already on disk. No model is called; this runs in under a minute on
any machine, GPU or not.

    1. CONFIDENCE INTERVALS.  A p-value says "not zero". It does not say how
       large the effect is or how precisely it is pinned down. Every accuracy
       delta gets a paired bootstrap 95% CI, and every gate delta gets an
       exact CI derived through the paper's own s* identity.

    2. MULTIPLICITY.  The paper runs roughly ten paired tests and calls
       p < 0.05 significant in each. Across a family that size the chance of at
       least one false positive is about 40%. Holm-Bonferroni fixes this and
       costs nothing, because the findings that matter have p-values many
       orders of magnitude below the threshold and survive any correction.

    3. A RANDOM BASELINE.  The gates are compared against the better fixed
       policy but never against chance. Without that, "the judge gate scored
       57.9%" has no reference point.

WHAT THE BOOTSTRAP RESAMPLES
----------------------------
Questions, not answers. Each replicate draws n question indices with
replacement and recomputes every statistic on that resample, so the CI
reflects sampling variability of the test set - the thing that would change if
the benchmark had drawn different questions. Stratification is on the paired
outcome pattern (both right, both wrong, only-A right, only-B right), which
keeps the discordant cell from vanishing in a replicate and is what makes the
interval for a McNemar-style delta stable.

Seeded with 42, matching every other stochastic component in this project.

USAGE
-----
    python run_robustness_stats.py                # 10,000 resamples
    python run_robustness_stats.py --n-boot 2000  # quicker
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULT_DIR = "evaluation/results"
SEED = 42
DATASETS = ("medqa", "medmcqa")


# ==========================================================  loading
def load_outcomes(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index("index")["correct"].astype(bool)


def paired(a, b):
    """Align two arms on the questions both answered. Returns numpy bools."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return j.a.astype(bool).values, j.b.astype(bool).values


# ==========================================================  bootstrap
def bootstrap_ci_at(a_ok, b_ok, stat_fn, n_boot, rng, alpha):
    """Paired bootstrap at an arbitrary level, for simultaneous intervals.

    A per-comparison 95% CI and a Holm-corrected p-value answer different
    questions, so on a marginal row they can disagree - which looks like an
    inconsistency in the results table. Reporting the family-wise interval at
    1 - alpha/m alongside the individual one removes the apparent conflict.
    """
    n = len(a_ok)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = stat_fn(a_ok[idx], b_ok[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(vals.std(ddof=1))


def bootstrap_ci(a_ok, b_ok, stat_fn, n_boot, rng):
    """Paired bootstrap over questions. Returns (lo, hi, se).

    Resampling is UNSTRATIFIED and deliberately so. Stratifying on the paired
    outcome pattern (both right / both wrong / only-A / only-B) and drawing
    within strata at fixed size holds every cell count constant, and a paired
    accuracy delta is a function of exactly those counts - so every replicate
    returns the identical value and the interval collapses to a point. That is
    a real trap: the first version of this script did it and produced
    zero-width CIs that looked like a formatting bug rather than a design
    error. The question, not the outcome pattern, is the resampling unit.
    """
    n = len(a_ok)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = stat_fn(a_ok[idx], b_ok[idx])

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float(vals.std(ddof=1))


def delta_pp(a_ok, b_ok):
    return 100.0 * (b_ok.mean() - a_ok.mean())


def mcnemar_p(a_ok, b_ok):
    g = int(((~a_ok) & b_ok).sum())
    l = int((a_ok & (~b_ok)).sum())
    if g + l == 0:
        return 1.0
    return float(binomtest(g, g + l, 0.5).pvalue)


# ==========================================================  Holm
def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p-values in input order.

    Uniformly more powerful than Bonferroni and makes no independence
    assumption, which matters here because the arms share a test set and their
    p-values are positively dependent.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)

    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


def fmt_p(p):
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"


# ==========================================================  gating
def gate_cells(no_ctx, ctx):
    """The four counts every gate statistic is built from."""
    a_only = int((no_ctx & (~ctx)).sum())     # no-context right, context wrong
    c_only = int(((~no_ctx) & ctx).sum())     # context right, no-context wrong
    return a_only, c_only, a_only + c_only


def gate_delta_stat(no_ctx, ctx, use_ctx):
    """Accuracy of a gated policy minus the better fixed policy, in pp."""
    gated = np.where(use_ctx, ctx, no_ctx)
    best_fixed = max(no_ctx.mean(), ctx.mean())
    return 100.0 * (gated.mean() - best_fixed)


def random_gate_baseline(no_ctx, ctx, rate, n_rep, rng):
    """Gate that keeps context on a random `rate` fraction of questions.

    This is the reference a learned gate has to beat before it is worth
    anything. Note it is NOT centred on zero: when the two arms differ in
    accuracy, random gating interpolates between them, so its expected delta
    is negative whenever the better fixed policy is the one being diluted.
    """
    n = len(no_ctx)
    k = int(round(rate * n))
    out = np.empty(n_rep, dtype=float)
    for i in range(n_rep):
        pick = np.zeros(n, dtype=bool)
        pick[rng.choice(n, size=k, replace=False)] = True
        out[i] = gate_delta_stat(no_ctx, ctx, pick)
    return out


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI. Used for gate skill s = k/d.

    Exact rather than bootstrapped because s is a plain binomial proportion
    over the d discordant questions - resampling it would only approximate a
    quantity with a closed form.
    """
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def delta_from_skill(s, d, a_only, c_only, n):
    """The paper's identity: gate delta in pp as a function of gate skill.

        delta = (100/n) * (s*d - max(a, c))

    Because it is linear in s, an exact CI on s maps straight onto an exact CI
    on the delta - no resampling needed, and tighter than a bootstrap would be.
    """
    return (100.0 / n) * (s * d - max(a_only, c_only))


# ==========================================================  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-rand", type=int, default=1_000)
    args = ap.parse_args()

    if not os.path.isdir(RESULT_DIR):
        sys.exit(f"{RESULT_DIR} not found - run this from the project root")

    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print(f"ROBUSTNESS STATISTICS  ({args.n_boot:,} bootstrap resamples, seed {SEED})")
    print("=" * 78)

    # ---------------------------------------------- 1. ladder with CIs
    print("\n1. PAIRED COMPARISONS WITH BOOTSTRAP 95% CI\n")

    steps = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("A", "D")]
    pairs = []

    for ds in DATASETS:
        arms = {k: load_outcomes(f"full_{ds}_{k}.csv") for k in "ABCD"}
        for x, y in steps:
            if arms[x] is not None and arms[y] is not None:
                pairs.append((ds, f"{x}->{y}") + paired(arms[x], arms[y]))

    m = len(pairs)
    fam_alpha = 0.05 / m

    rows = []
    for ds, step, a_ok, b_ok in pairs:
        d = delta_pp(a_ok, b_ok)
        lo, hi, se = bootstrap_ci(a_ok, b_ok, delta_pp, args.n_boot, rng)
        flo, fhi, _ = bootstrap_ci_at(a_ok, b_ok, delta_pp, args.n_boot,
                                      rng, fam_alpha)
        p = mcnemar_p(a_ok, b_ok)
        rows.append({
            "dataset": ds, "step": step, "n": len(a_ok),
            "delta_pp": round(d, 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "fam_ci_low": round(flo, 2), "fam_ci_high": round(fhi, 2),
            "se": round(se, 3),
            "gained": int(((~a_ok) & b_ok).sum()),
            "lost": int((a_ok & (~b_ok)).sum()),
            "p_raw": p,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "fam_excludes_zero": bool(flo > 0 or fhi < 0),
        })

    ladder = pd.DataFrame(rows)

    # ---------------------------------------------- 2. Holm
    ladder["p_holm"] = holm(ladder["p_raw"].values)
    ladder["sig_raw"] = ladder["p_raw"] < 0.05
    ladder["sig_holm"] = ladder["p_holm"] < 0.05

    show = ladder.copy()
    show["p_raw"] = show["p_raw"].map(fmt_p)
    show["p_holm"] = show["p_holm"].map(fmt_p)
    show["95% CI"] = ["[" + f"{r.ci_low:+.2f}, {r.ci_high:+.2f}" + "]"
                      for r in ladder.itertuples()]
    show["FW CI"] = ["[" + f"{r.fam_ci_low:+.2f}, {r.fam_ci_high:+.2f}" + "]"
                     for r in ladder.itertuples()]
    print(show[["dataset", "step", "n", "delta_pp", "95% CI", "FW CI", "se",
                "gained", "lost", "p_raw", "p_holm",
                "sig_raw", "sig_holm"]].to_string(index=False))

    flipped = ladder[ladder.sig_raw & ~ladder.sig_holm]
    print(f"\n  family size m = {m}   family-wise level = {fam_alpha:.4f}")
    print(f"  significant before correction: {int(ladder.sig_raw.sum())}")
    print(f"  significant after Holm       : {int(ladder.sig_holm.sum())}")
    if len(flipped):
        print("  LOST significance under correction:")
        for r in flipped.itertuples():
            print(f"    {r.dataset} {r.step}: p {fmt_p(r.p_raw)} -> {fmt_p(r.p_holm)}")
    else:
        print("  no conclusion changes under correction")

    agree_ind = int((ladder.sig_raw == ladder.excludes_zero).sum())
    agree_fam = int((ladder.sig_holm == ladder.fam_excludes_zero).sum())
    print(f"  per-comparison CI agrees with raw p on  {agree_ind}/{m} rows")
    print(f"  family-wise CI agrees with Holm p on    {agree_fam}/{m} rows")

    # ---------------------------------------------- 3. gate deltas with CIs
    print("\n\n2. GATE DELTAS WITH BOOTSTRAP 95% CI\n")

    judge_path = os.path.join(RESULT_DIR, "full_gating.csv")
    if not os.path.exists(judge_path):
        sys.exit(f"missing {judge_path}")
    judge_tbl = pd.read_csv(judge_path)

    gate_rows = []

    for _, jr in judge_tbl.iterrows():
        ds, arm = jr["dataset"], jr["context_arm"]
        A = load_outcomes(f"full_{ds}_A.csv")
        ctx_s = load_outcomes(f"full_{ds}_{arm}.csv")
        if A is None or ctx_s is None:
            continue

        no_ctx, ctx = paired(A, ctx_s)
        n = len(no_ctx)

        a_only, c_only, disc = gate_cells(no_ctx, ctx)
        s_star = 100.0 * max(a_only, c_only) / disc

        k = int(jr["judge_correct"])
        d_reported = int(jr["discordant"])
        skill = 100.0 * k / d_reported

        # The identity is exact, so verify it reproduces the measured delta
        # before using it to build an interval.
        measured = 100.0 * (jr["judge_gate_acc"]
                            - max(jr["acc_no_context"], jr["acc_always_context"]))
        predicted = delta_from_skill(k / d_reported, d_reported,
                                     a_only, c_only, n)

        # Exact CI on skill, mapped through the identity (linear in s).
        s_lo, s_hi = clopper_pearson(k, d_reported)
        d_lo = delta_from_skill(s_lo, d_reported, a_only, c_only, n)
        d_hi = delta_from_skill(s_hi, d_reported, a_only, c_only, n)

        rand = random_gate_baseline(no_ctx, ctx,
                                    float(jr["judge_says_useful_frac"]),
                                    args.n_rand, rng)

        gate_rows.append({
            "dataset": ds, "arm": arm, "n": n,
            "discordant": disc, "s_star_pct": round(s_star, 2),
            "judge_skill_pct": round(skill, 2),
            "skill_ci_low": round(100 * s_lo, 2),
            "skill_ci_high": round(100 * s_hi, 2),
            "beats_s_star": bool(skill > s_star),
            "s_star_in_skill_ci": bool(100 * s_lo <= s_star <= 100 * s_hi),
            "gate_delta_pp": round(predicted, 2),
            "measured_delta_pp": round(measured, 2),
            "identity_error": round(abs(predicted - measured), 4),
            "ci_low": round(d_lo, 2), "ci_high": round(d_hi, 2),
            "random_mean_pp": round(float(rand.mean()), 2),
            "random_sd": round(float(rand.std(ddof=1)), 3),
            "random_p95_pp": round(float(np.percentile(rand, 95)), 2),
            "beats_random": bool(predicted > np.percentile(rand, 95)),
            "skill_p": round(float(binomtest(k, d_reported, 0.5).pvalue), 6),
        })

    gates = pd.DataFrame(gate_rows)
    if len(gates):
        gates["skill_p_holm"] = holm(gates["skill_p"].values)

        g = gates.copy()
        g["skill 95% CI"] = ["[" + f"{r.skill_ci_low:.1f}, {r.skill_ci_high:.1f}" + "]"
                             for r in gates.itertuples()]
        g["delta 95% CI"] = ["[" + f"{r.ci_low:+.2f}, {r.ci_high:+.2f}" + "]"
                             for r in gates.itertuples()]
        print(g[["dataset", "arm", "n", "discordant", "s_star_pct",
                 "judge_skill_pct", "skill 95% CI", "beats_s_star",
                 "gate_delta_pp", "delta 95% CI"]].to_string(index=False))

        print("\n  identity check (max abs error, pp):",
              round(float(gates["identity_error"].max()), 5))

        print("\n\n3. GATE vs RANDOM GATING AT THE SAME SELECTION RATE\n")
        print(g[["dataset", "arm", "gate_delta_pp", "random_mean_pp",
                 "random_sd", "random_p95_pp", "beats_random",
                 "skill_p", "skill_p_holm"]].to_string(index=False))

        print("\n  s_star is the skill a gate must exceed to beat the better")
        print("  fixed policy. random_* summarises 1,000 random gates drawn at")
        print("  the judge's own selection rate; beats_random compares the gate")
        print("  against that distribution's 95th percentile.")

    # ---------------------------------------------- 4. save
    out1 = os.path.join(RESULT_DIR, "bootstrap_ladder.csv")
    out2 = os.path.join(RESULT_DIR, "bootstrap_gating.csv")
    ladder.to_csv(out1, index=False)
    if len(gates):
        gates.to_csv(out2, index=False)

    print(f"\n\nWrote {out1}")
    if len(gates):
        print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
