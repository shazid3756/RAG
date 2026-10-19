"""
Full statistical treatment of the selective-retrieval gates.

run_full_study.py reports each gate's ACCURACY but not whether that accuracy
is significantly better than doing nothing. On MedQA the cosine gate scored
62.4% held-out against a 60.6% best fixed policy -- promising, but on n=636
(the held-out half) a +1.8 pp difference could easily be noise. This script
supplies the missing tests.

WHAT IT COMPUTES, per dataset x context arm:

  1. McNemar: cosine gate vs the better fixed policy, HELD-OUT half only
     (the threshold was tuned on the other half, so only this half is honest)
  2. McNemar: LLM-judge gate vs the better fixed policy, all items
     (nothing was tuned, so all items are reportable)
  3. Judge skill on discordant items, with Bonferroni correction across the
     four settings tested
  4. Pooled-across-datasets versions of all of the above

Reads only the per-question CSVs run_full_study.py already wrote. No LLM
calls, runs in seconds.

Run from the project root, AFTER run_full_study.py finishes:
    python gate_significance.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RESULT_DIR = "evaluation/results"
TAG = "full"

DATASET_FILES = {
    "medqa": "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

N_SETTINGS = 4          # 2 datasets x 2 context arms -> Bonferroni divisor
ALPHA = 0.05


def load(dataset, name):
    path = os.path.join(RESULT_DIR, f"{TAG}_{dataset}_{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def mcnemar(a_ok, b_ok):
    """b relative to a. Returns (gained, lost, p)."""
    g = int((b_ok & ~a_ok).sum())
    l = int((~b_ok & a_ok).sum())
    p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
    return g, l, p


def main():

    rows = []

    # accumulators for pooled tests
    pool = {
        "cos_g": 0, "cos_l": 0,
        "jud_g": 0, "jud_l": 0,
        "skill_k": 0, "skill_m": 0,
    }

    for dataset in DATASET_FILES:

        need = ["A", "C", "D", "JUDGE"]
        dfs = {n: load(dataset, n) for n in need}

        if any(v is None for v in dfs.values()):
            missing = [n for n in need if dfs[n] is None]
            print(f"[{dataset}] missing passes {missing} -- skipping")
            continue

        n_expected = len(json.load(open(DATASET_FILES[dataset], encoding="utf-8")))

        if any(len(v) != n_expected for v in dfs.values()):
            print(f"[{dataset}] incomplete: "
                  + ", ".join(f"{k}={len(v)}" for k, v in dfs.items())
                  + f" (expected {n_expected}) -- skipping")
            continue

        gate_path = os.path.join(RESULT_DIR, f"gate_scores_{TAG}_{dataset}.csv")
        scores = pd.read_csv(gate_path)["top1"].values

        ok = {k: dfs[k]["correct"].astype(bool).values for k in ("A", "C", "D")}
        use_judge = dfs["JUDGE"]["verdict"].values == "YES"

        n = n_expected
        idx = np.arange(n)
        tune, test = idx % 2 == 0, idx % 2 == 1

        for arm in ("C", "D"):

            no_ctx, ctx = ok["A"], ok[arm]

            # better fixed policy, decided on the SAME half we report on
            # (choosing it on the full set would leak)
            if no_ctx[test].mean() >= ctx[test].mean():
                fixed_name, fixed = "no-context", no_ctx
            else:
                fixed_name, fixed = "always-context", ctx

            # ---- 1. cosine gate, threshold tuned on `tune`, tested on `test`
            best_t, best_acc = None, -1.0
            for t in np.quantile(scores, np.linspace(0, 1, 41)):
                a = np.where(scores >= t, ctx, no_ctx)[tune].mean()
                if a > best_acc:
                    best_acc, best_t = a, t

            cos = np.where(scores >= best_t, ctx, no_ctx)
            cg, cl, cp = mcnemar(fixed[test], cos[test])

            # ---- 2. judge gate, nothing tuned -> all items
            jud = np.where(use_judge, ctx, no_ctx)

            if no_ctx.mean() >= ctx.mean():
                jfixed_name, jfixed = "no-context", no_ctx
            else:
                jfixed_name, jfixed = "always-context", ctx

            jg, jl, jp = mcnemar(jfixed, jud)

            # ---- 3. judge skill on discordant
            helpful, harmful = ctx & ~no_ctx, no_ctx & ~ctx
            disc = helpful | harmful
            calls = np.where(helpful, use_judge, ~use_judge)[disc]
            k, m = int(calls.sum()), int(disc.sum())
            sp = binomtest(k, m, 0.5).pvalue if m else 1.0

            pool["cos_g"] += cg; pool["cos_l"] += cl
            pool["jud_g"] += jg; pool["jud_l"] += jl
            pool["skill_k"] += k; pool["skill_m"] += m

            print(f"\n{'=' * 70}")
            print(f"{dataset}  |  context arm {arm}  (n={n})")
            print("=" * 70)
            print(f"  no-context {no_ctx.mean():.1%}   always-context {ctx.mean():.1%}"
                  f"   oracle {(no_ctx | ctx).mean():.1%}")
            print()
            print(f"  cosine gate (held-out n={test.sum()}, threshold {best_t:.4f})")
            print(f"    {cos[test].mean():.1%} vs {fixed_name} {fixed[test].mean():.1%}"
                  f"   {(cos[test].mean() - fixed[test].mean()) * 100:+.1f} pp"
                  f"   +{cg}/-{cl}  p={cp:.4f}"
                  f"{'  SIGNIFICANT' if cp < ALPHA else ''}")
            print()
            print(f"  LLM-judge gate (all n={n})")
            print(f"    {jud.mean():.1%} vs {jfixed_name} {jfixed.mean():.1%}"
                  f"   {(jud.mean() - jfixed.mean()) * 100:+.1f} pp"
                  f"   +{jg}/-{jl}  p={jp:.4f}"
                  f"{'  SIGNIFICANT' if jp < ALPHA else ''}")
            print()
            print(f"  judge skill on {m} discordant: {k}/{m} = {k / m:.1%}"
                  f"  p={sp:.4f}")
            print(f"    Bonferroni across {N_SETTINGS} settings: "
                  f"alpha={ALPHA / N_SETTINGS:.4f} -> "
                  f"{'SURVIVES' if sp < ALPHA / N_SETTINGS else 'does NOT survive'}")

            rows.append({
                "dataset": dataset, "context_arm": arm, "n": n,
                "acc_no_context": round(float(no_ctx.mean()), 4),
                "acc_always_context": round(float(ctx.mean()), 4),
                "acc_oracle": round(float((no_ctx | ctx).mean()), 4),
                "cos_threshold": round(float(best_t), 4),
                "cos_heldout_acc": round(float(cos[test].mean()), 4),
                "cos_vs_fixed_pp": round(
                    float((cos[test].mean() - fixed[test].mean()) * 100), 2),
                "cos_gained": cg, "cos_lost": cl, "cos_p": round(float(cp), 5),
                "judge_acc": round(float(jud.mean()), 4),
                "judge_vs_fixed_pp": round(
                    float((jud.mean() - jfixed.mean()) * 100), 2),
                "judge_gained": jg, "judge_lost": jl, "judge_p": round(float(jp), 5),
                "discordant": m, "judge_correct": k,
                "judge_skill_pct": round(100.0 * k / m, 2) if m else None,
                "judge_skill_p": round(float(sp), 5),
                "judge_skill_survives_bonferroni": bool(sp < ALPHA / N_SETTINGS),
            })

    if not rows:
        print("\nNothing complete yet.")
        return

    print(f"\n{'=' * 70}")
    print("POOLED ACROSS DATASETS AND ARMS")
    print("=" * 70)

    for label, g, l in [
        ("cosine gate vs fixed (held-out)", pool["cos_g"], pool["cos_l"]),
        ("LLM-judge gate vs fixed", pool["jud_g"], pool["jud_l"]),
    ]:
        p = binomtest(g, g + l, 0.5).pvalue if (g + l) else 1.0
        print(f"  {label:34s} +{g}/-{l}  p={p:.5f}"
              f"{'  SIGNIFICANT' if p < ALPHA else ''}")

    k, m = pool["skill_k"], pool["skill_m"]
    p = binomtest(k, m, 0.5).pvalue if m else 1.0
    print(f"  {'judge skill on discordant':34s} {k}/{m} = {k / m:.1%}"
          f"  p={p:.5f}{'  SIGNIFICANT' if p < ALPHA else ''}")

    out = os.path.join(RESULT_DIR, f"{TAG}_gate_significance.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
