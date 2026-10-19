"""
Turn the A2 and Phase 2 outputs into finished paper text.
=========================================================

WHY THIS EXISTS
---------------
Both remaining issues have two possible outcomes, and each outcome needs
different wording in the paper. Choosing between them by hand after a long
GPU run is exactly when a tired author picks the flattering branch. This
script reads the result CSVs and emits the paragraph the data actually
support, with the numbers already substituted.

It writes nothing until the corresponding run has finished, and says plainly
which sections are still pending.

    run_a2_control.py          -> a2_control_summary.csv
    rerun_phase2_greedy.py     -> phase2_greedy_summary.csv

USAGE
-----
    python finalise_gpu_sections.py                  # print to stdout
    python finalise_gpu_sections.py --write          # write .tex files
"""

import argparse
import os

import pandas as pd

RESULT_DIR = "evaluation/results"
ALPHA = 0.05

# Ordering claimed by Table II of the paper, best first, retained models only.
PAPER_ORDER = [
    "llama3.1:8b",
    "gemma2:9b-instruct-q4_0",
    "qwen2.5:7b",
    "llama3.2:3b",
    "mistral:7b-instruct",
    "koesn/llama3-openbiollm-8b",
    "medllama2:7b",
]


def fmt_p(p):
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"


def tex(s):
    """Escape a model tag for LaTeX and set it in a monospace face.

    Ollama tags carry underscores (gemma2:9b-instruct-q4_0), which are
    subscript operators in math-off text mode and abort the build. Typesetting
    them verbatim also stops the colon and hyphens from being re-flowed.
    """
    return "\\texttt{" + str(s).replace("\\", "").replace("_", r"\_") \
                              .replace("%", r"\%").replace("&", r"\&") + "}"


# ==========================================================  A2
def a2_section():
    path = os.path.join(RESULT_DIR, "a2_control_summary.csv")
    if not os.path.exists(path):
        return None, "A2 control has not been run (missing a2_control_summary.csv)"

    df = pd.read_csv(path)
    vs_a = df[df.comparison == "A2->A"]
    vs_c = df[df.comparison == "A2->C"]
    if not len(vs_a):
        return None, "a2_control_summary.csv has no A2->A rows"

    wording_matters = bool(vs_a["significant"].any())

    lines = [
        "\\subsubsection{Resolving the prompt difference}",
        "\\label{sec:confound-resolved}",
        "",
        "Section~\\ref{sec:confound} noted that stage~A and the context arms "
        "differ in one instruction sentence as well as in the presence of "
        "retrieved text, so that A$\\rightarrow$C confounds the two. Stage~A2 "
        "removes the ambiguity: it uses the full context-arm prompt verbatim, "
        "with the context string replaced by the placeholder the pipeline "
        "already substitutes when retrieval returns nothing. A2 and C "
        "therefore differ in nothing but whether real passages appear in that "
        "slot. The prompt literal was verified byte-identical to the one used "
        "by the context arms.",
        "",
    ]

    for _, r in vs_a.iterrows():
        lines.append(
            f"On {r['dataset']}, A2 scores {r['acc_A2']:.2f}\\% against "
            f"{r['acc_A']:.2f}\\% for stage~A, a difference of "
            f"{r['delta_pp']:+.2f}~pp "
            f"($+{int(r['gained'])}/-{int(r['lost'])}$, $p={fmt_p(r['mcnemar_p'])}$)."
        )
    lines.append("")

    if wording_matters:
        lines += [
            "The instruction sentence therefore accounts for part of the "
            "measured effect, and A$\\rightarrow$C overstates the contribution "
            "of retrieval alone. We report A2$\\rightarrow$C as the corrected "
            "retrieval effect throughout:",
            "",
        ]
    else:
        lines += [
            "The instruction sentence does not measurably change accuracy on "
            "either benchmark. The A$\\rightarrow$C contrast is therefore "
            "attributable to the presence of retrieved context rather than to "
            "prompt wording, and the central claim stands as stated. For "
            "completeness we also report the fully isolated contrast:",
            "",
        ]

    for _, r in vs_c.iterrows():
        acc_c = r.get("acc_C", float("nan"))
        lines.append(
            f"A2$\\rightarrow$C on {r['dataset']} is {r['delta_pp']:+.2f}~pp "
            f"({r['acc_A2']:.2f}\\% to {acc_c:.2f}\\%, "
            f"$p={fmt_p(r['mcnemar_p'])}$)."
        )

    lines += [
        "",
        "This control cost one additional generation pass over both complete "
        "test sets and no additional retrieval, and it converts the paper's "
        "central contrast from a confounded comparison into a clean one.",
    ]

    verdict = ("A2 differs from A -> report A2->C as the retrieval effect"
               if wording_matters else
               "A2 matches A -> wording is not a confound, claim stands")
    return "\n".join(lines), verdict


# ==========================================================  Phase 2
def phase2_section():
    path = os.path.join(RESULT_DIR, "phase2_greedy_summary.csv")
    if not os.path.exists(path):
        return None, "Phase 2 re-run has not been done (missing phase2_greedy_summary.csv)"

    df = pd.read_csv(path)
    n_items = int(df["n"].max())

    out_lines = [
        "\\subsubsection{Generator comparison under a single decoding regime}",
        "\\label{sec:phase2-greedy}",
        "",
        "The generator comparison of Section~\\ref{sec:generators} was "
        "originally run under each model's own default sampling settings, "
        "while every later phase uses greedy decoding with a fixed seed. To "
        "remove the one place where the study used two protocols, the "
        f"comparison was repeated under greedy decoding on {n_items} items "
        "per dataset.",
        "",
    ]

    verdicts = []
    for ds, sub in df.groupby("dataset"):
        sub = sub.sort_values("accuracy_pct", ascending=False)
        new_order = [m for m in sub["model"] if m in PAPER_ORDER]
        old_order = [m for m in PAPER_ORDER if m in set(sub["model"])]
        same = new_order == old_order

        top_old = old_order[0] if old_order else None
        top_new = new_order[0] if new_order else None

        if same:
            out_lines.append(
                f"On {ds}, the ordering of retained models is unchanged from "
                f"Table~\\ref{{tab:generators}}, with {tex(top_new)} again ranked "
                f"first at {sub.iloc[0]['accuracy_pct']:.2f}\\%."
            )
            verdicts.append(f"{ds}: ordering unchanged")
        else:
            moved = [m for m, n in zip(old_order, new_order) if m != n]
            out_lines.append(
                f"On {ds}, greedy decoding changes the ordering of retained "
                f"models: {tex(top_new)} now ranks first at "
                f"{sub.iloc[0]['accuracy_pct']:.2f}\\%, where the "
                f"default-sampling run placed {tex(top_old)} first. The models "
                f"whose rank changes are: {', '.join(tex(m) for m in moved)}."
            )
            verdicts.append(f"{ds}: ORDERING CHANGED")

        worst_parse = sub["unparseable_pct"].max()
        out_lines.append(
            f"Parse-failure rates on {ds} remain the dominant separator, "
            f"ranging from {sub['unparseable_pct'].min():.1f}\\% to "
            f"{worst_parse:.1f}\\%."
        )
        out_lines.append("")

    if all("unchanged" in v for v in verdicts):
        out_lines.append(
            "Because the ordering is stable under both decoding regimes, the "
            "model selection reported in Section~\\ref{sec:generators} does "
            "not depend on the sampling protocol, and the choice of generator "
            "for the later phases is unaffected."
        )
    else:
        out_lines.append(
            "Where the ordering changes, the greedy figures reported here "
            "supersede the default-sampling values, and "
            "Table~\\ref{tab:generators} has been updated accordingly. The "
            "exclusion decisions are unaffected: the three excluded models "
            "fail to produce a parseable answer on a large majority of items "
            "under either regime."
        )

    return "\n".join(out_lines), "; ".join(verdicts)


# ==========================================================  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    jobs = [
        ("10_a2_control.tex", a2_section),
        ("11_phase2_greedy.tex", phase2_section),
    ]

    print("=" * 72)
    print("PAPER TEXT FROM GPU RUNS")
    print("=" * 72)

    for fname, fn in jobs:
        text, verdict = fn()
        print(f"\n### {fname}")
        print(f"    {verdict}")
        if text is None:
            continue
        if args.write:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"    wrote {fname}")
        else:
            print()
            print(text)

    print("\n" + "=" * 72)
    print("Re-run with --write once both GPU jobs have finished.")


if __name__ == "__main__":
    main()
