"""
Build Figures 3-6 for the Medical RAG paper from the per-question result CSVs.

Design follows the project's data-viz rules: validated 3-slot categorical
palette (blue / orange / aqua -- passes all-pairs CVD and normal-vision floors
on the light surface), direct value labels on every mark (the aqua slot sits
below 3:1 contrast, so labels are the required relief), recessive grid, no
dual axes, legend present whenever two or more series share a panel.

Outputs PDF (vector, for the journal) and 300-dpi PNG into figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "/mnt/user-data/uploads/Medical_RAG_Project/evaluation/results"
OUT = "/home/claude/figures"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a95"
GRID = "#e3e3e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

DATASETS = [("medqa", "MedQA  (n = 1,273)"), ("medmcqa", "MedMCQA  (n = 4,183)")]


def acc(ds, arm):
    return pd.read_csv(f"{R}/full_{ds}_{arm}.csv")["correct"].astype(bool).mean() * 100


def lat(ds, arm):
    d = pd.read_csv(f"{R}/full_{ds}_{arm}.csv")
    return d["latency"].mean() if "latency" in d.columns else None


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ============================================================ FIGURE 3
# Ablation ladder: does each pipeline component help? Two models, two datasets.
def figure3():

    stages = ["A", "B", "C", "D"]
    labels = ["LLM only\n(no RAG)", "+ naive RAG\n(bge, dense)",
              "+ optimised\nretrieval", "+ fine-tuned\nLLM"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

    for ax, (ds, title) in zip(axes, DATASETS):

        qwen = [acc(ds, s) for s in stages]
        llama = [acc(ds, "L_A" if s == "A" else "L_C" if s == "C" else
                     "L_D" if s == "D" else None) if s != "B" else np.nan
                 for s in stages]

        x = np.arange(4)
        w = 0.36

        b1 = ax.bar(x - w / 2 - 0.01, qwen, w, color=BLUE,
                    label="Qwen2.5-7B", zorder=3)
        b2 = ax.bar(x + w / 2 + 0.01, llama, w, color=ORANGE,
                    label="Llama-3.2-3B", zorder=3)

        for bars, vals in ((b1, qwen), (b2, llama)):
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.9,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=8, color=INK)

        ax.text(1 + w / 2 + 0.01, 1.5, "not run\nfor 3B", ha="center",
                va="bottom", fontsize=7.2, color=INK3, style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 72)
        ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
        style(ax)

        if ds == "medqa":
            ax.set_ylabel("Accuracy (%)")

    axes[0].legend(frameon=False, loc="upper right", fontsize=8.5,
                   handlelength=1.2, borderaxespad=0.2)

    fig.suptitle("Figure 3. Adding retrieval lowers accuracy on both models "
                 "and both datasets; only fine-tuning recovers it",
                 fontsize=11, fontweight="bold", color=INK, x=0.008, ha="left",
                 y=1.045)

    fig.text(0.008, -0.09,
             "Adding retrieval is significant in every model x dataset cell "
             "(McNemar's exact test): Qwen A->B p=0.020 / p<1e-6; Llama A->C "
             "p=0.0010 / p=2.5e-8.\nThe naive-RAG stage was run only for "
             "Qwen, so the Llama retrieval test compares A against C. Stage "
             "C->D is significant only on MedMCQA (Qwen +9.92 pp, p<1e-6;\n"
             "Llama +6.60 pp, p=9.0e-15) and null on MedQA for both models.",
             fontsize=7.6, color=INK2, ha="left", va="top")

    save(fig, "fig3_ablation_ladder")


# ============================================================ FIGURE 4
# The selective-retrieval gap: a large oracle ceiling neither gate reaches.
def figure4():

    g = pd.read_csv(f"{R}/full_gating.csv")
    sig = pd.read_csv(f"{R}/full_gate_significance.csv")

    rows = []

    for _, r in g.iterrows():
        s = sig[(sig.dataset == r.dataset) &
                (sig.context_arm == r.context_arm)].iloc[0]
        arm = ("base LLM" if r.context_arm == "C" else "fine-tuned LLM")
        rows.append({
            "label": f"{'MedQA' if r.dataset == 'medqa' else 'MedMCQA'}\n"
                     f"context from {arm}",
            "cos_delta": s.cos_vs_fixed_pp,
            "judge_delta": s.judge_vs_fixed_pp,
            "headroom": r.headroom_pp,
        })

    d = pd.DataFrame(rows)

    # Plot CHANGE against the better fixed policy, not raw accuracy. Raw bars
    # would be dishonest here: the cosine gate is measured on the held-out
    # half while the fixed policy's full-set accuracy comes from all items, so
    # side-by-side bars make the gate look like a winner when it is not. Each
    # delta below is computed against its own correct basis, and every delta
    # shares the same zero, so the comparison is exact.
    x = np.arange(len(d))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 4.2))

    b1 = ax.bar(x - w / 2 - 0.01, d.cos_delta, w, color=BLUE, zorder=3,
                label="Cosine-threshold gate (held-out half)")
    b2 = ax.bar(x + w / 2 + 0.01, d.judge_delta, w, color=ORANGE, zorder=3,
                label="LLM-judge gate (all items)")

    for bars, vals in ((b1, d.cos_delta), (b2, d.judge_delta)):
        for b, v in zip(bars, vals):
            off = 0.28 if v >= 0 else -0.28
            ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=INK)

    for i, o in enumerate(d.headroom):
        ax.plot([i - 0.40, i + 0.40], [o, o], color=INK, linewidth=1.8,
                zorder=4)
        ax.text(i, o + 0.35, f"oracle ceiling  +{o:.2f}", ha="center",
                va="bottom", fontsize=8, color=INK, fontweight="bold")

    ax.axhline(0, color=INK2, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(d.label, fontsize=8.5)
    ax.set_xlim(-0.55, len(d) - 0.45)
    ax.set_ylim(-4.5, 14.5)
    ax.set_ylabel("Change in accuracy vs best fixed policy (pp)")
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5,
              handlelength=1.2, borderaxespad=0.3)

    fig.suptitle("Figure 4. Perfect per-question gating is worth 8.7-11.5 pp; "
                 "both practical gates capture none of it",
                 fontsize=11, fontweight="bold", color=INK, x=0.008, ha="left",
                 y=1.02)

    fig.text(0.008, -0.07,
             "Each bar is that gate's accuracy minus the better of the two "
             "fixed policies (always inject context / never inject), computed "
             "on the same items as the gate.\nThe cosine gate's threshold was "
             "tuned on one half and scored on the other; pooled +72/-93, "
             "p=0.119. The LLM-judge gate is pooled significantly WORSE than "
             "doing\nnothing (+496/-650, p=0.00001), despite judging correctly "
             "on 53.0% of the items where the two arms disagree (p=0.0025) -- "
             "real signal, far too weak to act on.",
             fontsize=7.6, color=INK2, ha="left", va="top")

    save(fig, "fig4_selective_retrieval_gap")


# ============================================================ FIGURE 5
# Efficiency: accuracy against latency. The claim behind "Efficient".
def figure5():

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

    qwen_lat = {"medqa": {"C": 2.72, "D": 2.77},
                "medmcqa": {"C": 2.71, "D": 1.81}}

    for ax, (ds, title) in zip(axes, DATASETS):

        pts_3b = [("3B no retrieval", acc(ds, "L_A"), lat(ds, "L_A")),
                  ("3B + retrieval", acc(ds, "L_C"), lat(ds, "L_C")),
                  ("3B fine-tuned\n+ retrieval", acc(ds, "L_D"), lat(ds, "L_D"))]

        pts_7b = [("7B + retrieval", acc(ds, "C"), qwen_lat[ds]["C"]),
                  ("7B fine-tuned\n+ retrieval", acc(ds, "D"), qwen_lat[ds]["D"])]

        # Offsets are hand-placed per point: automatic placement collided on
        # the two Qwen points, which sit within 0.05 s of each other on MedQA.
        off = {
            "medqa": {"3B no retrieval": (0, 15, "center"),
                      "3B + retrieval": (12, -6, "left"),
                      "3B fine-tuned\n+ retrieval": (-12, -6, "right"),
                      "7B + retrieval": (-12, -8, "right"),
                      "7B fine-tuned\n+ retrieval": (0, 14, "center")},
            "medmcqa": {"3B no retrieval": (-12, -6, "right"),
                        "3B + retrieval": (0, -30, "center"),
                        "3B fine-tuned\n+ retrieval": (0, 14, "center"),
                        "7B + retrieval": (0, -30, "center"),
                        "7B fine-tuned\n+ retrieval": (0, 14, "center")},
        }[ds]

        for group, colour, marker in ((pts_3b, ORANGE, "o"),
                                      (pts_7b, BLUE, "s")):
            for lbl, a, l in group:
                ax.scatter(l, a, s=66, color=colour, marker=marker, zorder=4,
                           edgecolor="white", linewidth=1.4)
                dx, dy, ha = off[lbl]
                ax.annotate(f"{lbl}\n{a:.1f}%", (l, a),
                            textcoords="offset points", xytext=(dx, dy),
                            ha=ha, fontsize=7.7, color=INK2,
                            linespacing=1.35)

        ax.set_xlim(0.55, 3.35)
        ax.set_ylim(42, 67)
        ax.set_xlabel("Latency (s per question)  -  lower is better")
        ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.7)

        if ds == "medqa":
            ax.set_ylabel("Accuracy (%)  -  higher is better")

    h = [plt.Line2D([], [], marker="s", color=BLUE, linestyle="", markersize=7,
                    label="Qwen2.5-7B"),
         plt.Line2D([], [], marker="o", color=ORANGE, linestyle="", markersize=7,
                    label="Llama-3.2-3B")]
    axes[0].legend(handles=h, frameon=False, loc="lower right", fontsize=8.5,
                   handlelength=1.0, borderaxespad=0.3)

    fig.suptitle("Figure 5. A fine-tuned 3B beats a 7B with full retrieval on "
                 "MedMCQA, at roughly a third of the latency",
                 fontsize=11, fontweight="bold", color=INK, x=0.008, ha="left",
                 y=1.03)

    fig.text(0.008, -0.09,
             "MedMCQA: fine-tuned 3B 53.4% vs base 7B + retrieval 49.2% "
             "(+4.23 pp, p<0.0001). MedQA: the two are indistinguishable "
             "(-2.75 pp, p=0.097).\nUp and to the left is better. Llama "
             "latencies come from the full-test-set run; Qwen latencies from "
             "the greedy n=500 run, which is the only one that recorded them.",
             fontsize=7.6, color=INK2, ha="left", va="top")

    save(fig, "fig5_accuracy_vs_latency")


# ============================================================ FIGURE 6
# Generator benchmark: general-purpose models beat medical fine-tunes.
def figure6():

    b = pd.read_csv(f"{R}/llm_benchmark.csv")
    p = b.pivot_table(index="llm", columns="dataset", values="accuracy")
    fb = b.pivot_table(index="llm", columns="dataset", values="fallback_rate")

    p = p.sort_values("medqa")

    medical = {"medllama2:7b", "koesn/llama3-openbiollm-8b",
               "cniongolo/biomistral", "meditron:7b"}

    names = {"llama3.1:8b": "Llama-3.1-8B", "gemma2:9b-instruct-q4_0": "Gemma2-9B",
             "qwen2.5:7b": "Qwen2.5-7B", "llama3.2:3b": "Llama-3.2-3B",
             "mistral:7b-instruct": "Mistral-7B",
             "koesn/llama3-openbiollm-8b": "OpenBioLLM-8B",
             "medllama2:7b": "MedLlama2-7B", "phi3.5": "Phi-3.5-mini",
             "cniongolo/biomistral": "BioMistral-7B", "meditron:7b": "Meditron-7B"}

    y = np.arange(len(p))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 4.6))

    b1 = ax.barh(y - h / 2 - 0.01, p["medqa"] * 100, h, color=BLUE,
                 label="MedQA", zorder=3)
    b2 = ax.barh(y + h / 2 + 0.01, p["medmcqa"] * 100, h, color=ORANGE,
                 label="MedMCQA", zorder=3)

    for bars, col in ((b1, "medqa"), (b2, "medmcqa")):
        for bar, m in zip(bars, p.index):
            v = p.loc[m, col] * 100
            ax.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                    va="center", fontsize=7.8, color=INK)

    labels = []
    for m in p.index:
        f = max(fb.loc[m, "medqa"], fb.loc[m, "medmcqa"])
        tag = "  ▲" if f > 0.15 else ""
        labels.append(names.get(m, m) + tag)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)

    for tick, m in zip(ax.get_yticklabels(), p.index):
        tick.set_color(ORANGE if m in medical else INK)

    ax.axvline(25, color=INK3, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(25.4, -0.85, "random baseline (4 options)", fontsize=7.5,
            color=INK3)

    ax.set_xlim(0, 72)
    ax.set_xlabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, handlelength=1.2)

    fig.suptitle("Figure 6. General-purpose models take the top five places; "
                 "medical fine-tunes cluster at the bottom",
                 fontsize=11, fontweight="bold", color=INK, x=0.008, ha="left",
                 y=1.00)

    fig.text(0.008, -0.055,
             "Model names in orange are medical or biomedical fine-tunes. "
             "▲ marks models that failed to produce a parseable answer on "
             "more than 15% of items;\nall four so marked are medical or "
             "continued-pretraining models. n = 500 per dataset, retrieval "
             "fixed at MiniLM + hybrid.",
             fontsize=7.6, color=INK2, ha="left", va="top")

    save(fig, "fig6_generator_benchmark")


if __name__ == "__main__":
    print("Building figures...")
    figure3()
    figure4()
    figure5()
    figure6()
    print(f"\nDone -> {OUT}")
