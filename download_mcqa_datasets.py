"""
Download MedQA (USMLE, 4-option) and MedMCQA, and convert both into a
single unified JSON format so mcqa_evaluation.py can evaluate the RAG
pipeline on either/both with the same code.

Why MedQA/MedMCQA instead of relying only on PubMedQA:
    PubMedQA classification requires the model to have access to the
    ONE specific source abstract each question was derived from. With
    a 100K-document sample out of ~23.9M PubMed abstracts, the odds
    any given question's source paper is even IN the corpus are only
    ~0.4% — so PubMedQA accuracy there is dominated by corpus
    coverage, not retrieval or embedding quality (confirmed by the
    oracle-context test: 63% with the true passage vs ~18% via RAG,
    the same ~18% across ALL 5 embedding models).

    MedQA and MedMCQA are different: they're general medical-knowledge
    board-exam questions, NOT tied to one specific source document.
    Any relevant PubMed abstract about the same topic can help — so
    they're a fairer benchmark for an OPEN-CORPUS retrieval system
    like this one.

IMPORTANT: MedMCQA's official "test" split does NOT include public
ground-truth labels (they're withheld for a leaderboard submission
process) — so this script uses the "validation" split instead, which
does have real labels, for local evaluation.

Unified output format (list of dicts), one file per dataset:
    {
        "question": str,
        "options": {"A": str, "B": str, "C": str, "D": str},
        "answer_idx": "A" | "B" | "C" | "D",
        "source": "medqa" | "medmcqa"
    }

Requires: pip install datasets --break-system-packages

Run from the project root:
    python download_mcqa_datasets.py
"""

import json
import os

from datasets import load_dataset


OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MEDQA_OUTPUT = os.path.join(OUTPUT_DIR, "medqa_test.json")
MEDMCQA_OUTPUT = os.path.join(OUTPUT_DIR, "medmcqa_val.json")

COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def build_medqa():

    print("Downloading MedQA (GBaker/MedQA-USMLE-4-options, test split)...")

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    records = []

    for row in ds:

        options = row["options"]  # already {"A":..., "B":..., "C":..., "D":...}

        answer_idx = row["answer_idx"]

        if answer_idx not in options:
            # Guard against any malformed rows rather than crashing
            # a multi-minute download over one bad record.
            continue

        records.append({
            "question": row["question"],
            "options": options,
            "answer_idx": answer_idx,
            "source": "medqa",
        })

    with open(MEDQA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Saved {len(records)} MedQA questions to {MEDQA_OUTPUT}")


def build_medmcqa():

    print("Downloading MedMCQA (openlifescienceai/medmcqa, validation split)...")

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")

    records = []

    for row in ds:

        cop = row["cop"]

        answer_idx = COP_TO_LETTER.get(cop)

        if answer_idx is None:
            continue

        options = {
            "A": row["opa"],
            "B": row["opb"],
            "C": row["opc"],
            "D": row["opd"],
        }

        records.append({
            "question": row["question"],
            "options": options,
            "answer_idx": answer_idx,
            "source": "medmcqa",
        })

    with open(MEDMCQA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Saved {len(records)} MedMCQA questions to {MEDMCQA_OUTPUT}")


if __name__ == "__main__":

    build_medqa()
    build_medmcqa()

    print("\nDone. Files ready for mcqa_evaluation.py:")
    print(" -", MEDQA_OUTPUT)
    print(" -", MEDMCQA_OUTPUT)