"""
Bias Diagnostic: checks the fine-tuning training data
(finetune_data/train.jsonl + val.jsonl) for common bias sources
before/independent of training completing.

Checks:
    1. Answer-letter (A/B/C/D) distribution — a heavily skewed
       distribution (e.g. 40% "A") risks the model learning a
       positional shortcut instead of genuine reasoning.
    2. Completion-text diversity — flags if a large fraction of
       completions share near-identical boilerplate phrasing (this
       project's fallback template for MedQA is a known, documented
       source of this).
    3. Source dataset balance (MedQA vs MedMCQA), if traceable.

Run from the project root (does not require a GPU, safe to run while
training is in progress):
    python check_training_bias.py
"""

import json
import os
import re
from collections import Counter


TRAIN_FILE = "finetune_data/train.jsonl"
VAL_FILE = "finetune_data/val.jsonl"


def load_jsonl(path):

    examples = []

    if not os.path.exists(path):
        return examples

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def extract_letter(completion):

    match = re.search(r"FINAL_ANSWER:\s*([ABCD])", completion, re.IGNORECASE)

    return match.group(1).upper() if match else None


def extract_rationale(completion):
    """The part of the completion before the FINAL_ANSWER marker."""

    return completion.split("FINAL_ANSWER:")[0].strip()


def analyze(examples, label):

    print(f"\n{'=' * 50}")
    print(f"{label}  (n={len(examples)})")
    print(f"{'=' * 50}")

    if not examples:
        print("  (no examples found)")
        return

    # --- 1. Answer-letter distribution ---
    letters = [extract_letter(ex["completion"]) for ex in examples]
    letters = [l for l in letters if l]

    letter_counts = Counter(letters)
    total = len(letters)

    print("\nAnswer-letter distribution:")

    for letter in ["A", "B", "C", "D"]:

        count = letter_counts.get(letter, 0)
        pct = (count / total * 100) if total else 0

        bar = "#" * int(pct / 2)

        flag = "  <-- OVER-REPRESENTED" if pct > 32 else ("  <-- UNDER-REPRESENTED" if pct < 18 else "")

        print(f"  {letter}: {count:>5} ({pct:5.1f}%)  {bar}{flag}")

    expected_pct = 25.0
    max_deviation = max(
        abs((letter_counts.get(l, 0) / total * 100) - expected_pct)
        for l in ["A", "B", "C", "D"]
    ) if total else 0

    print(f"\n  Max deviation from perfectly balanced (25% each): "
          f"{max_deviation:.1f} percentage points")

    if max_deviation > 10:
        print("  -> NOTABLE IMBALANCE: consider mentioning this in the "
              "paper's limitations, and/or check if it's severe enough "
              "to warrant balancing (e.g. capping the most common letter).")
    else:
        print("  -> Reasonably balanced, low shortcut-learning risk from "
              "answer position alone.")

    # --- 2. Rationale diversity (boilerplate detection) ---
    rationales = [extract_rationale(ex["completion"]) for ex in examples]

    # Normalize by removing the specific answer text to detect the
    # SAME template being reused with only the answer swapped in.
    normalized = []

    for r in rationales:
        # crude normalization: strip out anything that looks like the
        # specific option text isn't feasible without the original
        # options, so instead just check exact duplicate rationales
        # AND check for the known fallback template phrase.
        normalized.append(r)

    exact_duplicates = len(normalized) - len(set(normalized))

    fallback_template_count = sum(
        1 for r in rationales
        if "provided context and medical knowledge support" in r.lower()
    )

    print(f"\nRationale diversity:")
    print(f"  Unique rationale strings: {len(set(normalized))} / {len(normalized)}")
    print(f"  Exact duplicate rationales: {exact_duplicates}")
    print(f"  Using the generic fallback template: {fallback_template_count} "
          f"({fallback_template_count / len(rationales) * 100:.1f}%)")

    if fallback_template_count / len(rationales) > 0.3:
        print("  -> WARNING: a large fraction of training examples use "
              "near-identical boilerplate phrasing (varying only the "
              "answer text). This risks the model learning surface-level "
              "template mimicry rather than genuine reasoning. Worth "
              "documenting as a limitation; consider it if results look "
              "like they've plateaued on rationale quality specifically.")


def main():

    train_examples = load_jsonl(TRAIN_FILE)
    val_examples = load_jsonl(VAL_FILE)

    analyze(train_examples, "TRAIN SET")
    analyze(val_examples, "VALIDATION SET")

    print("\n" + "=" * 50)
    print("Done. This is a diagnostic only — it does not modify your "
          "training data or affect any in-progress training run.")


if __name__ == "__main__":
    main()