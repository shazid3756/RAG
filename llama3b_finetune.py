"""
QLoRA fine-tune of Llama-3.2-3B-Instruct on the SAME training data and the
SAME hyperparameters used for Qwen2.5-7B.

WHY THIS MODEL
--------------
Finding 3 (fine-tuning helps recall-style MedMCQA items, does nothing for
MedQA's multi-step reasoning) currently rests on a single model, and the
obvious reviewer question is whether that is a property of QLoRA or a quirk
of Qwen2.5. A second family answers it.

Llama-3.2-3B was chosen over same-size alternatives (Mistral-7B ~32 h,
Llama-3.1-8B ~37 h) because at ~14 h it is the only one that fits in a
reasonable window, and because it also buys an efficiency claim: it was the
fastest model in the benchmark (1.55-1.77 s/question) and scored 51.2% MedQA
/ 45.8% MedMCQA at baseline. If fine-tuning lifts it near Qwen-7B on MedMCQA,
that is a genuine result for a paper with "Efficient" in its title.

The size change is a confound for "different family" -- report it as two
dimensions (family AND scale), not as a clean family-only comparison.

HOW IT WORKS
------------
This does NOT copy qlora_finetune.py. It imports it and overrides two module
globals before calling main(), so every hyperparameter -- LoRA rank, alpha,
target modules, batch size, LR, early stopping, checkpointing -- is provably
identical to the Qwen run. That is what makes the comparison meaningful.

Resume, checkpointing and early stopping all work exactly as before: re-run
this file after an interruption and it continues from the last checkpoint.

Run from the project root:
    python llama3b_finetune.py
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from transformers import AutoTokenizer

import qlora_finetune as q


# meta-llama repos are GATED on HuggingFace: without an accepted licence and
# an HF token the download fails. The mirrors below carry the same weights
# and are ungated. Tried in order; the first that loads a tokenizer wins.
CANDIDATE_REPOS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "NousResearch/Llama-3.2-3B-Instruct",
]

OUTPUT_DIR = "qlora_output/llama3.2-3b-medqa-lora"


def resolve_base_model():

    for repo in CANDIDATE_REPOS:
        try:
            print(f"Trying base model: {repo}")
            tok = AutoTokenizer.from_pretrained(repo)
            print(f"  OK -- using {repo}")
            return repo, tok
        except Exception as e:
            msg = str(e).split("\n")[0][:160]
            print(f"  unavailable: {msg}")

    print("\nNone of the candidate repos could be loaded.")
    print("If they are all gated, either:")
    print("  1. accept the licence at huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    print("     and run:  huggingface-cli login")
    print("  2. or add another ungated mirror to CANDIDATE_REPOS above.")
    sys.exit(1)


def main():

    base, tok = resolve_base_model()

    # The completion is trained with tokenizer.eos_token appended. Llama-3.2
    # instruct models end turns with <|eot_id|>, NOT <|end_of_text|>, and the
    # Ollama Modelfile must stop on whichever one is used here -- otherwise
    # generation runs to the token cap on every question. Printed so the
    # value is on the record before hours of training are spent.
    print("\n" + "=" * 66)
    print("TOKENIZER FACTS (needed later for the Ollama Modelfile)")
    print("=" * 66)
    print(f"  eos_token   : {tok.eos_token!r}  (id {tok.eos_token_id})")
    print(f"  pad_token   : {tok.pad_token!r}")
    print(f"  vocab size  : {len(tok)}")
    print("  -> llama3b_merge.py writes the matching Modelfile automatically.")
    print("=" * 66 + "\n")

    # Override ONLY the model identity. Everything else -- LORA_R=16,
    # LORA_ALPHA=32, all attn+MLP target modules, batch 1 x accum 16,
    # lr 2e-4, 3 epochs cap, eval/save every 25 steps, early stopping
    # patience 3, load_best_model_at_end -- stays exactly as the Qwen run.
    q.BASE_MODEL = base
    q.OUTPUT_DIR = OUTPUT_DIR

    print(f"BASE_MODEL  = {q.BASE_MODEL}")
    print(f"OUTPUT_DIR  = {q.OUTPUT_DIR}")
    print(f"TRAIN_FILE  = {q.TRAIN_FILE}  (identical data to the Qwen run)")
    print(f"LORA_R={q.LORA_R}  ALPHA={q.LORA_ALPHA}  "
          f"batch={q.PER_DEVICE_BATCH_SIZE}x{q.GRADIENT_ACCUMULATION_STEPS}  "
          f"lr={q.LEARNING_RATE}")
    print()

    q.main()


if __name__ == "__main__":
    main()
