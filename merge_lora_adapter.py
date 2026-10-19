"""
QLoRA Pipeline — Stage 3a: Merge LoRA Adapter into Base Model.

Loads the base Qwen2.5-7B-Instruct model (full precision this time,
NOT 4-bit — merging requires full-precision weights to combine
cleanly with the LoRA adapter), applies the trained LoRA adapter on
top, merges the two into a single set of weights, and saves the
result as a standalone HuggingFace model directory.

This merged model is what gets converted to GGUF in Stage 3b — Ollama
runs GGUF files, it cannot load a base model + separate LoRA adapter
directly.

Note on memory: this loads the FULL-PRECISION (not 4-bit) base model
to merge cleanly, which needs more RAM than the 4-bit training stage
did. If you hit an out-of-memory error, close other applications
first (see project history — OneDrive, browser, etc. have caused RAM
pressure before).

Requires: pip install transformers peft accelerate --break-system-packages

Run from the project root (after qlora_finetune.py has completed):
    python merge_lora_adapter.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================
# Configuration
# ==========================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Must match OUTPUT_DIR in qlora_finetune.py — this is where the
# trained LoRA adapter (and its own copy of the tokenizer) was saved.
ADAPTER_DIR = "qlora_output/qwen2.5-7b-medqa-lora-v2"

MERGED_OUTPUT_DIR = "qlora_output/qwen2.5-7b-medqa-merged"


def main():

    print("Loading base model (full precision):", BASE_MODEL)
    print("(This is a larger download/load than the 4-bit training "
          "stage — merging needs full-precision weights.)")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merging doesn't need GPU; keeps VRAM free
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading LoRA adapter from:", ADAPTER_DIR)

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            f"Make sure qlora_finetune.py has finished and saved to "
            f"this path (check OUTPUT_DIR in that script matches)."
        )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging adapter into base weights...")

    merged_model = model.merge_and_unload()

    print("Saving merged model to:", MERGED_OUTPUT_DIR)

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

    merged_model.save_pretrained(
        MERGED_OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print("\nDone. Merged model saved to:", MERGED_OUTPUT_DIR)
    print("Next step (Stage 3b): convert this to GGUF format using "
          "llama.cpp, then register it with Ollama.")


if __name__ == "__main__":
    main()