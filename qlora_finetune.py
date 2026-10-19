"""
QLoRA Fine-Tuning — Stage 2: Training.

Fine-tunes Qwen2.5-7B-Instruct (the exact base model behind the
qwen2.5:7b Ollama tag used throughout this project) using QLoRA
(4-bit quantized base model + trainable LoRA adapters) on the
RAG-formatted training data from Stage 1 (download_finetune_data.py).

VRAM strategy for a 12GB GPU (RTX 3060):
    - Base model loaded in 4-bit (bitsandbytes NF4 + double quant) —
      uses ~4.5-5GB VRAM for the frozen base weights.
    - Only small LoRA adapter matrices are trainable (~1-2% of total
      parameters) — trainable-parameter memory and optimizer state
      are tiny by comparison.
    - Gradient checkpointing trades some compute time for a large
      activation-memory reduction — necessary to fit a 7B model's
      training activations in 12GB.
    - batch_size=1 with gradient_accumulation_steps=16 gives an
      effective batch size of 16 without needing the memory for a
      larger literal batch.
    - MAX_SEQ_LENGTH=2048 matches the num_ctx already used everywhere
      else in this pipeline (rag_pipeline.py's OLLAMA_OPTIONS), so
      training and inference see the same effective context window.
    - Loss is computed ONLY on the completion tokens (prompt tokens
      are masked to -100) — the model should learn to PRODUCE the
      right answer in the right format, not to memorize/reproduce
      retrieved passages.
    - When a prompt+completion pair exceeds MAX_SEQ_LENGTH, tokens are
      dropped from the START of the PROMPT only (never from the
      completion) — this preserves the critical instructions and
      FINAL_ANSWER format cue that sit at the END of every prompt
      (the same reasoning as the Phase 1 fix that moved instructions
      to the end of the RAG prompt).

If you hit an out-of-memory error, try in this order:
    1. Lower MAX_SEQ_LENGTH (e.g. to 1536).
    2. Remove "gate_proj", "up_proj", "down_proj" from TARGET_MODULES,
       keeping only the attention projections.
    3. Close other GPU-using applications (browser hardware
       acceleration, other Python processes).

Requires:
    pip install transformers peft bitsandbytes accelerate datasets --break-system-packages

Run from the project root (this will take a while — expect at least
1-3 hours for 2 epochs over ~2700 examples on this hardware):
    python qlora_finetune.py
"""

import json
import os
import random

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==========================
# Configuration
# ==========================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_FILE = "finetune_data/train.jsonl"
VAL_FILE = "finetune_data/val.jsonl"

OUTPUT_DIR = "qlora_output/qwen2.5-7b-medqa-lora-v2"

MAX_SEQ_LENGTH = 2048

# TRAIN_SAMPLE_LIMIT=None -> use the full 2700-example training set.
# The FIRST training attempt (now abandoned in the "-v2" rename) used
# the full set and started overfitting right around epoch 1.0 (eval
# loss bottomed out around 0.60, then rose to 0.62+). Rather than
# guess a shorter epoch count, this version uses the proper fix:
# load_best_model_at_end + EarlyStoppingCallback below, so training
# automatically stops once eval loss stops improving, and the model
# that gets SAVED is genuinely the best one seen — not just whatever
# was most recent when training happened to stop or get interrupted.
TRAIN_SAMPLE_LIMIT = None
SAMPLE_SEED = 42

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters (tuned for 12GB VRAM, see module docstring)
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

# Set generously — EarlyStoppingCallback below will halt training
# automatically once eval_loss stops improving, well before this cap
# is normally reached. This just bounds the worst case.
NUM_EPOCHS = 3

LEARNING_RATE = 2e-4
LOGGING_STEPS = 10

# Checkpoint frequency: kept LOW (25 steps, not 100) because LoRA
# adapter checkpoints are small/cheap to write, and this hardware has
# experienced real power interruptions during training — losing at
# most 25 steps of progress is far better than losing 100.
EVAL_STEPS = 25
SAVE_STEPS = 25

# How many EVALUATIONS in a row with no improvement before stopping.
# At EVAL_STEPS=25, patience=3 means training stops after ~75 steps
# of no improvement — enough buffer to ride out noisy eval_loss
# fluctuations without stopping too early, but short enough to avoid
# wasting hours training deeper into overfitting once the trend is
# clear (see the first training attempt's eval_loss history: it rose
# for 3+ consecutive evals after the true minimum and never recovered).
EARLY_STOPPING_PATIENCE = 3

# THE key fix for what went wrong last time: with this on, the
# Trainer tracks the BEST eval_loss seen so far, protects that
# checkpoint from deletion regardless of SAVE_TOTAL_LIMIT, and loads
# it back as the final model at the end of training — so the model
# you actually get is the best one observed, not just the most recent.
LOAD_BEST_MODEL_AT_END = True

# Small increase from before (2 -> 3) as extra safety margin — with
# LOAD_BEST_MODEL_AT_END on, the Trainer will not delete the best
# checkpoint even if it's no longer among the most recent
# SAVE_TOTAL_LIMIT checkpoints, but a slightly larger limit gives more
# headroom in case of any edge case around a resume.
SAVE_TOTAL_LIMIT = 3


# ==========================
# Data loading
# ==========================

def load_jsonl(path):

    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def build_example(tokenizer, prompt, completion, max_seq_length):
    """
    Tokenizes a (prompt, completion) pair into model-ready
    input_ids/attention_mask/labels, with:
      - loss masked (-100) on the prompt portion (labels only cover
        the completion, so the model is trained to GENERATE the
        answer, not to reproduce the retrieved context)
      - truncation, if needed, removing tokens from the START of the
        PROMPT only — the completion and the end of the prompt
        (containing the critical output-format instructions) are
        always preserved intact.
    """

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    completion_ids = tokenizer.encode(
        completion + tokenizer.eos_token, add_special_tokens=False
    )

    total_len = len(prompt_ids) + len(completion_ids)

    if total_len > max_seq_length:

        overflow = total_len - max_seq_length

        # Drop the earliest `overflow` tokens of the PROMPT only.
        # If overflow exceeds the whole prompt (pathological/very
        # long completion case), keep at least 1 prompt token so the
        # example isn't purely completion with zero context marker.
        prompt_ids = prompt_ids[min(overflow, len(prompt_ids) - 1):]

    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + list(completion_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class MCQADataset(Dataset):

    def __init__(self, examples, tokenizer, max_seq_length):

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        item = self.examples[idx]

        return build_example(
            self.tokenizer, item["prompt"], item["completion"], self.max_seq_length
        )


def make_collate_fn(pad_token_id):
    """Pads a batch of variable-length examples. With
    PER_DEVICE_BATCH_SIZE=1 this rarely needs to pad more than one
    sequence, but is written to handle larger batch sizes correctly
    too, in case VRAM allows increasing it later."""

    def collate_fn(batch):

        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids, attention_mask, labels = [], [], []

        for ex in batch:

            pad_len = max_len - len(ex["input_ids"])

            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


# ==========================
# Main
# ==========================

def main():

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected. QLoRA training on CPU "
              "will be extremely slow or may fail. Check your PyTorch/"
              "CUDA installation before proceeding.")

    print("Loading tokenizer:", BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading train/val data...")

    train_examples = load_jsonl(TRAIN_FILE)
    val_examples = load_jsonl(VAL_FILE)

    if TRAIN_SAMPLE_LIMIT is not None and TRAIN_SAMPLE_LIMIT < len(train_examples):

        rng = random.Random(SAMPLE_SEED)
        train_examples = rng.sample(train_examples, TRAIN_SAMPLE_LIMIT)

        print(f"  Subsampled training set to {TRAIN_SAMPLE_LIMIT} examples "
              f"(from {len(load_jsonl(TRAIN_FILE))}) to keep training time "
              f"tractable on this hardware.")

    print(f"  {len(train_examples)} training examples")
    print(f"  {len(val_examples)} validation examples")

    train_dataset = MCQADataset(train_examples, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = MCQADataset(val_examples, tokenizer, MAX_SEQ_LENGTH)

    print("\nLoading base model in 4-bit:", BASE_MODEL)
    print("(this downloads the full-precision weights once, then "
          "quantizes on load — first run will take a while)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    print("\nStarting training...")
    print(f"(Early stopping active: will halt after {EARLY_STOPPING_PATIENCE} "
          f"evaluations with no eval_loss improvement, then automatically "
          f"reload the BEST checkpoint seen — not just the last one.)")

    # Resume from the latest checkpoint if one exists in OUTPUT_DIR —
    # essential for long training runs on hardware that can lose
    # power/crash mid-run: with save_steps=SAVE_STEPS, at most
    # SAVE_STEPS worth of progress is ever lost, not the whole run.
    last_checkpoint = None

    if os.path.isdir(OUTPUT_DIR):

        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

        if last_checkpoint:
            print(f"Found existing checkpoint: {last_checkpoint}")
            print("Resuming training from this checkpoint (not starting over).")
        else:
            print(f"OUTPUT_DIR exists but no checkpoint found inside it — "
                  f"starting a fresh run.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("\nSaving LoRA adapter to:", OUTPUT_DIR)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone. Next step: merge the adapter into the base model "
          "and convert to GGUF for use with Ollama (Stage 3).")


if __name__ == "__main__":
    main()