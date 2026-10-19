"""
Diagnostic: the Ollama model qwen2.5-medqa-lora emits repeated garbage
("@@@@@@..."). This isolates WHICH stage broke it.

Three possible culprits:
    A. the LoRA adapter itself (training went wrong)
    B. merge_lora_adapter.py (merge corrupted the weights)
    C. Ollama's safetensors -> GGUF -> Q4_K_M conversion

Test A: base model in 4-bit + adapter on top  == exactly the training-time
        configuration. If this is coherent, training and the adapter are fine.
Test B: the merged folder loaded directly in 4-bit. If A is coherent but B is
        garbage, merge_lora_adapter.py is the culprit.

If BOTH A and B are coherent, the weights are fine and Ollama's conversion is
the culprit -> re-create with --quantize q8_0 (or f16) to test.

Run from the project root:
    python diagnose_finetuned.py
"""

import gc
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "qlora_output/qwen2.5-7b-medqa-lora-v2"
MERGED_DIR = "qlora_output/qwen2.5-7b-medqa-merged"

PROMPT = (
    "A 45-year-old man presents with crushing substernal chest pain "
    "radiating to the left arm.\n\n"
    "A) Stable angina\nB) Myocardial infarction\nC) GERD\nD) Pneumonia\n\n"
    "Give a one-sentence explanation, then on a new line output exactly:\n"
    "FINAL_ANSWER: <letter>"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def generate(model, tokenizer, label):

    messages = [{"role": "user", "content": PROMPT}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print("=" * 70)
    print(repr(response))

    # Crude garbage detector: one character dominating the output.
    stripped = "".join(response.split())
    if stripped:
        most_common = max(set(stripped), key=stripped.count)
        ratio = stripped.count(most_common) / len(stripped)
        verdict = "GARBAGE" if ratio > 0.5 else "COHERENT"
        print(f"\n  verdict: {verdict}  "
              f"(most common char {most_common!r} = {ratio:.0%} of output)")

    return response


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- TEST A
print("\n### TEST A: base (4-bit) + LoRA adapter  [training configuration]")
print("Loading base model in 4-bit...")

tok_a = AutoTokenizer.from_pretrained(BASE_MODEL)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model_a = PeftModel.from_pretrained(base, ADAPTER_DIR)
model_a.eval()

resp_a = generate(model_a, tok_a, "TEST A — base 4-bit + adapter")

free(model_a, base)

# ---------------------------------------------------------------- TEST B
print("\n\n### TEST B: merged model loaded directly (4-bit)")
print("Loading merged model...")

tok_b = AutoTokenizer.from_pretrained(MERGED_DIR)

model_b = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)
model_b.eval()

resp_b = generate(model_b, tok_b, "TEST B — merged model")

free(model_b)

# ---------------------------------------------------------------- verdict
print("\n\n" + "=" * 70)
print("HOW TO READ THIS")
print("=" * 70)
print("A coherent, B coherent  -> weights are fine; Ollama's Q4_K_M")
print("                           conversion is the culprit. Re-create with")
print("                           --quantize q8_0 and re-run the smoke test.")
print("A coherent, B garbage   -> merge_lora_adapter.py corrupted the")
print("                           weights. Re-merge (in float32, or on GPU).")
print("A garbage               -> the adapter/training itself is broken.")
print("                           Fall back to the attempt-1 checkpoint or")
print("                           retrain. This would be the bad outcome.")
print("=" * 70)
