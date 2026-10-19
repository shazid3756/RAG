"""
TEST B only, in its own process.

diagnose_finetuned.py established TEST A = COHERENT: the LoRA adapter and
the training run are fine. It then failed to run TEST B because the Test A
model was still resident in VRAM (my mistake -- two 7B models cannot share
12GB). This script runs TEST B alone in a fresh process.

TEST B loads qlora_output/qwen2.5-7b-medqa-merged directly and generates.

    COHERENT -> the merged weights are fine, so Ollama's safetensors ->
                GGUF -> Q4_K_M conversion is what produced the "@@@@@"
                garbage. Fix: re-create the Ollama model with q8_0.

    GARBAGE  -> merge_lora_adapter.py corrupted the weights during the
                merge. Fix: re-merge in float32.

Run from the project root, with nothing else on the GPU:
    python diagnose_merged.py
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MERGED_DIR = "qlora_output/qwen2.5-7b-medqa-merged"

PROMPT = (
    "A 45-year-old man presents with crushing substernal chest pain "
    "radiating to the left arm.\n\n"
    "A) Stable angina\nB) Myocardial infarction\nC) GERD\nD) Pneumonia\n\n"
    "Give a one-sentence explanation, then on a new line output exactly:\n"
    "FINAL_ANSWER: <letter>"
)

print(f"Free VRAM check: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB available")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading merged model in 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    quantization_config=bnb_config,
    device_map={"": 0},   # force everything onto the GPU; no CPU offload
)
model.eval()

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

print("\n" + "=" * 70)
print("TEST B — merged model")
print("=" * 70)
print(repr(response))

stripped = "".join(response.split())

if stripped:
    most_common = max(set(stripped), key=stripped.count)
    ratio = stripped.count(most_common) / len(stripped)
    verdict = "GARBAGE" if ratio > 0.5 else "COHERENT"
    print(f"\n  verdict: {verdict}  "
          f"(most common char {most_common!r} = {ratio:.0%} of output)")
else:
    verdict = "EMPTY"
    print("\n  verdict: EMPTY (model produced nothing)")

print("\n" + "=" * 70)
if verdict == "COHERENT":
    print("Merged weights are FINE. Ollama's Q4_K_M conversion is the culprit.")
    print("Next:  ollama rm qwen2.5-medqa-lora")
    print("       ollama create qwen2.5-medqa-lora --quantize q8_0 -f Modelfile")
    print("       python smoke_test_finetuned.py")
else:
    print("The merge corrupted the weights. merge_lora_adapter.py needs to")
    print("re-run in float32. Tell Claude this result and it will send the fix.")
print("=" * 70)
