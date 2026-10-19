"""
Why did every Llama arm come back 0.0% with a constant ~2.04 s latency?

All three arms -- including base llama3.2:3b, which scored 51.2% in the
earlier benchmark -- returned unparseable output on all 1,273 MedQA items.
A constant latency across thousands of calls means every call failed the
same way, not that the model answered badly.

eval_llama3b.py discarded the raw response when parsing failed, so the CSVs
cannot say why. This makes one call per model and prints everything:
the exact options sent, the raw response, and what Ollama has registered.

Run from the project root:
    python diagnose_llama_eval.py
"""

import json
import subprocess

import rag_pipeline
from rag_pipeline import generate_answer, OLLAMA_OPTIONS, OLLAMA_URL
from mcqa_evaluation import build_mcqa_prompt, extract_choice

import requests


MODELS = ["llama3.2:3b", "llama3.2-3b-medqa-lora", "qwen2.5:7b"]

GREEDY = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 42,
    "repeat_penalty": 1.0,
}


print("=" * 72)
print("1. WHAT OLLAMA HAS REGISTERED")
print("=" * 72)

try:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                         timeout=60)
    print(out.stdout.strip() or out.stderr.strip())
except Exception as e:
    print("could not run `ollama list`:", e)


print("\n" + "=" * 72)
print("2. BASELINE: default options, trivial prompt (no greedy overrides)")
print("=" * 72)

for m in MODELS:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": m, "prompt": "Reply with exactly: OK",
                  "stream": False, "options": {"num_predict": 10}},
            timeout=120,
        )
        body = r.json() if r.ok else r.text
        resp = body.get("response") if isinstance(body, dict) else body
        print(f"  {m:26s} status={r.status_code}  {resp!r}")
    except Exception as e:
        print(f"  {m:26s} EXCEPTION {e}")


print("\n" + "=" * 72)
print("3. WITH THE GREEDY OPTIONS eval_llama3b.py USES")
print("=" * 72)
print(f"  base OLLAMA_OPTIONS = {OLLAMA_OPTIONS}")
print(f"  greedy overrides    = {GREEDY}")

opts = dict(OLLAMA_OPTIONS)
opts.update(GREEDY)
opts["num_predict"] = 150

print(f"  merged options sent = {opts}\n")

for m in MODELS:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": m, "prompt": "Reply with exactly: OK",
                  "stream": False, "options": opts},
            timeout=180,
        )
        body = r.json() if r.ok else r.text
        resp = body.get("response") if isinstance(body, dict) else body
        done = body.get("done_reason") if isinstance(body, dict) else None
        print(f"  {m:26s} status={r.status_code}  done_reason={done}  {resp!r}")
    except Exception as e:
        print(f"  {m:26s} EXCEPTION {e}")


print("\n" + "=" * 72)
print("4. ONE ISOLATED OPTION AT A TIME (base + that option only)")
print("=" * 72)
print("  Identifies which specific option breaks generation.\n")

probe = "llama3.2:3b"

for name, extra in [
    ("none (num_predict only)", {}),
    ("temperature=0", {"temperature": 0}),
    ("top_p=1.0", {"top_p": 1.0}),
    ("top_k=0", {"top_k": 0}),
    ("seed=42", {"seed": 42}),
    ("repeat_penalty=1.0", {"repeat_penalty": 1.0}),
    ("num_gpu=20", {"num_gpu": 20}),
    ("num_ctx=2048", {"num_ctx": 2048}),
]:
    o = {"num_predict": 20}
    o.update(extra)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": probe, "prompt": "Reply with exactly: OK",
                  "stream": False, "options": o},
            timeout=180,
        )
        body = r.json() if r.ok else r.text
        resp = body.get("response") if isinstance(body, dict) else body
        print(f"  {name:26s} -> {resp!r}")
    except Exception as e:
        print(f"  {name:26s} -> EXCEPTION {e}")


print("\n" + "=" * 72)
print("5. THE REAL MCQA PROMPT, through generate_answer()")
print("=" * 72)

with open("evaluation/medqa_test.json", encoding="utf-8") as f:
    q = json.load(f)[0]

with open("evaluation/results/contexts_full_medqa_minilm_hybrid.json",
          encoding="utf-8") as f:
    ctx = json.load(f)[0]

prompt = build_mcqa_prompt(ctx, q["question"], q["options"])

print(f"  prompt length: {len(prompt)} chars\n")

rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)

for m in MODELS:
    resp = generate_answer(prompt, model=m, num_predict=150)
    print(f"  {m}")
    print(f"    extract_choice -> {extract_choice(resp)}   true={q['answer_idx']}")
    print(f"    raw: {resp[:400]!r}\n")

print("=" * 72)
print("Read section 4 first: if one option breaks it, that is the cause.")
print("If section 2 works but section 3 fails, the options are at fault.")
print("If section 2 also fails, the problem is the model or the server.")
print("=" * 72)
