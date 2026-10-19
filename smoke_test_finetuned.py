"""
Smoke test for the fine-tuned model BEFORE committing to the full
1000-question llm_benchmark.py run.

Runs 5 real MedQA questions through the EXACT same path the benchmark
uses (same retrieval, same prompt builder, same extractor) and reports
whether the model produces parseable FINAL_ANSWER output.

If fallback_rate here is high, the Modelfile TEMPLATE is wrong and the
full run would waste hours producing garbage. Fix the template first.

Run from the project root:
    python smoke_test_finetuned.py
"""

import json
import random
import time

import retriever
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice

# Mirror llm_benchmark.py exactly.
MODEL = "qwen2.5-medqa-lora"
BASELINE = "qwen2.5:7b"
FIXED_EMBEDDING_MODEL = "minilm"
RETRIEVAL_MODE = "hybrid"
TOP_K = 8
N = 5
RANDOM_SEED = 42

with open("evaluation/medqa_test.json", "r", encoding="utf-8") as f:
    all_questions = json.load(f)

# Same seed-42 sample the benchmark draws from, first N of it.
rng = random.Random(RANDOM_SEED)
questions = rng.sample(all_questions, 500)[:N]

print("=" * 70)
print(f"Smoke test: {MODEL}  vs  {BASELINE}   ({N} MedQA questions)")
print("=" * 70)

for model in (MODEL, BASELINE):

    print(f"\n--- {model} ---")

    correct = 0
    unparseable = 0
    latencies = []

    for i, item in enumerate(questions):

        question = item["question"]
        options = item["options"]
        truth = item["answer_idx"]

        start = time.time()

        results = retriever.retrieve_context(
            question, k=TOP_K, mode=RETRIEVAL_MODE, model_key=FIXED_EMBEDDING_MODEL
        )
        context_text = "\n\n".join(r["chunk"] for r in results)

        prompt = build_mcqa_prompt(context_text, question, options)
        response = generate_answer(prompt, model=model, num_predict=150)

        latencies.append(time.time() - start)

        prediction = extract_choice(response)

        if prediction is None:
            unparseable += 1
            status = "UNPARSEABLE"
        else:
            hit = prediction == truth
            correct += hit
            status = f"pred={prediction} true={truth} {'OK' if hit else 'wrong'}"

        print(f"  Q{i+1}: {status}   ({latencies[-1]:.1f}s)")

        if prediction is None:
            print(f"       raw response: {response[:300]!r}")

    print(f"  -> {correct}/{N} correct, {unparseable}/{N} unparseable, "
          f"avg {sum(latencies)/len(latencies):.1f}s")

print("\n" + "=" * 70)
print("Accuracy on 5 questions means NOTHING statistically -- ignore it.")
print("The only thing that matters here: unparseable should be 0.")
print("If it is, run:  python llm_benchmark.py")
print("If it is not, the Modelfile TEMPLATE needs fixing first.")
print("=" * 70)
