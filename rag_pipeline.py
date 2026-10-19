import os

# Must be set BEFORE numpy/faiss/sentence-transformers are imported.
# Prevents "OpenBLAS: malloc failed in gemm_driver" crashes caused by
# OpenBLAS spawning too many threads and over-allocating memory,
# especially common on Windows with many CPU cores.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import requests

import retriever


# ==================================
# Configuration
# ==================================

# Which embedding model + retrieval strategy to use. Change these to
# switch models/modes without touching retriever.py — this overrides
# retriever.py's own ACTIVE_MODEL/RETRIEVAL_MODE defaults for calls
# made through this file.
ACTIVE_MODEL = "bge"

# "dense" | "sparse" | "hybrid"
RETRIEVAL_MODE = "hybrid"

TOP_K = 8

LLM_MODEL = "qwen2.5:7b"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# VRAM-friendly generation options for RTX 3060 12GB.
# num_ctx: smaller context window uses less VRAM.
# num_predict: caps answer length so generation doesn't run away.
# num_gpu: number of layers offloaded to GPU (rest stay on CPU).
#          Lower this further (e.g. 15) if OOM persists.
OLLAMA_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 500,
    "num_gpu": 20,
}

# Keep the model loaded for 5 minutes between calls so it isn't
# reloaded into VRAM for every single question during evaluation.
OLLAMA_KEEP_ALIVE = "5m"


# ==================================
# Retrieval (delegated to retriever.py)
# ==================================
#
# retriever.py handles model loading, FAISS/BM25 index loading, and
# the dense/sparse/hybrid RRF fusion logic. This file just calls it
# with the ACTIVE_MODEL / RETRIEVAL_MODE / TOP_K configured above, so
# switching embedding models for benchmarking means changing the
# three constants above — no other code needs to change.

def retrieve_context(question, k=None, mode=None, model_key=None):

    k = k or TOP_K
    mode = mode or RETRIEVAL_MODE
    model_key = model_key or ACTIVE_MODEL

    results = retriever.retrieve_context(question, k=k, mode=mode, model_key=model_key)

    # retriever.py returns dicts with metadata; rag_answer() below only
    # needs the chunk text itself for building the prompt.
    return [r["chunk"] for r in results]


# ==================================
# Ollama LLM
# ==================================

def _call_ollama(prompt, num_gpu, model=None, num_predict=None):

    # model=None uses the default LLM_MODEL (Qwen2.5) — passing an
    # explicit model lets callers like llm_benchmark.py swap in a
    # different Ollama model per call without touching this file's
    # constants. Ollama itself handles loading/unloading models in
    # VRAM as the requested model changes between calls.
    #
    # num_predict=None uses OLLAMA_OPTIONS' default (500). Passing a
    # smaller value (e.g. for short multiple-choice answers) caps how
    # many tokens a model can ramble through before stopping — some
    # models (observed: meditron:7b) don't reliably respect "answer in
    # one sentence" and will generate up to the full token budget
    # every time, which is the dominant cause of high per-call latency
    # for those models, not GPU/CPU speed itself.
    options = dict(OLLAMA_OPTIONS)
    options["num_gpu"] = num_gpu

    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model or LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": options,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "").strip()


def generate_answer(prompt, model=None, num_predict=None):

    num_gpu = OLLAMA_OPTIONS["num_gpu"]

    try:

        output = _call_ollama(prompt, num_gpu, model=model, num_predict=num_predict)

        if output:
            return output

        return "No response generated."

    except requests.exceptions.Timeout:

        return "Ollama timeout"

    except requests.exceptions.RequestException as e:

        # Try to pull the actual error body from Ollama's response,
        # since a generic 500 alone doesn't tell us the real cause.
        body_text = ""

        if getattr(e, "response", None) is not None:

            try:
                body_text = e.response.text
            except Exception:
                body_text = ""

        error_text = (str(e) + " " + body_text).lower().replace("-", " ")

        is_oom = (
            "out of memory" in error_text
            or "cudamalloc" in error_text
            or "unable to allocate" in error_text
            or "failed to allocate" in error_text
            or "bad_alloc" in error_text
        )

        if is_oom:

            print(
                "OOM detected, retrying with fewer GPU layers "
                "(num_gpu=10) after a short cooldown..."
            )

            time.sleep(5)

            try:

                output = _call_ollama(prompt, num_gpu=10, model=model, num_predict=num_predict)

                if output:
                    return output

                return "No response generated."

            except Exception as e2:

                return "Ollama Error (after retry):\n" + str(e2)

        return "Ollama Error:\n" + str(e) + "\nBody: " + body_text

    except Exception as e:

        return str(e)


# ==================================
# RAG Function
# ==================================

def build_prompt(context_text, question, mode="explanatory"):
    """
    Builds the LLM prompt from a given context_text + question.

    Pulled out as its own function (rather than inlined in rag_answer)
    so that other scripts — e.g. an oracle-context diagnostic test
    that bypasses retrieval and feeds the TRUE source passage
    directly — can reuse the EXACT same prompt template. That keeps
    any accuracy comparison between "RAG-retrieved context" and
    "oracle/true context" a fair one (same prompt wording either way).
    """

    if mode == "classification":

        # Used for yes/no/maybe benchmarks like PubMedQA.
        # Forces an unambiguous marker line so extract_label() never
        # has to guess from free-text hedging language like
        # "no significant risk" or "no clear evidence".
        prompt = f"""

You are a medical research assistant.


========================

Medical Context:

{context_text}


========================

Question:

{question}


Instructions:

Answer the question using only the context above. Do not invent
information.

Give a definitive "yes" or "no" whenever the context provides
supporting evidence, even if it isn't 100% conclusive. Only answer
"maybe" if the context is genuinely silent on this specific question
or gives directly conflicting evidence with no majority direction.

First give a SHORT explanation in ONE sentence only (max ~25 words).
Then, on a new final line, output EXACTLY in this format (replace
<label> with yes, no, or maybe), with nothing else after it:

FINAL_ANSWER: <label>

Answer:

"""

    else:

        # Default: free-text explanatory answer (used for general
        # queries and interactive use).
        prompt = f"""

You are a medical research assistant.

Answer the question using only the provided PubMed research context.

Do not invent information.

Give a scientific but easy-to-understand explanation.


========================

Medical Context:

{context_text}


========================

Question:

{question}


Answer:

"""

    return prompt


def rag_answer(question, mode="explanatory"):

    start = time.time()

    # Retrieve documents
    contexts = retrieve_context(question)

    if not contexts:

        print("No relevant context found.")

        return "No relevant context found."

    context_text = "\n\n".join(contexts)

    prompt = build_prompt(context_text, question, mode=mode)

    answer = generate_answer(prompt)

    latency = time.time() - start

    print("\n==============================")

    print("QUESTION:")

    print(question)

    print("\nANSWER:")

    print(answer)

    print("\nLatency:", round(latency, 2), "seconds")

    print("==============================")

    return answer


# ==================================
# Main
# ==================================

if __name__ == "__main__":

    print("\nMedical RAG System Started")

    print(f"ACTIVE_MODEL = {ACTIVE_MODEL}, RETRIEVAL_MODE = {RETRIEVAL_MODE}")

    while True:

        query = input("\nMedical Query (exit): ")

        if query.lower() == "exit":
            break

        rag_answer(query)