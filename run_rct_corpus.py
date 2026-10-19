"""
Rebuild the retrieval corpus from PubMed 20k RCT and re-run the ablation.
========================================================================

WHAT THIS DOES
--------------
Replaces the corpus - not adds to it. The 100,000-abstract index is left
completely untouched; everything new is written to separate paths:

    data/chunks/rct_chunks.csv      new chunk table
    vector_db_rct/                  new FAISS index
    bm25s_index_rct/                new BM25 index
    evaluation/results/rct_*.csv    new results

Nothing overwrites the main study. If the result is uninteresting you delete
three folders and you are back where you started.

WHAT IS BEING TESTED
--------------------
Corpus COMPOSITION at reduced size. The comparison is:

    stage A   generator alone, no retrieval        (reused, see below)
    stage C   generator + retrieval over 20k RCT   (this script)

Stage A does not touch the corpus at all, so the existing
full_medqa_A.csv / full_medmcqa_A.csv are still valid and are reused. That
saves about four hours and, more importantly, keeps the no-retrieval
baseline identical to the one in the paper.

WHAT TO EXPECT - stated in advance so the result cannot be rationalised
----------------------------------------------------------------------
Coverage drops from 0.42% of PubMed to 0.084%, five times lower, and RCT
abstracts report trial outcomes while MedQA asks about mechanism and
diagnosis. The honest prediction is that the retrieval penalty gets LARGER,
not smaller. Writing that down first is the point: if the penalty shrinks
instead, that is a real and surprising finding about composition beating
coverage, and it will be believable precisely because it was not the
expected outcome.

MAKING IT A CONTROLLED COMPARISON
---------------------------------
20k RCT differs from the main corpus in two ways at once - size AND
composition. To attribute anything to composition you need the size-matched
control, which is one line:

    python run_corpus_sweep.py --levels 20000

That builds a random 20,000-abstract subset of your existing PubMed corpus.
Same size, same genre mix as the original, different composition from RCT.
Run both and you have a clean two-arm comparison at matched coverage.

TWO DETAILS THAT MATTER
-----------------------
1. The RCT text is TOKENISED: "pain , mobility" and "( OA )" with spaces
   before punctuation. Your existing corpus is raw text. Left alone, that is
   a second confound - the embedding model sees differently-formatted text.
   This script detokenises before chunking.

2. The repository ships two versions. We use the one with real numbers, not
   the *_numbers_replaced_with_at_sign variant, which substitutes @ for every
   digit and would destroy every dose, lab value and confidence interval.

SENTENCE LABELS
---------------
Each sentence carries BACKGROUND / OBJECTIVE / METHODS / RESULTS /
CONCLUSIONS. The labels are stored alongside every chunk, so
--conclusions-only can later build an index from the answer-bearing
sentences alone without re-downloading anything.

USAGE
-----
    python run_rct_corpus.py --build              # download, chunk, embed, index
    python run_rct_corpus.py --eval               # stage C on both datasets
    python run_rct_corpus.py --eval --finetuned   # also stage D
    python run_rct_corpus.py --build --conclusions-only   # RESULTS+CONCLUSIONS index

COST
----
    build   ~10 min   (36,693 chunks; download 39 MB, embed on GPU)
    eval    ~4 hours  (5,456 generations)
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ==========================
# Configuration
# ==========================

BASE_MODEL = "qwen2.5:7b"
FT_MODEL   = "qwen2.5-medqa-lora"

RESULT_DIR = "evaluation/results"
CHUNK_FILE = "data/chunks/rct_chunks.csv"
VECTOR_DIR = "vector_db_rct"
BM25_DIR   = "bm25s_index_rct"
RAW_DIR    = "data/raw/pubmed_rct"

# Chunking - identical to chunking_step3.py so the only thing that changes
# is which text goes in.
CHUNK_SIZE_WORDS   = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS    = 20

# Retrieval - identical to retriever.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K          = 8
RRF_K          = 60
CANDIDATE_POOL = 30

NUM_PREDICT      = 150
CHECKPOINT_EVERY = 20

GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0,
          "seed": 42, "repeat_penalty": 1.0}

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

BASE_URL = ("https://raw.githubusercontent.com/Franck-Dernoncourt/"
            "pubmed-rct/master/PubMed_20k_RCT/")
SPLITS = ["train.txt", "dev.txt", "test.txt"]

ANSWER_LABELS = ("RESULTS", "CONCLUSIONS")


# ==========================================================  BUILD

# Gold answers live under "answer_idx", not "answer". The eval JSON has exactly
# these keys: question, options, answer_idx, source.
ANSWER_KEY = "answer_idx"


def check_schema(questions, name=""):
    """Fail immediately on a schema mismatch instead of hours into a run."""
    if not questions:
        raise SystemExit(f"FAILED: {name} is empty")
    missing = [k for k in ("question", "options", ANSWER_KEY)
               if k not in questions[0]]
    if missing:
        raise SystemExit(
            f"FAILED: {name} items are missing {missing}. "
            f"Found keys: {sorted(questions[0].keys())}")


def detokenise(s):
    """The RCT files are whitespace-tokenised; restore normal spacing.

    Without this the encoder sees "pain , mobility ( OA ) ." which is not how
    the 100k corpus is written, and the comparison would confound corpus
    composition with text formatting.
    """
    s = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", s)
    s = re.sub(r"([\(\[])\s+", r"\1", s)
    s = re.sub(r"\s+/\s+", "/", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    for split in SPLITS:
        dest = os.path.join(RAW_DIR, split)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  {split}: already downloaded")
            continue
        print(f"  {split}: downloading ...", end="", flush=True)
        urllib.request.urlretrieve(BASE_URL + split, dest)
        print(f" {os.path.getsize(dest) / 1e6:.1f} MB")


def parse_abstracts():
    """Return [(pmid, [(label, sentence), ...]), ...]."""
    out, pmid, cur = [], None, []
    for split in SPLITS:
        path = os.path.join(RAW_DIR, split)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("###"):
                    if pmid and cur:
                        out.append((pmid, cur))
                    pmid, cur = line[3:].strip(), []
                elif line.strip():
                    parts = line.split("\t")
                    if len(parts) == 2:
                        cur.append((parts[0], detokenise(parts[1])))
    if pmid and cur:
        out.append((pmid, cur))
    return out


def chunk_text(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= size:
        return [text]
    out, step, start = [], size - overlap, 0
    while True:
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return out


def build_chunks(conclusions_only=False):
    print("Parsing abstracts ...")
    abstracts = parse_abstracts()
    print(f"  {len(abstracts):,} abstracts")

    rows = []
    for pmid, sents in abstracts:
        if conclusions_only:
            keep = [s for lab, s in sents if lab in ANSWER_LABELS]
            labels_present = ANSWER_LABELS
        else:
            keep = [s for _, s in sents]
            labels_present = tuple(sorted({lab for lab, _ in sents}))
        if not keep:
            continue

        body = " ".join(keep)
        title = keep[0][:120]        # RCT files carry no title field

        for ci, ch in enumerate(chunk_text(body)):
            if len(ch.split()) < MIN_CHUNK_WORDS:
                continue
            rows.append({
                "doc_id": pmid,
                "chunk_id": f"{pmid}_{ci}",
                "PMID": pmid,
                "title": title,
                "chunk": ch,
                "labels": "|".join(labels_present),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)
    df.to_csv(CHUNK_FILE, index=False)
    print(f"  {len(df):,} chunks -> {CHUNK_FILE}")
    return df


def build_indexes(df):
    import faiss
    import bm25s
    from sentence_transformers import SentenceTransformer
    import torch

    texts = df["chunk"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(texts):,} chunks on {device} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    vecs = model.encode(texts, batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    vecs = np.ascontiguousarray(vecs.astype("float32"))

    os.makedirs(VECTOR_DIR, exist_ok=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "minilm_index.faiss"))
    print(f"  FAISS index -> {VECTOR_DIR}/minilm_index.faiss ({index.ntotal:,} vectors)")

    print("Building BM25 ...")
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokens, show_progress=False)
    bm25.save(BM25_DIR)
    print(f"  BM25 index -> {BM25_DIR}")


# ==========================================================  RETRIEVE
class RctRetriever:
    """Hybrid RRF over the RCT indexes. Mirrors retriever.py exactly:
    dense + sparse, CANDIDATE_POOL each, RRF with RRF_K, top-k returned."""

    def __init__(self):
        import faiss
        import bm25s
        from sentence_transformers import SentenceTransformer
        import torch

        self.df = pd.read_csv(CHUNK_FILE)
        self.index = faiss.read_index(os.path.join(VECTOR_DIR, "minilm_index.faiss"))
        self.bm25 = bm25s.BM25.load(BM25_DIR, mmap=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self._bm25s = bm25s

        if self.index.ntotal != len(self.df):
            sys.exit(f"index has {self.index.ntotal} vectors but chunk table has "
                     f"{len(self.df)} rows - rebuild with --build")

    def _dense(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
        _, idx = self.index.search(q, k)
        return [int(i) for i in idx[0] if i >= 0]

    def _sparse(self, query, k):
        k = min(k, len(self.df))
        toks = self._bm25s.tokenize(query, stopwords="en", show_progress=False)
        res, _ = self.bm25.retrieve(toks, k=k, show_progress=False)
        return [int(i) for i in res[0]]

    def context(self, query, k=TOP_K):
        rrf = {}
        for rank, i in enumerate(self._dense(query, CANDIDATE_POOL)):
            rrf[i] = rrf.get(i, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(self._sparse(query, CANDIDATE_POOL)):
            rrf[i] = rrf.get(i, 0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [str(self.df.iloc[i]["chunk"]) for i, _ in top]


# ==========================================================  EVALUATE
def show_decoding():
    """Print the decoding settings actually in force.

    This project once reported results from two arms decoding under different
    rules without anyone noticing, so the settings get printed rather than
    assumed.
    """
    import rag_pipeline
    keys = ("temperature", "top_p", "top_k", "seed", "repeat_penalty",
            "num_ctx", "num_gpu")
    active = {k: rag_pipeline.OLLAMA_OPTIONS.get(k) for k in keys
              if k in rag_pipeline.OLLAMA_OPTIONS}
    print("Decoding in force:", ", ".join(f"{k}={v}" for k, v in active.items()))
    missing = [k for k in GREEDY if rag_pipeline.OLLAMA_OPTIONS.get(k) != GREEDY[k]]
    if missing:
        raise SystemExit(f"FAILED: greedy settings did not take effect: {missing}")


def preflight(model):
    from rag_pipeline import generate_answer
    print(f"Preflight: {model} ... ", end="", flush=True)
    reply = generate_answer("Reply with exactly: OK", model=model,
                            num_predict=8)
    if not isinstance(reply, str) or not reply.strip():
        sys.exit(f"\nFAILED: empty reply from {model}. Is ollama running?")
    if any(w in reply.lower() for w in ("error", "connection", "refused",
                                        "not found", "try pulling")):
        sys.exit(f"\nFAILED: {model} error string: {reply[:200]}")
    print("ok")


def run_arm(tag, dataset_key, questions, retr, model):
    from rag_pipeline import generate_answer
    from mcqa_evaluation import build_mcqa_prompt, extract_choice

    path = os.path.join(RESULT_DIR, f"rct_{dataset_key}_{tag}.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): iterrows yields Series, and a
        # list mixing Series with the dicts appended below makes
        # pd.DataFrame() raise "'dict' object has no attribute 'dtype'".
        # This only ever bites on resume, which is why a fresh run looked fine.
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"  {tag}/{dataset_key}: complete")
            return pd.DataFrame(list(done.values()))
        print(f"  {tag}/{dataset_key}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    t0 = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue
        chunks = retr.context(item["question"])
        ctx = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"
        prompt = build_mcqa_prompt(ctx, item["question"], item["options"])
        reply = generate_answer(prompt, model=model,
                                num_predict=NUM_PREDICT)
        pred = extract_choice(reply)

        rows.append({"index": i, "true_answer": item[ANSWER_KEY],
                     "prediction": pred, "unparseable": pred is None,
                     "correct": (pred == item[ANSWER_KEY]),
                     "raw_on_failure": reply[:300] if pred is None else ""})

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - t0, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"    {i+1}/{len(questions)}  {rate:.2f} it/s  ~{left:.0f} min left",
                  end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n  {tag}/{dataset_key}: accuracy {100*df.correct.mean():.2f}%, "
          f"unparseable {100*df.unparseable.mean():.2f}%")
    return df


def mcnemar(a_ok, b_ok):
    gained = int(((~a_ok) & b_ok).sum())
    lost = int((a_ok & (~b_ok)).sum())
    delta = 100.0 * (b_ok.mean() - a_ok.mean())
    if gained + lost == 0:
        return round(delta, 2), gained, lost, 1.0
    return (round(delta, 2), gained, lost,
            float(binomtest(gained, gained + lost, 0.5).pvalue))


def evaluate(with_finetuned):
    import rag_pipeline
    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)

    for p in (CHUNK_FILE, os.path.join(VECTOR_DIR, "minilm_index.faiss"), BM25_DIR):
        if not os.path.exists(p):
            sys.exit(f"missing {p} - run --build first")

    show_decoding()
    preflight(BASE_MODEL)
    if with_finetuned:
        preflight(FT_MODEL)

    print("Loading RCT indexes ...")
    retr = RctRetriever()
    print(f"  {len(retr.df):,} chunks ready")

    summary = []
    for ds in ("medqa", "medmcqa"):
        with open(DATASET_FILES[ds], encoding="utf-8") as f:
            questions = json.load(f)
        check_schema(questions, ds)
        print(f"\n{ds}: {len(questions)} items")

        # Stage A is corpus-independent, so the main study's file is reused.
        a_path = os.path.join(RESULT_DIR, f"full_{ds}_A.csv")
        if not os.path.exists(a_path):
            sys.exit(f"missing {a_path} - the no-retrieval baseline comes from "
                     f"the main study and is not recomputed here")
        a_ok = pd.read_csv(a_path).set_index("index")["correct"].astype(bool)

        arms = [("C", BASE_MODEL)] + ([("D", FT_MODEL)] if with_finetuned else [])
        for tag, model in arms:
            df = run_arm(tag, ds, questions, retr, model)
            x_ok = df.set_index("index")["correct"].astype(bool)
            j = pd.concat([a_ok.rename("a"), x_ok.rename("x")], axis=1).dropna()
            d, g, l, p = mcnemar(j.a.astype(bool), j.x.astype(bool))

            summary.append({
                "dataset": ds, "corpus": "pubmed_20k_rct", "arm": tag,
                "n": len(j),
                "acc_no_retrieval": round(100 * j.a.mean(), 2),
                "acc_with_retrieval": round(100 * j.x.mean(), 2),
                "retrieval_effect_pp": d, "gained": g, "lost": l,
                "mcnemar_p": round(p, 6), "significant": p < 0.05,
            })
            print(f"  A->{tag}: {d:+.2f} pp, +{g}/-{l}, p={p:.4g}")
            pd.DataFrame(summary).to_csv(
                os.path.join(RESULT_DIR, "rct_corpus_summary.csv"), index=False)

    out = pd.DataFrame(summary)
    print("\n" + "=" * 70)
    print(out.to_string(index=False))
    print("=" * 70)
    print(f"""
COMPARE AGAINST THE MAIN STUDY

  100k PubMed, stage A->C:  MedQA -2.75 pp (p=0.034)
                            MedMCQA -5.71 pp (p=2.5e-14)

  (The paper's -3.06 / -4.26 figures are A->B, the NAIVE retrieval arm.
   A->C is the correct comparison here because this script's arm C uses
   the tuned retrieval configuration.)

  20k RCT, stage C:       the rows above

AND AGAINST THE SIZE-MATCHED CONTROL

  python run_corpus_sweep.py --levels 20000

  That is 20,000 random abstracts from your original corpus. Only with that
  arm can a difference be attributed to COMPOSITION rather than to size.

READING THE OUTCOME

  penalty larger than 100k   -> expected. Less coverage, worse genre match.
                                Confirms the coverage mechanism from below.

  penalty about the same     -> corpus size matters less than the paper
                                assumes between 20k and 100k. Check the
                                sweep for where the curve actually bends.

  penalty smaller            -> surprising, and the interesting case. It
                                would mean a focused corpus beats a larger
                                scattered one. Before believing it, confirm
                                against the size-matched control - otherwise
                                size and composition are confounded.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="download, chunk, embed and index the RCT corpus")
    ap.add_argument("--eval", action="store_true",
                    help="run stage C (and D) against the RCT corpus")
    ap.add_argument("--finetuned", action="store_true",
                    help="with --eval, also run the fine-tuned model")
    ap.add_argument("--conclusions-only", action="store_true",
                    help="with --build, index only RESULTS and CONCLUSIONS sentences")
    args = ap.parse_args()

    if not (args.build or args.eval):
        ap.error("choose --build and/or --eval")

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.build:
        print("=" * 70)
        print("BUILDING RCT CORPUS" +
              ("  (RESULTS + CONCLUSIONS only)" if args.conclusions_only else ""))
        print("=" * 70)
        download()
        df = build_chunks(args.conclusions_only)
        build_indexes(df)
        print("\nBuild complete. The 100k corpus was not touched.")

    if args.eval:
        print("\n" + "=" * 70)
        print("EVALUATING AGAINST THE RCT CORPUS")
        print("=" * 70)
        evaluate(args.finetuned)


if __name__ == "__main__":
    main()
