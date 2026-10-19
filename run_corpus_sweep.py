"""
Corpus-coverage sweep — turning "retrieval hurts at 0.4%" into a curve.
======================================================================

WHY THIS IS CHEAPER THAN IT LOOKS
---------------------------------
Sweeping coverage DOWNWARD needs no re-collection and no re-embedding. The
111,045 chunk vectors are already on disk. Taking a random subset of those
vectors and building a fresh FAISS index over it is an exact simulation of
having collected fewer abstracts, and it takes seconds per level rather than
hours.

So the sweep costs only generation: one context arm per coverage level.

WHAT IT TESTS
-------------
The paper's explanation for Finding 4 is corpus coverage: with 0.4% of the
literature indexed, retrieved passages are on-topic but do not contain the
answer, and that distracts a model which already holds the knowledge.

If that explanation is right, the retrieval penalty should SHRINK as coverage
rises and GROW as it falls. This script measures that slope. A monotone trend
across four levels is strong confirmation; a flat line falsifies the
explanation and would mean something else is going on - which is worth
knowing before a reviewer asks.

Note the direction of inference: going down from 100k confirms or refutes the
mechanism, but it cannot show where retrieval STARTS to help. For that you
need to collect more abstracts and sweep upward. Do this script first - it is
nearly free and it tells you whether the upward sweep is worth the days it
costs.

COST
----
MedQA only (default): 1,273 items x 4 levels = 5,092 generations, ~4 hours.
Both datasets: 21,824 generations, ~15 hours. Start with MedQA.
Stage A is reused from the existing full run, so the no-retrieval baseline
costs nothing.

USAGE
-----
    python run_corpus_sweep.py                 # MedQA, 4 levels
    python run_corpus_sweep.py --both          # both datasets
    python run_corpus_sweep.py --levels 10000 50000 100000
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import retriever
import rag_pipeline
from rag_pipeline import generate_answer
from mcqa_evaluation import build_mcqa_prompt, extract_choice


BASE_MODEL = "qwen2.5:7b"
RESULT_DIR = "evaluation/results"
EMBEDDING = "minilm"          # the configuration used for stage C
TOP_K = 8
NUM_PREDICT = 150
CHECKPOINT_EVERY = 20
SEED = 42

DATASET_FILES = {
    "medqa":   "evaluation/medqa_test.json",
    "medmcqa": "evaluation/medmcqa_val.json",
}

GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0,
          "seed": 42, "repeat_penalty": 1.0}

# Coverage levels as ABSTRACT counts. 100,000 is the full corpus, so that
# level should reproduce stage C and acts as a built-in sanity check.
DEFAULT_LEVELS = [10_000, 25_000, 50_000, 100_000]

TOTAL_ABSTRACTS = 100_000
TOTAL_CHUNKS = 111_045



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
    print(f"Preflight: {model} ... ", end="", flush=True)
    reply = generate_answer("Reply with exactly: OK", model=model,
                            num_predict=8)
    if not isinstance(reply, str) or not reply.strip():
        sys.exit(f"\nFAILED: empty reply from {model}. Is ollama running?")
    if any(w in reply.lower() for w in ("error", "connection", "refused")):
        sys.exit(f"\nFAILED: {model} error string: {reply[:200]}")
    print("ok")


def subset_index(level, rng, total_abstracts=TOTAL_ABSTRACTS):
    """Build a FAISS index over a random subset of the existing vectors.

    The production index is IndexFlatIP over L2-normalised vectors, so its
    contents can be read back exactly with reconstruct_n - no re-embedding.

    Returns (index, id_map) where id_map[i] is the original chunk row for
    position i in the subset, so retrieved hits still resolve against the
    unchanged chunk table.

    MEMORY
    ------
    At 100k abstracts this was free: 111,045 vectors is 170 MB, so
    reconstructing the whole matrix and slicing it cost nothing worth
    thinking about. At 1.16M chunks the same two lines are 1.78 GB for the
    reconstruction plus up to 1.78 GB for the slice, on top of the 1.78 GB
    index and a multi-GB chunk table, on a 16 GB machine that is also
    running a 7B generator. So two changes:

      * the full level reuses the loaded index directly - there is nothing
        to subset, and copying it to produce an identical index is pure
        waste;
      * every other level reconstructs in blocks and adds as it goes, so
        the transient peak is one block rather than the whole corpus.

    Neither changes what is computed: IndexFlatIP over the same vectors in
    the same order.
    """
    import faiss

    full, _ = retriever._load_dense(EMBEDDING)
    n_total = full.ntotal

    frac = level / float(total_abstracts)
    n_keep = min(int(round(frac * n_total)), n_total)

    if n_keep >= n_total:
        print(f"    index reused: all {n_total:,} chunks (no subsetting)")
        return full, np.arange(n_total)

    id_map = np.sort(rng.choice(n_total, size=n_keep, replace=False))

    dim = full.d
    index = faiss.IndexFlatIP(dim)                # cosine on normalised vectors
    BLOCK = 200_000
    pos = 0
    for start in range(0, n_total, BLOCK):
        stop = min(start + BLOCK, n_total)
        # id_map is sorted, so the ids for this block are one contiguous run
        end = np.searchsorted(id_map, stop, side="left")
        if end == pos:
            continue
        want = id_map[pos:end]
        block = np.asarray(full.reconstruct_n(start, stop - start),
                           dtype="float32")
        index.add(np.ascontiguousarray(block[want - start]))
        del block
        pos = end

    if index.ntotal != n_keep:
        raise RuntimeError(f"subset index has {index.ntotal:,} vectors, "
                           f"expected {n_keep:,}")
    print(f"    index built: {n_keep:,} / {n_total:,} chunks "
          f"({100 * n_keep / n_total:.1f}%)")
    return index, id_map


def retrieve_with(index, id_map, question, k=TOP_K):
    """Dense retrieval against the subset index, returning chunk texts.

    Mirrors retriever._dense_search, including the E5 query prefix rule, so
    that the 100,000 level reproduces stage C.
    """
    _, model = retriever._load_dense(EMBEDDING)
    _, needs_prefix = retriever.MODEL_HF_IDS[EMBEDDING]

    q = ("query: " + question) if needs_prefix else question
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    _, idx = index.search(q_emb, k)

    chunks_df = retriever._load_chunks()
    texts = []
    for j in idx[0]:
        if j < 0:
            continue
        row = chunks_df.iloc[int(id_map[j])]
        texts.append(str(row["chunk"]))
    return texts


def run_level(dataset_key, questions, level, index, id_map, tag="sweep"):
    path = os.path.join(RESULT_DIR, f"{tag}_{dataset_key}_{level}.csv")

    done = {}
    if os.path.exists(path):
        prev = pd.read_csv(path)
        # to_dict("records") not iterrows(): iterrows yields Series, and a
        # list mixing Series with the dicts appended below makes
        # pd.DataFrame() raise "'dict' object has no attribute 'dtype'".
        # This only ever bites on resume, which is why a fresh run looked fine.
        done = {int(r["index"]): r for r in prev.to_dict("records")}
        if len(done) == len(questions):
            print(f"    {level:,}: complete")
            return pd.DataFrame(list(done.values()))
        print(f"    {level:,}: resuming at {len(done)}/{len(questions)}")

    rows = [done[i] for i in sorted(done)]
    started = time.time()

    for i, item in enumerate(questions):
        if i in done:
            continue

        chunks = retrieve_with(index, id_map, item["question"])
        context = "\n\n".join(chunks) if chunks else "(no reference material retrieved)"
        prompt = build_mcqa_prompt(context, item["question"], item["options"])
        reply = generate_answer(prompt, model=BASE_MODEL,
                                num_predict=NUM_PREDICT)
        pred = extract_choice(reply)

        rows.append({"index": i, "true_answer": item[ANSWER_KEY],
                     "prediction": pred, "unparseable": pred is None,
                     "correct": (pred == item[ANSWER_KEY])})

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = (i + 1 - len(done)) / max(time.time() - started, 1e-9)
            left = (len(questions) - i - 1) / max(rate, 1e-9) / 60
            print(f"      {i + 1}/{len(questions)}  ~{left:.0f} min left",
                  end="\r")

    df = pd.DataFrame(rows).sort_values("index")
    df.to_csv(path, index=False)
    print(f"\n    {level:,}: accuracy {100 * df.correct.mean():.2f}%")
    return df


def mcnemar(a_ok, b_ok):
    gained = int(((~a_ok) & b_ok).sum())
    lost = int((a_ok & (~b_ok)).sum())
    delta = 100.0 * (b_ok.mean() - a_ok.mean())
    if gained + lost == 0:
        return round(delta, 2), gained, lost, 1.0
    return (round(delta, 2), gained, lost,
            float(binomtest(gained, gained + lost, 0.5).pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true",
                    help="run MedMCQA as well (about 4x the time)")
    ap.add_argument("--datasets", nargs="+", choices=["medqa", "medmcqa"],
                    default=None,
                    help="run exactly these datasets. Use it to add MedMCQA "
                         "to a tag whose MedQA levels are already finished: "
                         "--both would also work, because a complete level "
                         "resumes as a no-op, but naming the dataset says so "
                         "out loud instead of relying on that.")
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)

    # Pointing the sweep at a bigger corpus. Without these the script uses the
    # 100k production index, which can only be swept DOWNWARD. The upward half
    # of the curve needs a corpus built by collect_large_corpus.py.
    ap.add_argument("--chunks-file", default=None,
                    help="alternative chunk table (default: the 100k corpus)")
    ap.add_argument("--index-file", default=None,
                    help="alternative FAISS index over that chunk table")
    ap.add_argument("--total-abstracts", type=int, default=None,
                    help="abstracts behind that corpus; sets the subset fraction")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed for the subset draw. Changing it draws a "
                         "different random subset at the SAME coverage; use a "
                         "distinct --tag with it so the outputs do not "
                         "overwrite each other.")
    ap.add_argument("--tag", default="sweep",
                    help="output prefix, so an upward run cannot overwrite the "
                         "downward one")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    rag_pipeline.OLLAMA_OPTIONS.update(GREEDY)

    # ---- corpus selection -------------------------------------------------
    # By default everything comes from the 100k production corpus already
    # loaded by retriever. Passing --chunks-file/--index-file swaps in a
    # larger one built by collect_large_corpus.py, which is what makes the
    # UPWARD half of the curve possible: the subset trick can only ever go
    # down from whatever index it is given.
    total_abstracts = args.total_abstracts or TOTAL_ABSTRACTS

    if bool(args.chunks_file) != bool(args.index_file):
        sys.exit("--chunks-file and --index-file must be given together")

    if args.chunks_file:
        import faiss
        for p in (args.chunks_file, args.index_file):
            if not os.path.exists(p):
                sys.exit(f"missing {p}")
        if not args.total_abstracts:
            sys.exit("--total-abstracts is required with a custom corpus: the "
                     "subset fraction is level/total, and defaulting it to "
                     "100,000 would silently mis-scale every level")

        print(f"corpus override:\n  chunks {args.chunks_file}"
              f"\n  index  {args.index_file}"
              f"\n  abstracts behind it: {total_abstracts:,}")

        # Prime retriever's lazy globals so subset_index and retrieve_with,
        # which both call into it, see the large corpus instead of the 100k one.
        # Only the "chunk" column is ever read (retrieve_with), and at 1.16M
        # rows the other columns are hundreds of MB of RAM that nothing
        # touches. Load one column when it is there, everything when it is
        # not, so a differently-shaped chunk table still works.
        head = pd.read_csv(args.chunks_file, nrows=0)
        if "chunk" in head.columns:
            retriever._chunks_df = pd.read_csv(args.chunks_file,
                                               usecols=["chunk"])
        else:
            sys.exit(f"{args.chunks_file} has no 'chunk' column; found "
                     f"{list(head.columns)}")
        retriever._dense_index = faiss.read_index(args.index_file)
        retriever._load_dense._loaded_model_key = EMBEDDING
        if retriever._dense_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            hf_id, _ = retriever.MODEL_HF_IDS[EMBEDDING]
            retriever._dense_model = SentenceTransformer(
                hf_id, device="cuda" if torch.cuda.is_available() else "cpu")

        n_idx = retriever._dense_index.ntotal
        n_chunks = len(retriever._chunks_df)
        if n_idx != n_chunks:
            sys.exit(f"index has {n_idx:,} vectors but the chunk table has "
                     f"{n_chunks:,} rows - row i must map to vector i, so "
                     f"these files do not belong together")
        print(f"  {n_chunks:,} chunks, index verified aligned")

        if max(args.levels) > total_abstracts:
            sys.exit(f"level {max(args.levels):,} exceeds the corpus "
                     f"({total_abstracts:,} abstracts) - collect more first")

    print("=" * 68)
    print("CORPUS-COVERAGE SWEEP")
    print("levels:", ", ".join(f"{l:,}" for l in args.levels))
    print(f"corpus: {total_abstracts:,} abstracts   tag: {args.tag}   "
          f"subset seed: {args.seed}")
    print("=" * 68)
    show_decoding()
    preflight(BASE_MODEL)

    if args.datasets:
        datasets = list(dict.fromkeys(args.datasets))
    else:
        datasets = ["medqa", "medmcqa"] if args.both else ["medqa"]
    summary = []

    for ds in datasets:
        with open(DATASET_FILES[ds], "r", encoding="utf-8") as f:
            questions = json.load(f)
        check_schema(questions, ds)
        print(f"\n{ds}: {len(questions)} items")

        a_path = os.path.join(RESULT_DIR, f"full_{ds}_A.csv")
        if not os.path.exists(a_path):
            sys.exit(f"missing {a_path} - run the main study first")
        a_ok = pd.read_csv(a_path).set_index("index")["correct"].astype(bool)

        for level in args.levels:
            print(f"  coverage {level:,} abstracts")
            # One rng per level, seeded from --seed, so a level's subset
            # depends only on (level, seed) and not on how many levels ran
            # before it. Re-running with a different --seed draws a DIFFERENT
            # random subset at the same coverage, which is how the
            # corpus-sampling spread at fixed coverage gets measured.
            index, id_map = subset_index(level,
                                         np.random.default_rng(args.seed),
                                         total_abstracts)
            df = run_level(ds, questions, level, index, id_map, args.tag)
            c_ok = df.set_index("index")["correct"].astype(bool)

            j = pd.concat([a_ok.rename("a"), c_ok.rename("c")], axis=1).dropna()
            d, g, l, p = mcnemar(j.a.astype(bool), j.c.astype(bool))
            summary.append({
                "dataset": ds, "abstracts": level,
                "coverage_pct_of_pubmed": round(100 * level / 23_900_000, 4),
                "corpus_abstracts": total_abstracts,
                "acc_no_retrieval": round(100 * j.a.mean(), 2),
                "acc_with_retrieval": round(100 * j.c.mean(), 2),
                "retrieval_effect_pp": d, "gained": g, "lost": l,
                "mcnemar_p": round(p, 6), "significant": p < 0.05,
            })
            print(f"    retrieval effect {d:+.2f} pp, p={p:.4g}")

    out = pd.DataFrame(summary)
    out.to_csv(os.path.join(RESULT_DIR, f"{args.tag}_summary.csv"), index=False)
    print("\n" + "=" * 68)
    print(out.to_string(index=False))
    print("=" * 68)
    print("""
HOW TO READ THIS

  This footer used to predict what the first sweep would show. It has been
  rewritten because that sweep already ran and answered the question: the
  penalty DEEPENS as coverage rises (-1.83 pp per decade on the 100k corpus,
  -2.32 on the 1M corpus), which falsified the coverage explanation rather
  than confirming it. Do not read a deepening trend here as a bug.

  What to check, depending on what was run:

  a level whose effect sits far outside its neighbours
      -> before believing it, look at the unparseable count in the
         per-question CSV. A level that silently lost the generator scores
         near chance and can masquerade as a real effect.

  repeat draws at ONE level (different --seed, same --levels)
      -> the spread across seeds is the corpus-sampling noise floor. Any
         level-to-level difference smaller than that spread is not evidence
         of a trend, however clean the ordering looks.

  a level that should reproduce an earlier run but does not
      -> check --seed and --total-abstracts first. The subset fraction is
         level / total, so a wrong total silently rescales every level, and
         a different seed draws different passages at the same coverage.
""")


if __name__ == "__main__":
    main()
