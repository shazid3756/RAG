"""
Build a 1M-abstract corpus on 16GB of RAM without ever holding it in RAM.
========================================================================

WHY THIS EXISTS
---------------
The downward sweep showed the retrieval penalty growing monotonically from
10k to 100k abstracts. But the oracle diagnostic says that with the RIGHT
passage in hand the model reaches 63%, far above any arm measured here. Both
cannot be the whole story, so the curve must turn somewhere: past some
coverage, having the answer present has to start outweighing better-looking
distractors.

This script builds the corpus needed to look for that turning point. It is the
one experiment in the project that could convert "retrieval hurts" into
"retrieval hurts below X% coverage and helps above it", which is a deployment
threshold rather than a warning.

THE MEMORY PROBLEM, AND HOW IT IS AVOIDED
-----------------------------------------
This machine has 16GB and a documented history of OOM during this project.
1M abstracts is roughly 1.1M chunks. Doing it the obvious way - read all
abstracts into a DataFrame, chunk into a list, call model.encode(everything) -
needs well over 20GB and would die, probably an hour in.

Three rules are followed instead, and none of them can be relaxed:

  1. The abstracts are never all in memory. They are streamed from the
     HuggingFace dataset and written to the chunk CSV in batches.
  2. The chunks are never all in memory. Phase 2 reads the CSV back with
     pandas' chunksize iterator.
  3. The embeddings are never all in memory. They are written straight into a
     numpy memmap on disk, batch by batch.

Peak RSS stays near a few hundred MB regardless of corpus size. The cost is
disk: ~1.6GB for the embeddings of 1.1M chunks at 384 dims, plus ~1GB of CSV.

FAISS INDEX
-----------
IndexFlatIP over normalised vectors, added in slices from the memmap so the
full matrix is never materialised at once. The index itself does hold all
vectors (that is what a flat index is), which is ~1.6GB - fine on this
machine, and it is loaded only when the sweep runs, not during the build.

NO BM25 HERE, DELIBERATELY
--------------------------
The coverage sweep uses dense retrieval only, because that is the one axis it
varies. Building a BM25 index over 1.1M chunks would cost time and disk for
something the experiment does not consume. If a later experiment needs hybrid
retrieval at this scale, build it then.

RESUMABLE
---------
Every phase checkpoints and skips completed work. A power cut or Ctrl+C costs
at most one batch. Re-run the same command to continue.

USAGE
-----
    python collect_large_corpus.py --target 1000000            # all phases
    python collect_large_corpus.py --target 1000000 --phase collect
    python collect_large_corpus.py --phase embed
    python collect_large_corpus.py --phase index
    python collect_large_corpus.py --status
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import json
import re
import sys
import time

import numpy as np
import pandas as pd


# ==========================
# Configuration
# ==========================

TARGET_DEFAULT = 1_000_000

OUT_DIR      = "data/large"
CHUNK_FILE   = os.path.join(OUT_DIR, "pubmed_1m_chunks.csv")
EMB_FILE     = os.path.join(OUT_DIR, "pubmed_1m_minilm.f32")
META_FILE    = os.path.join(OUT_DIR, "large_corpus_meta.json")
INDEX_DIR    = "vector_db_large"
INDEX_FILE   = os.path.join(INDEX_DIR, "minilm_index.faiss")

# Matches the main pipeline exactly - a different chunking would make the
# sweep compare two things at once.
CHUNK_SIZE_WORDS    = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS     = 20
MIN_CONTENT_WORDS   = 10

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM   = 384
ENCODE_BATCH = 256          # sentence-transformers batch
CSV_READ_CHUNK = 20_000     # rows pulled off disk at a time
FLUSH_EVERY_DOCS = 2_000

FIELDS = ["chunk_id", "doc_id", "PMID", "title", "chunk"]


# ==========================
# Shared helpers
# ==========================

def normalise_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_text(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= size:
        return [text]
    out, step, start = [], size - overlap, 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return out


def read_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_meta(**kw):
    m = read_meta()
    m.update(kw)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m


def count_chunks():
    """Row count without loading the file.

    The chunk text can contain quoted newlines, so a raw line count is wrong.
    Counting through the csv reader is slower but correct, and it only runs
    once per phase.
    """
    if not os.path.exists(CHUNK_FILE):
        return 0
    n = 0
    with open(CHUNK_FILE, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for _ in r:
            n += 1
    return n


# ==========================
# Phase 1 - collect + clean + chunk, streaming
# ==========================

def phase_collect(target):
    from datasets import load_dataset

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = read_meta()

    docs_done = int(meta.get("docs_collected", 0))
    raw_seen  = int(meta.get("raw_consumed", 0))
    chunk_id  = int(meta.get("chunks_written", 0))

    if docs_done >= target:
        print(f"collect: already have {docs_done:,} / {target:,} abstracts")
        return

    if docs_done:
        print(f"collect: resuming at {docs_done:,} abstracts "
              f"({chunk_id:,} chunks, {raw_seen:,} raw rows consumed)")
    else:
        print(f"collect: starting fresh, target {target:,} abstracts")

    ds = load_dataset("MedRAG/pubmed", split="train", streaming=True)
    if raw_seen:
        print(f"  skipping {raw_seen:,} already-consumed rows ...")
        ds = ds.skip(raw_seen)

    mode = "a" if (os.path.exists(CHUNK_FILE) and chunk_id) else "w"
    seen_hashes = set()          # dedup within this run only; see note below

    started = time.time()
    batch_docs = 0

    with open(CHUNK_FILE, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()

        try:
            for row in ds:
                raw_seen += 1

                title = normalise_whitespace(row.get("title") or "")
                content = normalise_whitespace(row.get("content") or "")
                if not title or not content:
                    continue
                if len(content.split()) < MIN_CONTENT_WORDS:
                    continue

                # Exact-duplicate abstracts are dropped, matching
                # cleaning_step2. A hash set over 1M items is ~60MB, which is
                # affordable; storing the texts themselves would not be.
                h = hash(content)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                for piece in chunk_text(content):
                    full = f"{title}. {piece}" if title else piece
                    if len(full.split()) < MIN_CHUNK_WORDS:
                        continue
                    w.writerow({"chunk_id": chunk_id,
                                "doc_id": row.get("id"),
                                "PMID": row.get("PMID"),
                                "title": title,
                                "chunk": full})
                    chunk_id += 1

                docs_done += 1
                batch_docs += 1

                if batch_docs >= FLUSH_EVERY_DOCS:
                    f.flush()
                    write_meta(docs_collected=docs_done, raw_consumed=raw_seen,
                               chunks_written=chunk_id, target=target)
                    rate = docs_done / max(time.time() - started, 1e-9)
                    left = (target - docs_done) / max(rate, 1e-9) / 60
                    print(f"  {docs_done:,}/{target:,} abstracts  "
                          f"{chunk_id:,} chunks  ~{left:.0f} min left   ",
                          end="\r")
                    batch_docs = 0

                if docs_done >= target:
                    break
        finally:
            f.flush()
            write_meta(docs_collected=docs_done, raw_consumed=raw_seen,
                       chunks_written=chunk_id, target=target)

    print(f"\ncollect: {docs_done:,} abstracts -> {chunk_id:,} chunks "
          f"-> {CHUNK_FILE}")
    if docs_done < target:
        print("  target not reached (stream ended or interrupted) - "
              "re-run to continue")


# ==========================
# Phase 2 - embed into a memmap
# ==========================

def phase_embed():
    import torch
    from sentence_transformers import SentenceTransformer

    n_chunks = count_chunks()
    if not n_chunks:
        sys.exit("embed: no chunks found - run --phase collect first")

    meta = read_meta()
    done = int(meta.get("embedded", 0))
    if done >= n_chunks:
        print(f"embed: already complete ({done:,} vectors)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embed: {n_chunks:,} chunks, device={device}, resuming at {done:,}")

    # Pre-size the memmap once. Opening in r+ afterwards keeps prior work.
    if not os.path.exists(EMB_FILE) or done == 0:
        mm = np.memmap(EMB_FILE, dtype="float32", mode="w+",
                       shape=(n_chunks, EMBED_DIM))
        del mm
        done = 0
    mm = np.memmap(EMB_FILE, dtype="float32", mode="r+",
                   shape=(n_chunks, EMBED_DIM))

    model = SentenceTransformer(EMBED_MODEL, device=device)

    pos = 0
    started = time.time()
    reader = pd.read_csv(CHUNK_FILE, chunksize=CSV_READ_CHUNK,
                         usecols=["chunk"])

    for part in reader:
        lo, hi = pos, pos + len(part)
        pos = hi
        if hi <= done:                      # whole slice already embedded
            continue

        texts = part["chunk"].astype(str).tolist()
        if lo < done:                       # partially done: redo the tail only
            texts = texts[done - lo:]
            lo = done

        emb = model.encode(texts, batch_size=ENCODE_BATCH,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
        mm[lo:lo + len(emb)] = np.asarray(emb, dtype="float32")
        mm.flush()

        done = lo + len(emb)
        write_meta(embedded=done, embed_dim=EMBED_DIM, embed_model=EMBED_MODEL,
                   n_chunks=n_chunks)

        rate = done / max(time.time() - started, 1e-9)
        left = (n_chunks - done) / max(rate, 1e-9) / 60
        print(f"  {done:,}/{n_chunks:,}  {rate:.0f} chunks/s  "
              f"~{left:.0f} min left   ", end="\r")

        del emb
        gc.collect()

    del mm, model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\nembed: {done:,} vectors -> {EMB_FILE}")


# ==========================
# Phase 3 - FAISS index
# ==========================

def phase_index():
    import faiss

    meta = read_meta()
    n_chunks = int(meta.get("n_chunks") or count_chunks())
    done = int(meta.get("embedded", 0))
    if done < n_chunks:
        sys.exit(f"index: only {done:,}/{n_chunks:,} chunks embedded - "
                 f"finish --phase embed first")

    os.makedirs(INDEX_DIR, exist_ok=True)
    mm = np.memmap(EMB_FILE, dtype="float32", mode="r",
                   shape=(n_chunks, EMBED_DIM))

    index = faiss.IndexFlatIP(EMBED_DIM)   # cosine, vectors are normalised

    # Added in slices so the full matrix is never copied into RAM at once.
    step = 100_000
    for lo in range(0, n_chunks, step):
        hi = min(lo + step, n_chunks)
        index.add(np.ascontiguousarray(mm[lo:hi]))
        print(f"  indexed {hi:,}/{n_chunks:,}   ", end="\r")

    faiss.write_index(index, INDEX_FILE)
    write_meta(index_built=True, index_file=INDEX_FILE, index_ntotal=index.ntotal)
    print(f"\nindex: {index.ntotal:,} vectors -> {INDEX_FILE}")

    del index, mm
    gc.collect()


# ==========================
# Status
# ==========================

def phase_status():
    meta = read_meta()
    n_chunks = count_chunks()

    def size(p):
        return f"{os.path.getsize(p) / 1e9:.2f} GB" if os.path.exists(p) else "-"

    print("=" * 62)
    print("LARGE CORPUS STATUS")
    print("=" * 62)
    print(f"  target abstracts : {meta.get('target', '-'):,}"
          if isinstance(meta.get("target"), int) else "  target abstracts : -")
    print(f"  collected        : {meta.get('docs_collected', 0):,}")
    print(f"  chunks on disk   : {n_chunks:,}   ({size(CHUNK_FILE)})")
    print(f"  embedded         : {meta.get('embedded', 0):,}   ({size(EMB_FILE)})")
    print(f"  index built      : {bool(meta.get('index_built'))}   ({size(INDEX_FILE)})")
    print()
    print(f"  chunks file : {CHUNK_FILE}")
    print(f"  index file  : {INDEX_FILE}")
    print()
    if meta.get("index_built"):
        print("Ready. Run the upward sweep with:")
        print(f"  python run_corpus_sweep.py --chunks-file {CHUNK_FILE} \\")
        print(f"      --index-file {INDEX_FILE} \\")
        print(f"      --total-abstracts {meta.get('docs_collected', 0)} \\")
        print(f"      --levels 100000 250000 500000 {meta.get('docs_collected', 0)}")
    print("=" * 62)


# ==========================
# Main
# ==========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--phase", choices=["collect", "embed", "index", "all"],
                    default="all")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        phase_status()
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.phase in ("collect", "all"):
        phase_collect(args.target)
    if args.phase in ("embed", "all"):
        phase_embed()
    if args.phase in ("index", "all"):
        phase_index()

    phase_status()


if __name__ == "__main__":
    main()
