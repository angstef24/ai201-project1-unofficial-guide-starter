"""
Retrieval — Milestone 4, stage 2 of the pipeline.

Pipeline stage (from planning.md architecture diagram):

    ChromaDB (cosine)  ->  retrieve(query, k=4-8)  ->  ranked chunks + sources
     (vector store)        (this file)                 (to generation later)

retrieve() accepts a query string and returns the top-k most relevant chunks
along with their source information. Top-k is adaptive within the planning.md
4-8 range: more chunks for opinionated questions, fewer for factual ones. Weak
retrievals are flagged so the generation step can say "I didn't find much"
instead of answering confidently from poor evidence.

The embedding model and vector store live in embedding.py; we import _embed()
and get_collection() from there so the query is embedded with the SAME model
the chunks were embedded with.
"""

from __future__ import annotations

from typing import Optional

from embedding import _embed, get_collection

# Cosine similarity below this means "we didn't really find anything relevant".
# 1.0 = identical direction, 0.0 = unrelated. 0.30 is a conservative floor for
# MiniLM on short English text — tune after seeing real query scores.
LOW_SIMILARITY_THRESHOLD = 0.30

# Words that hint a question wants opinions/experiences rather than a fact.
_OPINION_HINTS = (
    "say", "think", "opinion", "people", "vibe", "like living", "experience",
    "best", "worst", "recommend", "better", "social", "quiet", "feel",
)


def suggest_k(query: str) -> int:
    """Pick top-k in the planning.md 4-8 range based on question type.

    Opinionated questions need more context (more voices) -> 8.
    Factual lookups need fewer, tighter chunks -> 5.
    """
    q = query.lower()
    return 8 if any(h in q for h in _OPINION_HINTS) else 5


def retrieve(query: str, k: Optional[int] = None, collection=None) -> dict:
    """Return the top-k chunks most similar to `query`, with source info.

    Args:
        query:      natural-language question (the only required input).
        k:          number of chunks to return; if None, chosen by suggest_k()
                    within the 4-8 range.
        collection: an open collection (defaults to the persisted one).

    Returns a dict:
        {
          "query": str,
          "k": int,
          "low_confidence": bool,   # True if even the best match is weak
          "results": [ {rank, similarity, text, source_file, source_title,
                        source_url, source_type, publisher}, ... ]
        }
    """
    if k is None:
        k = suggest_k(query)
    if collection is None:
        collection = get_collection()

    query_emb = _embed([query])

    res = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # query() returns lists-of-lists (one row per query); we sent one query.
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        similarity = 1.0 - dist  # cosine distance -> cosine similarity
        results.append(
            {
                "rank": rank,
                "similarity": round(similarity, 4),
                "text": doc,
                "source_file": meta.get("source_file", ""),
                "source_title": meta.get("source_title", ""),
                "source_url": meta.get("source_url", ""),
                "source_type": meta.get("source_type", ""),
                "publisher": meta.get("publisher", ""),
            }
        )

    low_confidence = (not results) or (results[0]["similarity"] < LOW_SIMILARITY_THRESHOLD)
    return {"query": query, "k": k, "low_confidence": low_confidence, "results": results}


if __name__ == "__main__":
    # Interactive retrieval: type a question, see the top-k chunks + sources.
    # Assumes the index already exists — run `python embedding.py` first.
    # Press Enter on an empty line (or type 'quit') to exit.
    print("Ask a GT housing question (empty line or 'quit' to exit).\n")

    # Open the collection once so we don't reload it for every question.
    coll = get_collection()

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        out = retrieve(query, collection=coll)
        print(f"\n  k={out['k']}  |  low_confidence={out['low_confidence']}")
        if out["low_confidence"]:
            print("  (Weak matches — the corpus may not cover this well.)")
        for r in out["results"]:
            preview = " ".join(r["text"].split())[:160]
            print(f"  [{r['rank']}] sim={r['similarity']:.3f}  "
                  f"{r['source_title']}")
            print(f"      {preview}")
        print()
