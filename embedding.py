"""
Embedding + vector store — Milestone 4, stage 1 of the pipeline.

Pipeline stage (from planning.md architecture diagram):

    chunk_corpus()  ->  all-MiniLM-L6-v2  ->  ChromaDB (cosine)
     (ingestion)        (this file: embed)     (this file: store)

This module owns everything about turning chunks into a searchable vector store:
  - Loads chunks from the ingestion pipeline (chunk_text.chunk_corpus()).
  - Embeds each chunk with all-MiniLM-L6-v2 via sentence-transformers.
  - Stores vectors + source metadata in a persistent ChromaDB collection,
    configured for COSINE similarity.

retrieval.py imports _embed() and get_collection() from here so the query is
embedded with the exact same model the chunks were embedded with.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb

from chunk_text import chunk_corpus

# --------------------------------------------------------------------------- #
# Configuration (shared with retrieval.py)                                    #
# --------------------------------------------------------------------------- #
EMBED_MODEL = "all-MiniLM-L6-v2"          # named in planning.md Retrieval Approach
COLLECTION_NAME = "gt_housing"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")   # vectors saved here
CORPUS_DIR = str(Path(__file__).parent / "GT Housing Info")


# --------------------------------------------------------------------------- #
# Embedding model (lazily loaded, cached)                                     #
# --------------------------------------------------------------------------- #
_model = None


def _get_model():
    """Load all-MiniLM-L6-v2 once and reuse it (the load is the slow part)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts into L2-normalized vectors.

    normalize_embeddings=True pairs with the collection's cosine space so the
    distances ChromaDB returns map cleanly to 1 - cosine_similarity.
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.tolist()


# --------------------------------------------------------------------------- #
# Build / load the vector store                                               #
# --------------------------------------------------------------------------- #
def build_index(rebuild: bool = True) -> "chromadb.Collection":
    """Embed every chunk and store it in ChromaDB with its source metadata.

    Args:
        rebuild: if True, drop any existing collection and rebuild from scratch
                 (so re-runs don't pile up duplicate vectors).

    Returns the populated ChromaDB collection.
    """
    # PersistentClient writes the DB to disk so we don't re-embed every run.
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a fresh DB

    # hnsw:space=cosine -> ChromaDB ranks by cosine distance (1 - cosine_sim).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    records = chunk_corpus(CORPUS_DIR)
    if not records:
        raise RuntimeError(f"No chunks found in {CORPUS_DIR!r} — run ingestion first.")

    documents = [r["text"] for r in records]
    # Stable, unique id per chunk so rebuilds are deterministic.
    ids = [f"{r['source_file']}::{r['chunk_index']}" for r in records]
    # Only primitive values are allowed in ChromaDB metadata.
    metadatas = [
        {
            "source_file": r["source_file"],
            "source_type": r["source_type"],
            "source_title": r["source_title"],
            "source_url": r["source_url"],
            "publisher": r["publisher"],
            "chunk_index": r["chunk_index"],
        }
        for r in records
    ]

    embeddings = _embed(documents)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


def get_collection() -> "chromadb.Collection":
    """Open the existing persisted collection (build_index() must have run)."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    # Run this once (or whenever your chunks change) to (re)build the store.
    print(f"Embedding all chunks with {EMBED_MODEL} and storing in ChromaDB...")
    coll = build_index(rebuild=True)
    print(f"Done. Indexed {coll.count()} chunks at {PERSIST_DIR}")
