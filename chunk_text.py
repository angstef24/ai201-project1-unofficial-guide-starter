"""
chunk_text() — implements the chunking strategy from planning.md.

Strategy (from planning.md > Chunking Strategy):
  - Blog / long-form pages : split into 100-token chunks with 15-token overlap.
  - Reddit threads         : one chunk per comment (plus the original post as its
                             own chunk). No overlap — each comment is a complete thought.

Token counting uses the SAME tokenizer as the embedding model (all-MiniLM-L6-v2),
so "100 tokens" here means 100 tokens as the embedder will actually see them. If the
tokenizer can't be loaded (e.g. offline), it falls back to a whitespace word count and
prints a warning, so the function still runs.

  NOTE on chunk size: at 100 tokens, blog chunks sit comfortably under the
  all-MiniLM-L6-v2 256-token input limit, so every chunk is embedded in full.
  Smaller chunks make retrieval more precise (each chunk is one tight idea) at the
  cost of more chunks overall and the risk of splitting a thought across a boundary
  — the 15-token overlap is there to soften that boundary effect.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# Tokenizer of the embedding model named in planning.md.
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Defaults straight from the Chunking Strategy section.
DEFAULT_CHUNK_SIZE = 100
DEFAULT_OVERLAP = 15


# --------------------------------------------------------------------------- #
# Tokenizer (lazily loaded, cached)                                           #
# --------------------------------------------------------------------------- #
class _Tokenizer:
    """Thin wrapper that encodes/decodes with the embedding model's tokenizer.

    Falls back to a whitespace word tokenizer if transformers/the model files
    aren't available, so chunk_text() never hard-fails just to count tokens.
    """

    def __init__(self) -> None:
        self._hf = None
        self._tried = False

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from transformers import AutoTokenizer  # ships with sentence-transformers

            # Loads only the tokenizer files (fast, no model weights).
            self._hf = AutoTokenizer.from_pretrained(_EMBED_MODEL)
        except Exception as exc:  # offline, not installed, etc.
            print(
                f"[chunk_text] Could not load '{_EMBED_MODEL}' tokenizer "
                f"({exc.__class__.__name__}); falling back to word-count "
                f"approximation. Token counts will be rough."
            )
            self._hf = None

    def encode(self, text: str) -> List:
        """Return a list of token ids (HF) or words (fallback)."""
        self._ensure()
        if self._hf is not None:
            return self._hf.encode(text, add_special_tokens=False)
        return text.split()

    def encode_offsets(self, text: str):
        """Return char (start, end) spans per token, or None if unsupported.

        Lets us chunk by true token counts while slicing the ORIGINAL text, so
        casing and punctuation are preserved exactly (WordPiece decode would
        lowercase and re-space the text).
        """
        self._ensure()
        if self._hf is None or not getattr(self._hf, "is_fast", False):
            return None
        enc = self._hf(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["offset_mapping"]

    def decode(self, tokens: List) -> str:
        """Inverse of encode(): turn a slice of tokens back into text (fallback)."""
        self._ensure()
        if self._hf is not None:
            return self._hf.decode(tokens, skip_special_tokens=True).strip()
        return " ".join(tokens)


_TOKENIZER = _Tokenizer()


# --------------------------------------------------------------------------- #
# Source-type detection                                                       #
# --------------------------------------------------------------------------- #
def infer_source_type(text: str = "", filename: str = "") -> str:
    """Return 'reddit' or 'blog'.

    Looks at the filename first, then the '**Source type:**' header that the
    scraped files in GT Housing Info/ carry. Defaults to 'blog'.
    """
    name = filename.lower()
    if "reddit" in name:
        return "reddit"
    if "ratemydorm" in name or "review" in name:
        return "reviews"
    header = text[:400].lower()
    stype_val = header.split("source type:", 1)[1][:80] if "source type:" in header else ""
    if "reddit" in stype_val:
        return "reddit"
    if "review" in stype_val:
        return "reviews"
    return "blog"


# --------------------------------------------------------------------------- #
# Citation header handling                                                    #
# --------------------------------------------------------------------------- #
# Every scraped file opens with a citation block (H1 title + **Field:** lines)
# that ends at the first horizontal rule (`---`). That block is metadata, not
# content, so it must NOT be chunked/embedded. We strip it before chunking and
# parse its fields into metadata so sources can still be cited downstream.
_HR = re.compile(r"^\s*---\s*$")
_FIELD = re.compile(r"^\*\*([^:*]+):\*\*\s*(.+?)\s*$")


def split_header(text: str):
    """Return (header, body): everything up to the first '---', and the rest.

    If no '---' separator is found, the whole text is treated as body (header
    is empty) so the function is safe on text that has no citation block.
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if _HR.match(ln):
            header = "\n".join(lines[:i])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            return header, body
    return "", text


def parse_citation(header: str) -> dict:
    """Pull citation fields out of the header block into a metadata dict.

    Captures the H1 title and any '**Field:** value' lines (Source type,
    Publisher, Title, Author, Published, URL, ...). Keys are lowercased with
    spaces turned into underscores, e.g. 'source_type', 'url', 'publisher'.
    """
    meta: dict = {}
    for ln in header.splitlines():
        ln = ln.strip()
        if ln.startswith("# "):
            meta.setdefault("title", ln[2:].strip())
            continue
        m = _FIELD.match(ln)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            meta[key] = m.group(2).strip()
    return meta


# --------------------------------------------------------------------------- #
# The two chunking modes                                                      #
# --------------------------------------------------------------------------- #
def _chunk_blog(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding-window token chunks with overlap (for long-form / blog pages)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: List[str] = []

    # Preferred path: chunk by token count but slice the ORIGINAL text via the
    # tokenizer's char offsets, so the chunk reads exactly like the source.
    offsets = _TOKENIZER.encode_offsets(text)
    if offsets is not None:
        n = len(offsets)
        if n == 0:
            return []
        for start in range(0, n, step):
            window = offsets[start : start + chunk_size]
            char_start, char_end = window[0][0], window[-1][1]
            piece = text[char_start:char_end].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_size >= n:
                break
        return chunks

    # Fallback path (no fast tokenizer / offline): encode->slice->decode.
    tokens = _TOKENIZER.encode(text)
    if not tokens:
        return []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        piece = _TOKENIZER.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(tokens):
            break  # last window already reached the end
    return chunks


# A comment line starts with an author in bold, optionally preceded by one or
# more '>' reply markers — matches the format of the scraped Reddit files, e.g.
#   **cyberchief** (47 upvotes):
#   > **TheTrueThymeLord** (9): ...
_COMMENT_START = re.compile(r"^\s*>*\s*\*\*[^*\n]+\*\*")
_HEADING = re.compile(r"^\s*(#{1,6}\s|---\s*$)")


def _chunk_reddit(text: str) -> List[str]:
    """One chunk per comment, plus the original post as its own chunk.

    Parses the scraped Reddit markdown layout: a '## Original Post' section
    followed by a '## Comments' section in which each comment begins with a
    bold author name. No overlap — each comment is self-contained.
    """
    lines = text.splitlines()

    # Split the doc into the original-post region and the comments region.
    comments_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().lower().startswith("## comments")),
        None,
    )
    if comments_idx is None:
        # No recognizable comments section — treat the whole thing as one chunk.
        body = text.strip()
        return [body] if body else []

    post_lines = lines[:comments_idx]
    comment_lines = lines[comments_idx + 1 :]

    chunks: List[str] = []

    # 1) Original post (title + question) as a single chunk.
    post = "\n".join(post_lines).strip()
    if post:
        chunks.append(post)

    # 2) Each comment (top-level or nested reply) becomes its own chunk.
    current: List[str] = []
    for ln in comment_lines:
        if _HEADING.match(ln):
            continue  # skip stray headings / horizontal rules
        if _COMMENT_START.match(ln):
            if current:
                block = "\n".join(current).strip()
                if block:
                    chunks.append(block)
            current = [ln]
        elif current:
            current.append(ln)  # continuation of the current comment
    if current:
        block = "\n".join(current).strip()
        if block:
            chunks.append(block)

    return chunks


# A review section starts with an H2 heading, e.g. "## #1 — Glenn (based on 6 reviews)".
_SECTION = re.compile(r"^##\s+\S")


def _chunk_reviews(text: str) -> List[str]:
    """One chunk per review section (for review/ranking pages like RateMyDorm).

    Splits on '## ' H2 headings so each dorm's heading + review text becomes a
    single chunk (the dorm name stays attached to its review). Any intro text
    before the first heading becomes its own chunk. No overlap — like Reddit,
    each review is a self-contained unit.
    """
    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        block = "\n".join(current).strip()
        if block:
            chunks.append(block)

    for ln in text.splitlines():
        if _SECTION.match(ln):
            flush()
            current = [ln]
        else:
            current.append(ln)
    flush()

    return chunks


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    source_type: str = "blog",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split `text` into chunks according to the planning.md strategy.

    The leading citation header (title + Source type / Publisher / Published /
    URL, up to the first '---') is stripped first so it never lands in a chunk.

    Args:
        text:        The document text to chunk.
        source_type: 'blog' (100-token sliding window, 15-token overlap),
                     'reddit' (one chunk per comment, no overlap), or
                     'reviews' (one chunk per review section, no overlap).
        chunk_size:  Tokens per chunk for blogs. Ignored for reddit/reviews.
        overlap:     Token overlap between blog chunks. Ignored for reddit/reviews.

    Returns:
        A list of chunk strings.
    """
    if not text or not text.strip():
        return []

    # Drop the citation/metadata header; only the body gets chunked.
    _, text = split_header(text)
    if not text.strip():
        return []

    source_type = source_type.lower()
    if source_type == "reddit":
        return _chunk_reddit(text)
    if source_type == "reviews":
        return _chunk_reviews(text)
    if source_type == "blog":
        return _chunk_blog(text, chunk_size, overlap)
    raise ValueError(
        f"Unknown source_type: {source_type!r} (use 'blog', 'reddit', or 'reviews')"
    )


def chunk_corpus(folder: str = "GT Housing Info") -> List[dict]:
    """Convenience helper for Milestone 3 verification / Milestone 4 ingestion.

    Walks every .md file in `folder`, infers its source type, chunks it, and
    returns a list of chunk records carrying the metadata you'll want for
    citation later. The citation header is parsed (not chunked) and its title /
    publisher / URL are attached to every chunk from that file.

    Each record: {source_file, source_type, source_title, source_url,
                  publisher, chunk_index, n_tokens, text}.
    """
    records: List[dict] = []
    for path in sorted(Path(folder).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        stype = infer_source_type(text=raw, filename=path.name)
        header, _ = split_header(raw)
        cite = parse_citation(header)
        for i, chunk in enumerate(chunk_text(raw, source_type=stype)):
            records.append(
                {
                    "source_file": path.name,
                    "source_type": stype,
                    "source_title": cite.get("title", ""),
                    "source_url": cite.get("url", ""),
                    "publisher": cite.get("publisher", ""),
                    "chunk_index": i,
                    "n_tokens": len(_TOKENIZER.encode(chunk)),
                    "text": chunk,
                }
            )
    return records


if __name__ == "__main__":
    # Verification run: chunk the scraped corpus and print stats so you can
    # eyeball whether the strategy is producing useful pieces (Milestone 3).
    here = Path(__file__).parent
    recs = chunk_corpus(str(here / "GT Housing Info"))

    by_file: dict = {}
    for r in recs:
        by_file.setdefault(r["source_file"], []).append(r)

    print(f"\nTotal chunks across corpus: {len(recs)}\n")
    print(f"{'file':<48}{'type':<8}{'chunks':>7}{'avg tok':>9}")
    print("-" * 72)
    for fname, rs in by_file.items():
        avg = sum(r["n_tokens"] for r in rs) / len(rs)
        print(f"{fname:<48}{rs[0]['source_type']:<8}{len(rs):>7}{avg:>9.0f}")

    # Print 5 random example chunks (full text) so the output is inspectable.
    import random

    sample = random.sample(recs, min(5, len(recs)))
    print(f"\n========== 5 random example chunks ==========")
    for n, r in enumerate(sample, 1):
        print(
            f"\n--- [{n}] {r['source_file']}  "
            f"(type={r['source_type']}, chunk #{r['chunk_index']}, "
            f"{r['n_tokens']} tokens) ---\n{r['text']}"
        )
