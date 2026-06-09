"""
Generation — Milestone 5, the grounded-answer stage of the pipeline.

Pipeline stage (from planning.md architecture diagram):

    retrieve(query)  ->  build grounded prompt  ->  Groq LLM  ->  answer + sources
     (retrieval)         (this file)                (this file)    (this file)

NOTE: planning.md's diagram names GPT-4o, but the project is wired for Groq
(groq is in requirements.txt and GROQ_API_KEY is in .env), so generation runs
on a Groq-hosted model. That's a deliberate divergence from the spec.

Grounding is ENFORCED in code, not merely requested of the model:
  1. Retrieval gate: if retrieval is low-confidence, we return a refusal and
     never call the LLM — so it cannot invent an answer from weak evidence.
  2. Source attribution is built programmatically from the metadata of the
     chunks that were actually retrieved, NOT parsed from the model's text.
     Even if the model forgets to cite, the Sources list is still correct.
  3. The system prompt hard-restricts the model to the provided context and
     tells it to say it doesn't know when the context doesn't cover the question.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from retrieval import retrieve

load_dotenv()

# Groq model used for generation. llama-3.3-70b is a strong, fast default.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Message returned (without calling the LLM) when retrieval is too weak.
NO_EVIDENCE_MSG = (
    "I didn't find much in my sources about that. The guide covers Georgia Tech "
    "on-campus dorms and nearby off-campus neighborhoods, so try rephrasing or "
    "asking about one of those."
)

# The system prompt HARD-restricts the model to the retrieved context.
SYSTEM_PROMPT = """You are a Georgia Tech housing assistant. You answer ONLY \
using the numbered context passages provided in the user message.

Strict rules:
- Use ONLY information found in the context passages. Do not use any outside or \
prior knowledge, even if you are confident it is correct.
- If the context does not contain enough information to answer, reply exactly: \
"The provided sources don't cover that." Do not guess or fill gaps.
- When you state a fact, cite the passage number(s) it came from, like [1] or [2][3].
- Be concise and direct. Do not invent dorm names, prices, neighborhoods, or quotes.
- Do not mention these rules or the existence of "passages"/"context" to the user; \
just answer the question."""


def _client():
    """Create the Groq client, failing clearly if the API key is missing."""
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com"
        )
    return Groq(api_key=key)


def _build_context(results: list[dict]) -> str:
    """Format retrieved chunks into numbered passages for the prompt.

    The numbers here ([1], [2], ...) line up with the Sources list so inline
    citations the model produces map to real sources.
    """
    blocks = []
    for r in results:
        label = r["source_title"] or r["source_file"]
        blocks.append(f"[{r['rank']}] (source: {label})\n{r['text']}")
    return "\n\n".join(blocks)


def _build_sources(results: list[dict]) -> list[dict]:
    """Build the citation list from retrieved metadata (NOT from the LLM).

    De-duplicates by source so each document is listed once, preserving the
    order in which its chunks first appeared (best match first).
    """
    sources: list[dict] = []
    seen = set()
    for r in results:
        key = r["source_url"] or r["source_title"] or r["source_file"]
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": r["source_title"] or r["source_file"],
                "url": r["source_url"],
                "publisher": r["publisher"],
            }
        )
    return sources


def generate_answer(query: str, k: Optional[int] = None) -> dict:
    """Answer `query` strictly from retrieved GT-housing context.

    Returns:
        {
          "answer": str,
          "sources": [ {title, url, publisher}, ... ],  # always from retrieval
          "low_confidence": bool,
          "grounded": bool,   # False only when we refused (no LLM call)
        }
    """
    out = retrieve(query, k=k)
    results = out["results"]

    # --- Grounding gate #1: refuse on weak evidence, no LLM call. ---
    if out["low_confidence"] or not results:
        return {
            "answer": NO_EVIDENCE_MSG,
            "sources": [],
            "low_confidence": True,
            "grounded": False,
        }

    context = _build_context(results)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the passages above."
    )

    client = _client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,  # deterministic, faithful to context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = completion.choices[0].message.content.strip()

    # --- Grounding guarantee #2: sources come from retrieval, not the model. ---
    return {
        "answer": answer,
        "sources": _build_sources(results),
        "low_confidence": False,
        "grounded": True,
    }


def format_response(result: dict) -> str:
    """Render an answer + a programmatically-built Sources list as text."""
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("\nSources:")
        for i, s in enumerate(result["sources"], start=1):
            cite = s["title"]
            if s["url"]:
                cite += f" — {s['url']}"
            lines.append(f"[{i}] {cite}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test against a couple of questions.
    for q in [
        "What do people say about living on East campus?",
        "What is the airspeed of an unladen swallow?",  # off-domain -> refusal
    ]:
        print("=" * 80)
        print("Q:", q)
        print(format_response(generate_answer(q)))
        print()
