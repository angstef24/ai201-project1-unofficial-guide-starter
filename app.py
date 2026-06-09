"""
Gradio interface — Milestone 5, the user-facing layer of the pipeline.

Pipeline stage (from planning.md architecture diagram):

    user question  ->  generate_answer()  ->  answer + Sources displayed
     (this file)        (generation.py)        (this file)

This is a thin UI: it only collects the question, calls generate_answer(), and
renders the result. All grounding/citation logic lives in generation.py, so the
interface cannot accidentally weaken the guarantees (it never talks to the LLM
directly and never fabricates sources).

Run:  .venv/bin/python app.py   then open the printed local URL.
"""

from __future__ import annotations

import gradio as gr

from generation import generate_answer

TITLE = "The Unofficial Guide — Georgia Tech Housing"
DESCRIPTION = (
    "Ask about Georgia Tech on-campus dorms and nearby off-campus neighborhoods. "
    "Answers come **only** from the collected sources, and every answer lists the "
    "sources it drew from. If the sources don't cover your question, it will say so."
)


def _render(result: dict) -> str:
    """Turn a generate_answer() result into display markdown (answer + sources)."""
    parts = []
    if result["low_confidence"]:
        parts.append("⚠️ *Low confidence — the sources may not cover this well.*\n")
    parts.append(result["answer"])

    if result["sources"]:
        parts.append("\n\n**Sources**")
        for i, s in enumerate(result["sources"], start=1):
            if s["url"]:
                parts.append(f"{i}. [{s['title']}]({s['url']})")
            else:
                parts.append(f"{i}. {s['title']}")
    return "\n".join(parts)


def answer_question(question: str) -> str:
    """Gradio callback: question text in, rendered answer+sources out."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question."
    try:
        result = generate_answer(question)
    except Exception as exc:
        # Surface setup problems (e.g. missing GROQ_API_KEY) instead of crashing.
        return f"⚠️ {exc}"
    return _render(result)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. Which side of campus is quieter?",
        lines=2,
    )
    submit = gr.Button("Ask", variant="primary")
    output = gr.Markdown(label="Answer")

    submit.click(fn=answer_question, inputs=question, outputs=output)
    question.submit(fn=answer_question, inputs=question, outputs=output)

    gr.Examples(
        examples=[
            "Where should I live on campus if I want a quieter place?",
            "What neighborhood is near a MARTA station?",
            "Where is the cheapest off-campus housing?",
            "What do people say about living on East campus?",
        ],
        inputs=question,
    )


if __name__ == "__main__":
    demo.launch()
