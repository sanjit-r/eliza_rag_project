"""
prompt.py — Prompt template construction for the SEC RAG system.

Final prompt design (V4):
  - Three-part output: Answer / Supporting Evidence / Gaps
  - Answer section: synthesis, no inline citations, 3-6 sentences
  - Supporting Evidence: bullet-point citations with optional short quotes
  - Gaps: explicit statement of what the retrieved excerpts don't cover

Prompt evolution is documented in docs/prompt_iterations.md.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a senior financial analyst specializing in SEC EDGAR regulatory filings \
(10-K annual reports and 10-Q quarterly reports). You interpret risk disclosures, \
financial statements, and management commentary to answer business questions with \
precision and analytical clarity.

When answering, you synthesize — you do not copy long passages. You are concise, \
specific, and you cite every material claim.\
"""

_USER_TEMPLATE = """\
The following excerpts are from SEC EDGAR filings. Each excerpt is labeled with its source.

{context}

---

Question: {question}

Respond in exactly three sections:

**Answer**
Directly address the question in 3–6 sentences. Synthesize and interpret — do not quote \
at length or list chunks verbatim. Use specific figures, dates, and named risks where \
the sources support them. Save inline citation clutter for the next section.

**Supporting Evidence**
For each key claim in the Answer, cite the source filing:
  [TICKER · FILING_TYPE · DATE · Item LABEL]
List as bullet points. Include a direct quote (≤20 words) only when the exact wording \
is materially significant (e.g., a specific warning, a defined term, a precise figure).

**Gaps**
List any part of the question the provided excerpts do not cover. Be specific \
(e.g., "JPM's cybersecurity risk is not present in the retrieved excerpts").
If all parts of the question are addressed, write "None."\
"""


def format_context(chunks: list[dict]) -> str:
    """
    Format a list of chunk dicts into labeled context blocks for the prompt.

    Each block looks like:
      [NVDA · 10-K · 2025-02-26 · Item 1A: Risk Factors]
      {chunk_text}
    """
    blocks: list[str] = []
    for chunk in chunks:
        ticker      = chunk.get("ticker", "?")
        filing_type = chunk.get("filing_type", "?")
        filing_date = chunk.get("filing_date", "?")
        item_id     = chunk.get("item_id")
        item_title  = chunk.get("item_title", "")
        text        = chunk.get("text", "")

        if item_id:
            item_label = f"Item {item_id}: {item_title}" if item_title else f"Item {item_id}"
        else:
            item_label = "General"

        header = f"[{ticker} · {filing_type} · {filing_date} · {item_label}]"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the full user message for the Claude API call."""
    context = format_context(chunks)
    return _USER_TEMPLATE.format(context=context, question=question)
