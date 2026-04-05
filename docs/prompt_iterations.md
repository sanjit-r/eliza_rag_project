# Prompt Iteration Log

This document records the evolution of the prompt template used for answer generation.
Each iteration describes what changed, what problem it addressed, and what remained unsolved.

---

## V1 — Naive baseline

```
Context:
{raw chunks concatenated, no labels}

Question: {question}
Answer:
```

**Problems observed:**
- Model hallucinated specific figures not present in the retrieved chunks (e.g., invented revenue numbers)
- No citations — impossible to verify claims against source filings
- No output structure — answers ranged from 2 sentences to 15 paragraphs unpredictably
- Multi-company questions blended all companies into a single undifferentiated answer
- Model used background training knowledge freely, defeating the purpose of retrieval

**Decision:** Need role definition, source attribution, and output structure.

---

## V2 — Role + citation requirement

```
You are a financial analyst. Use only the SEC filing excerpts below to answer the question.
Do not use outside knowledge.

[Chunk 1 | Source: NVDA 10-K 2025-02-26]
{text}

[Chunk 2 | Source: AAPL 10-K 2024-11-01]
{text}

...

Question: {question}
Answer, citing sources in the format [Company, Filing Type, Date]:
```

**Improvements:**
- Citations appeared in responses — traceability improved significantly
- Role framing ("financial analyst") improved analytical tone
- "Do not use outside knowledge" reduced but did not eliminate hallucination

**Problems:**
- Citation format was inconsistent — some answers used `[NVDA, 10-K, 2025]`, others used `[NVDA 10-K Feb 2025]`, others omitted Item information entirely
- Multi-company answers still mixed company content within the same paragraph — hard to compare
- Model would copy long verbatim passages (3-5 sentences) instead of synthesizing
- The "do not use outside knowledge" instruction caused over-refusal: model would sometimes say "I cannot answer" when filings gave partial information, rather than noting the gap

**Decision:** Add a structured output format and separate the "state gaps" instruction from "refuse to answer."

---

## V3 — Structured format + system/user separation

Split into a system prompt and a structured user message.

**System prompt:**
```
You are a senior financial analyst specializing in SEC EDGAR regulatory filings.
Your task is to answer business questions using the provided excerpts from SEC filings.
Cite each material claim with a source label.
```

**User message:**
```
Excerpts:

[NVDA · 10-K · 2025-02-26 · Item 1A: Risk Factors]
{text}

...

Question: {question}

Provide a structured answer:
1. Summary (2-3 sentences)
2. Per-company breakdown (one section per company)
3. Sources (formatted list of citations)
4. Missing information (what the excerpts don't cover)
```

**Improvements:**
- System/user separation allowed the system prompt to define the analyst persona persistently
- Structured 4-section output was more readable and demo-friendly
- "Missing information" section replaced the over-refusal behavior

**Problems:**
- For single-company questions, the "Per-company breakdown" header created unnecessary structure
- The 4-section schema was rigid — "Summary" and "Per-company breakdown" duplicated content
- Model still produced verbose answers with repeated claims across sections
- Citation list at the end was disconnected from the claims it supported — hard to audit
- Answers remained too extractive: the model would list chunk contents sequentially rather than synthesizing

**Decision:** Collapse to 3 sections; move citations into a Supporting Evidence section directly paired to the Answer; require synthesis over extraction.

---

## V4 — Final prompt (current)

**System prompt:**
```
You are a senior financial analyst specializing in SEC EDGAR regulatory filings
(10-K annual reports and 10-Q quarterly reports). You interpret risk disclosures,
financial statements, and management commentary to answer business questions with
precision and analytical clarity.

When answering, you synthesize — you do not copy long passages. You are concise,
specific, and you cite every material claim.
```

**User message:**
```
The following excerpts are from SEC EDGAR filings. Each excerpt is labeled with its source.

[NVDA · 10-K · 2025-02-26 · Item 1A: Risk Factors]
{chunk_text}

[AAPL · 10-K · 2024-11-01 · Item 1A: Risk Factors]
{chunk_text}

...

---

Question: {question}

Respond in exactly three sections:

**Answer**
Directly address the question in 3–6 sentences. Synthesize and interpret — do not
quote at length or list chunks verbatim. Use specific figures, dates, and named risks
where the sources support them. Save inline citation clutter for the next section.

**Supporting Evidence**
For each key claim in the Answer, cite the source filing:
  [TICKER · FILING_TYPE · DATE · Item LABEL]
List as bullet points. Include a direct quote (≤20 words) only when the exact wording
is materially significant.

**Gaps**
List any part of the question the provided excerpts do not cover. Be specific.
If all parts of the question are addressed, write "None."
```

**Key changes from V3:**
- Three clean sections (Answer / Supporting Evidence / Gaps) instead of four
- Answer section explicitly defers inline citations to Supporting Evidence — cleaner prose
- "Synthesize and interpret — do not quote at length" directly addresses the extractive tendency
- "Exactly three sections" instruction reduced format variability
- Gaps section now explicitly asks for specificity ("JPM's cybersecurity risk is not present in the retrieved excerpts") — turns a retrieval weakness into a transparent, demo-friendly feature

**Observed improvements:**
- Answers are concise (4-6 sentences) and genuinely analytical
- Multi-company comparisons organize naturally by theme or company without rigid schema
- Supporting Evidence section is auditable — each bullet maps directly to a claim
- Gaps section catches and surfaces low-coverage companies (e.g., Costco with 1 filing) gracefully
- No over-refusal observed in testing

---

## Prompt Template (Final)

See [rag/prompt.py](../rag/prompt.py) for the implementation.

The `SYSTEM_PROMPT` and `_USER_TEMPLATE` strings in that file are the canonical final versions.
