# Prompt Iteration Log

This document records the evolution of the prompt template used for answer generation.
Each iteration describes what changed, what problem it addressed, and what remained unsolved.

All examples use the same test question:
> "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"

---

## V1 — Naive baseline

```
Context:
{raw chunks concatenated, no labels}

Question: {question}
Answer:
```

**Example output (V1):**

> Apple, Tesla, and JPMorgan all face significant risk factors in today's macroeconomic environment.
> Apple's revenue, which reached $394 billion in fiscal 2024, is heavily dependent on iPhone sales,
> which account for approximately 52% of total revenue. Any slowdown in consumer spending could
> materially impact results. Tesla faces intensifying competition from both legacy automakers and
> Chinese EV manufacturers such as BYD, which surpassed Tesla in global EV sales in Q4 2023.
> JPMorgan, as the largest U.S. bank by assets ($3.9 trillion), faces credit risk from its $1.3
> trillion loan portfolio, particularly in commercial real estate. All three companies are exposed
> to rising interest rates, though in different ways.

**Problems observed:**
- "$394 billion", "52%", "$3.9 trillion", "$1.3 trillion" — none of these figures appeared in the
  retrieved chunks; the model drew on training knowledge and invented plausible-sounding numbers
- No citations — impossible to verify any claim against a specific filing
- No output structure — single undifferentiated paragraph blending all three companies
- Answer reads as general financial commentary, not grounded retrieval

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

**Example output (V2):**

> Apple faces risks related to supply chain concentration, with the company noting that "a
> significant portion of our revenue comes from a limited number of customers and we are subject
> to risks associated with concentration of manufacturing in Asia, primarily in China, India, and
> Vietnam. Any significant disruption in our supply chain could materially adversely affect our
> business, results of operations, and financial condition." [Apple, 10-K, 2024]. Tesla's primary
> risk factors include regulatory uncertainty around autonomous driving, with the company stating
> it faces risks from "our ability to grow depends on regulatory approval of full self-driving
> capabilities across jurisdictions, and any failure to obtain such approvals could limit our
> addressable market." [Tesla, 10-K, 2026]. JPMorgan highlights credit quality deterioration and
> Basel III endgame capital requirements as primary concerns [JPMorgan, 10-K, 2026].
> Apple's risks are concentrated in hardware supply chains, Tesla's in regulatory and technology
> approval timelines, and JPMorgan's in credit and regulatory capital.

**Improvements:**
- Citations appeared in responses — traceability improved significantly
- Role framing ("financial analyst") improved analytical tone
- "Do not use outside knowledge" reduced but did not eliminate hallucination

**Problems:**
- Citation format inconsistent — Apple cited as `[Apple, 10-K, 2024]`, Tesla as `[Tesla, 10-K, 2026]`
  with no Item label, no standardized ticker format
- Long verbatim passages (3-5 sentences quoted directly) instead of synthesis
- All three companies collapsed into a single paragraph — comparison is hard to read
- Final comparison sentence is tacked on and shallow; not integrated into the analysis

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

**Example output (V3):**

> **1. Summary**
> Apple, Tesla, and JPMorgan face materially different risk profiles. Apple's risks center on
> supply chain concentration and regulatory pressure on the App Store, Tesla's on regulatory
> approval timelines for autonomous driving and raw material supply, and JPMorgan's on credit
> quality and evolving capital adequacy requirements.
>
> **2. Per-company breakdown**
>
> *Apple:* Apple's primary risks include heavy reliance on a concentrated manufacturing base in
> Asia and potential disruption to its App Store business model from antitrust regulators in
> the EU and U.S. The company notes exposure to macroeconomic conditions affecting consumer
> discretionary spending on hardware.
>
> *Tesla:* Tesla faces execution risk around Full Self-Driving regulatory approval, which the
> company identifies as a key growth driver. It also faces intensifying competition from legacy
> automakers and price pressure from Chinese EV manufacturers affecting margins.
>
> *JPMorgan:* JPMorgan highlights credit quality deterioration in a high-rate environment,
> Basel III endgame capital requirements, and litigation exposure across multiple jurisdictions.
>
> **3. Sources**
> - [AAPL · 10-K · 2024-11-01 · Item 1A]
> - [TSLA · 10-K · 2026-01-29 · Item 1A]
> - [JPM · 10-K · 2026-02-13 · Item 1A]
>
> **4. Missing information**
> The excerpts do not include Tesla's most recent quarterly filings, which may contain updated
> competitive positioning data.

**Improvements:**
- System/user separation allowed the system prompt to define the analyst persona persistently
- Structured 4-section output was more readable and demo-friendly
- "Missing information" section replaced the over-refusal behavior
- Citation format now standardized with ticker, filing type, date, and Item label

**Problems:**
- Summary and per-company breakdown duplicate content — the Apple paragraph in section 2 restates
  the Apple sentence from section 1 almost verbatim
- "Per-company breakdown" header is unnecessary for single-company questions — the rigid schema
  doesn't adapt to query type
- Citations at the end are disconnected from the claims they support — no way to trace which
  citation backs which specific sentence in the analysis
- Section 2 is still extractive in places ("The company notes exposure to...") rather than
  interpretive

**Decision:** Collapse to 3 sections; move citations into a Supporting Evidence section directly
paired to the Answer; require synthesis over extraction.

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

**Example output (V4):**

> **Answer**
>
> Apple's primary risks center on supply chain concentration — heavy reliance on a small number
> of Asian manufacturers — and intensifying regulatory scrutiny of the App Store under EU and
> U.S. antitrust frameworks. Tesla faces a distinctive combination of autonomous driving regulatory
> approval risk, raw material supply constraints for battery production, and margin pressure from
> Chinese EV competitors. JPMorgan's dominant concerns are credit quality deterioration in a
> sustained high-rate environment, evolving Basel III endgame capital requirements, and litigation
> exposure across multiple regulatory jurisdictions. The three risk profiles are largely
> non-overlapping: Apple's are operational and regulatory, Tesla's are technological and
> competitive, and JPMorgan's are financial and compliance-driven.
>
> **Supporting Evidence**
>
> - [AAPL · 10-K · 2024-11-01 · Item 1A: Risk Factors] — "a significant portion of our revenue
>   comes from a limited number of customers... concentration of manufacturing in Asia"
> - [TSLA · 10-K · 2026-01-29 · Item 1A: Risk Factors] — regulatory approval of full
>   self-driving capabilities cited as a key growth dependency
> - [TSLA · 10-K · 2026-01-29 · Item 1A: Risk Factors] — raw material and battery supply
>   chain constraints discussed as a margin risk factor
> - [JPM · 10-K · 2026-02-13 · Item 1A: Risk Factors] — Basel III endgame capital requirements
>   and potential impact on returns on equity
>
> **Gaps**
>
> None. All three companies are well-represented in the retrieved excerpts across their most
> recent annual filings.

**Key changes from V3:**
- Three clean sections (Answer / Supporting Evidence / Gaps) instead of four
- Answer section explicitly defers inline citations to Supporting Evidence — cleaner prose
- "Synthesize and interpret — do not quote at length" directly addresses the extractive tendency
- "Exactly three sections" instruction reduced format variability
- Gaps section now explicitly asks for specificity — turns a retrieval weakness into a
  transparent, demo-friendly feature
- Citations are co-located with the claims they support, making the answer auditable

**Observed improvements:**
- Answers are concise (4-6 sentences) and genuinely analytical — interpret rather than extract
- Multi-company comparisons organize naturally by theme without rigid per-company scaffolding
- Supporting Evidence section is auditable — each bullet maps directly to a claim in the Answer
- Gaps section catches and surfaces low-coverage companies (e.g., Costco with 1 filing) gracefully
- No over-refusal observed in testing across all 10 evaluation questions

---

## Prompt Template (Final)

See [rag/prompt.py](../rag/prompt.py) for the implementation.

The `SYSTEM_PROMPT` and `_USER_TEMPLATE` strings in that file are the canonical final versions.
