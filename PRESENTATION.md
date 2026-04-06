# Interview Presentation Guide
## SEC EDGAR RAG System

---

## How to Run This Meeting

Treat this as a client demo, not a technical interview. Open your terminal and code side by side.
Lead with the business problem, show the system working, then go deep on design only when asked.

**Suggested flow:**
1. 2 min — Frame the problem
2. 3 min — Live demo (run `example_request.py`)
3. 10 min — Walk through the architecture
4. Open — Answer questions

---

## 1. Opening Frame (Non-Technical)

> "The goal was to build a system that lets you ask plain-English business questions about SEC filings
> and get a grounded, cited answer — not a hallucinated summary. The hard constraint was a single LLM
> API call for the final answer. Everything else — how you index, retrieve, and structure the prompt —
> was the design challenge."

**Why this is hard:**
- 246 filings, ~33,000 text chunks — you can't fit all of it in a prompt
- A naive "search and summarize" approach either hallucinates or gives stale data
- Multi-company questions are especially tricky — one company's filings tend to dominate search results

---

## 2. Live Demo

Run this before the meeting starts to confirm it works:
```bash
python example_request.py
```

**What to narrate while it runs:**
- "It's loading the pre-built index — 33,000 chunks from 246 filings across 54 companies"
- "Now it's retrieving — classifying the question, filtering by entity and date, ranking by relevance"
- "Single Claude API call... and here's the answer"

**Point out in the output:**
- The Answer section is synthesized — not copied from the filing
- Each claim in Supporting Evidence maps to a specific filing, date, and Item
- The Gaps section explicitly says what's missing — this is a feature, not a failure

---

## 3. Architecture Walkthrough

Walk through this flow, referencing the README diagram:

```
Question → Entity/intent detection
         → FAISS index (33K chunks)
         → 4-stage retrieval
         → Prompt construction
         → Single Claude API call → Answer / Evidence / Gaps
```

### The Index

- **246 filings** parsed into **33,362 chunks** (~600 tokens each)
- Each chunk knows what company it's from, what filing type, what date, and what section (Risk Factors, MD&A, etc.)
- Embeddings generated with **Voyage AI `voyage-3-lite`** — Anthropic's recommended provider, retrieval-optimized
- Stored in **FAISS** — two local files, no server, no configuration
- One-time cost: ~$0.15 to embed the whole corpus

### Chunking Strategy

- SEC filings have a standardized structure: PART I → PART II, with numbered Items (1A Risk Factors, Item 7 MD&A, etc.)
- Parse along those item boundaries first, then sub-chunk large sections
- Why this matters: a chunk that's entirely Risk Factors embeds differently than one mixing Risk Factors with financial tables
- **98.8% of files** parsed section-aware; 3 files (INTC, MS, MCD) used fallback due to non-standard formatting

### Retrieval — The 4-Stage Pipeline

This is the most important design decision. Simple top-K cosine similarity has a critical flaw for this use case:

> "If you ask about Apple, Tesla, and JPMorgan — JPMorgan's risk factor text happens to score highest
> semantically, so it fills 12 of 15 slots. Apple and Tesla barely appear in the context Claude sees."

The pipeline fixes this:

| Stage | What it does | Example |
|-------|-------------|---------|
| 1. Date filter | Hard-restrict by time window | "last two years" → only 2024+ filings |
| 2. Filing-type filter | Prefer 10-K or 10-Q | "annual report" → 10-K only |
| 3. Per-entity budget | Guarantee minimum slots per company | 3-company question → at least 3 chunks each |
| 4. Semantic ranking | Top cosine score within constraints | Highest relevance within the filtered pool |

**Ablation result** (run against Q1 and Q2):

| Config | Q1 result | Q2 result |
|--------|-----------|-----------|
| Naive top-15 | GOOG chunk intrudes; all 3 companies present but unbalanced | 8/15 chunks from stale 2022–2023 filings |
| Diversified pipeline | All 3 companies guaranteed; no off-topic company | 0 pre-2024 chunks; correct temporal window |

### The Prompt

Four iterations to get here — documented in `docs/prompt_iterations.md`.

**V1 problem:** Hallucinated figures. No citations. Unstructured.

**V4 solution:** Three-part output enforced by the prompt schema:
- **Answer** — synthesis only, no inline citations, 3-6 sentences
- **Supporting Evidence** — one bullet per claim, exact filing reference
- **Gaps** — explicit statement of what's missing

> "The Gaps section is intentional. It turns a retrieval limitation into a transparent feature.
> If Costco only has one filing in the corpus, Claude says that — it doesn't silently omit it."

---

## 4. Anticipated Questions

### Technical

**"Why FAISS and not a vector database like Pinecone?"**
> At 33K vectors, exact flat search takes ~1ms. A real vector database adds a server, configuration,
> and a signup for no meaningful performance gain at this scale. FAISS is two files you can commit to git.

**"Why 15 chunks? How did you choose that number?"**
> It's grounded in the multi-company budget math. For a 3-company question, the per-entity
> minimum is `max(2, 15 // (3+2)) = 3` slots per company — 9 guaranteed slots, leaving 6
> global slots for the highest-scoring content across all companies. Drop to 10 and the
> per-entity minimum shrinks to 2, which risks thin coverage for one company. Go to 20 and
> you start pulling in lower-scored chunks that are more likely to be boilerplate disclaimers
> or near-duplicates of content already in the prompt — noise without signal.
> At ~600 tokens per chunk, 15 chunks is ~9,000 tokens of context sent to Claude — well
> within the 200K window and cheap per query. It's a tunable parameter though;
> `query.py --k 20` works if a question genuinely spans more companies.

**"What happens if someone asks about a company not in the corpus?"**
> Entity detection won't find a ticker match. The question falls back to global semantic search.
> If there's genuinely nothing relevant, the Gaps section surfaces it. The system doesn't hallucinate
> a company that isn't there — it tells you coverage is missing.

**"How do you handle temporal questions like 'last two years'?"**
> The intent classifier parses the time window and hard-filters the candidate pool before any
> semantic search happens. "Last two years" from today = 2024 onward. Only chunks from filings
> in that date range are eligible. This is why the ablation for Q2 (NVDA) matters —
> naive retrieval pulled 8 stale 2022 chunks; the pipeline pulled zero.

**"Why Voyage AI for embeddings?"**
> It's Anthropic's recommended embedding provider — designed to complement Claude's context
> understanding. It's retrieval-optimized with asymmetric input types (document vs. query).
> 512 dimensions keeps the index small (~68MB). $0.02/1M tokens — the whole corpus cost $0.15 to embed.

**"What's the tradeoff with section-aware chunking vs. fixed-size sliding windows?"**
> Fixed windows are simpler but semantically noisy — a chunk might span a risk factor and a
> financial table, and the embedding tries to represent both. Section-aware chunks are semantically
> tight because everything in the chunk is about the same topic. The cost is parser complexity —
> 3 of 246 files had non-standard formatting and fell back to recursive chunking.

**"Could you add a reranker?"**
> Yes, and it would improve precision. A cross-encoder reranker (e.g., Cohere Rerank) would
> rescore the top-30 candidates and pick the best 15. The constraint here was a single LLM call
> for the answer — a reranker is a separate API call on the retrieval side, which is allowed.
> It's the most impactful near-term improvement.

---

### Non-Technical / Stakeholder

**"How do we know the answers are accurate?"**
> Two things: every claim in the answer is cited with a specific filing, date, and section.
> And the Gaps section explicitly flags when something isn't covered. The model is instructed
> to synthesize what's in the filings, not use background knowledge. You can trace any
> statement in the answer back to a source document.

**"What would it take to add new filings?"**
> Re-run the indexer pointing at the new files. It's additive — you can index new filings
> without rebuilding from scratch if you modify the indexer to append. The embedding cost
> is proportional to the new content only. For 10 new filings: ~minutes and a few cents.

**"Could this work for other types of documents, not just SEC filings?"**
> Yes — the chunker is the only SEC-specific piece. The PART/ITEM parser would be replaced
> with whatever structure your documents have (e.g., policy sections, contract clauses).
> The retrieval pipeline, prompt, and LLM layer are entirely document-agnostic.

**"What are the limitations I should know about?"**
> Three honest ones:
> 1. Financial tables (balance sheets, income statements) embed poorly — specific dollar figures
>    are harder to retrieve than prose descriptions of risk or strategy.
> 2. Companies with only one filing (Costco, some others) can't answer trend questions — the
>    Gaps section flags this.
> 3. The system retrieves what's most semantically similar, not what's most recent by default.
>    Explicit date language ("last two years", "most recent") triggers precise temporal filtering.

**"How would you scale this to thousands of companies?"**
> See the evolution section below — that's a natural next conversation.

---

## 5. How to Evolve This System

Use this when they ask about roadmap or scaling.

### Near-term (days)
- **Reranker** — add Cohere Rerank or a cross-encoder after the initial 30-candidate retrieval; improves precision without changing the architecture
- **Streaming responses** — stream Claude's output token by token for better UX in a web interface
- **Query expansion** — generate 2-3 query variants and merge retrieval results (improves recall for ambiguous questions)

### Medium-term (weeks)
- **Incremental indexing** — append new filings to the index without full rebuild; critical for staying current with quarterly filing cycles
- **Structured data extraction** — extract specific financial metrics (revenue, EPS, guidance) into a structured store alongside the vector index; enables precise numeric queries
- **Web interface** — wrap `query.py` in a FastAPI endpoint with a simple frontend; the CLI is already the hard part

### Longer-term (months)
- **Larger corpus** — extend beyond the 54 companies to the full S&P 500 or Russell 1000; the architecture scales linearly
- **Multi-modal** — ingest earnings call transcripts, investor presentations, press releases alongside filings
- **Agent layer** — allow multi-step reasoning (e.g., "compare Apple's 2024 risks to 2022 risks" requires two retrieval calls and a synthesis step); the current single-call constraint was an assessment requirement, not an architectural one

---

## 6. Key Numbers to Know

| Metric | Value |
|--------|-------|
| Total filings | 246 |
| Companies | 54 |
| Total chunks | 33,362 |
| Section-aware parse rate | 98.8% |
| Embedding model | voyage-3-lite (512 dims) |
| Index size | ~68MB |
| Indexing cost | ~$0.15 one-time |
| Retrieval budget | 15 chunks per query |
| Context sent to Claude | ~9,000 tokens |
| Claude model | claude-sonnet-4-6 |
| Filing date range | 2022–2026 (+ 2015 GE outlier) |

---

## 7. Files to Have Open

- `README.md` — for setup overview
- `rag/retriever.py` — if they ask about the pipeline internals
- `rag/prompt.py` — if they ask about the prompt template
- `docs/prompt_iterations.md` — for the prompt evolution story
- `docs/design_decisions.md` — for any "why did you choose X" question
- Terminal running `python example_request.py` — keep this ready
