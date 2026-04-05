# Design Decisions

This document records the major architectural choices made in building the SEC EDGAR RAG system,
with justification and status tags for each decision.

Status tags:
- **[SUPPORTED]** — strongly supported by sample file inspection or the spec
- **[ASSUMPTION]** — reasonable for MVP; not yet measured at full scale
- **[VALIDATED]** — confirmed correct during implementation or evaluation

---

## 1. Embedding model: `text-embedding-3-small` (OpenAI API)

**Decision:** Use OpenAI's `text-embedding-3-small` API for embeddings rather than a local model.

**Justification:**
- Works on any machine regardless of OS or hardware — no PyTorch/MPS dependency
- 1536-dimensional embeddings; embeddings are L2-normalized by the API, so
  `IndexFlatIP` inner product gives exact cosine similarity without any extra normalization step
- No local model download or disk space required for model weights
- OpenAI handles query/document embedding symmetry internally — no query prefix needed
- Strong retrieval quality: `text-embedding-3-small` outperforms older `ada-002` on MTEB

**Trade-off considered:** A local model (e.g., `BAAI/bge-small-en-v1.5`) avoids a second API key
and runs offline, but introduces platform-specific PyTorch issues (MPS OOM on Intel Macs,
segfaults with certain torch builds). The OpenAI approach is more portable and reliable across
machines at the cost of an additional API key and per-token pricing on the indexing step.

**Cost note:** Indexing ~33K chunks of ~700 tokens each ≈ ~23M tokens.
`text-embedding-3-small` pricing is $0.02/1M tokens → ~$0.46 one-time indexing cost.
Query-time embedding is a single short string — negligible cost.

**Status:** [VALIDATED — works on macOS Intel and Apple Silicon without platform-specific config]

---

## 2. Vector store: FAISS `IndexFlatIP` with L2-normalized embeddings

**Decision:** Use an exact inner-product FAISS index with no approximation.

**Justification:**
- At ~33K chunks × 1536 dims × 4 bytes = ~200MB index — well within RAM
- Flat index gives exact nearest-neighbor results (no approximation error)
- No server, no Docker, no configuration: just two files (`index.faiss` + `metadata.pkl`)
- Cosine similarity is achieved by combining L2-normalization of embeddings with inner-product
  search — `IndexFlatIP` with normalized vectors is exactly cosine similarity
- FAISS's C++ backend is fast even for exact search at this scale (~1ms per query)

**Trade-off considered:** `IndexIVFFlat` (approximate) would be faster at 1M+ vectors.
Not needed here — flat exact search is appropriate for the corpus size.

**Status:** [SUPPORTED — FAISS is well-established; corpus size is within comfortable range]

---

## 3. Chunking: Section-aware with 700-token target, 100-token overlap

**Decision:** Parse filings into item-level sections first, then sub-chunk large sections.

**Justification:**
- SEC filings have a standardized PART/ITEM structure that provides strong semantic boundaries
- Chunking within a section (e.g., only Risk Factors content together) produces embeddings
  that are semantically tighter and retrieve more precisely than fixed-size sliding windows
- 700-token target (2800 chars) is large enough to contain 2-5 complete risk factor sub-items
  while staying below 1000 tokens where signal dilution becomes measurable
- 100-token overlap (400 chars) preserves cross-boundary context without excessive redundancy
- The section boundary itself (Item 1A: Risk Factors) is preserved in chunk metadata,
  enabling downstream section-type filtering in retrieval

**Key parser choices:**
- PART detection uses a fixed regex (`I{1,3}|IV`) to avoid character-class ambiguity
- `in_toc` state flag prevents TOC lines from being mistaken for section bodies
- 10-Q items are disambiguated by `(filing_type, part, item_id)` because Item 1 means
  "Financial Statements" in Part I but "Legal Proceedings" in Part II
- XBRL blob skipped by scanning for first PART/TOC/ITEM marker (not a line number)

**Status:** [ASSUMPTION — 700-token target based on common RAG practice; validate chunk sizes
during parser robustness check]

---

## 4. LLM: `claude-sonnet-4-6`, single API call

**Decision:** Use Claude Sonnet 4.6 for answer generation in a single `messages.create()` call.

**Justification:**
- Satisfies the assessment constraint: "the final answer must come from a single LLM call"
- Claude Sonnet 4.6 offers strong analytical reasoning for financial text
- 200K context window easily accommodates 15 × ~800-token chunks (~12K tokens of context)
  plus the system prompt and question (~1K tokens) — total well under 20K tokens
- The three-part output structure (Answer / Supporting Evidence / Gaps) is reliably
  produced by Sonnet-class models with clear schema instructions

**Trade-off considered:** Claude Haiku 4.5 is faster and cheaper; Opus 4.6 is stronger.
Sonnet 4.6 is the right balance for an analytical task with well-structured retrieved context.

**Status:** [SUPPORTED — single-call constraint is from the spec; context math is exact]

---

## 5. Retrieval: 4-stage dynamic pipeline

**Decision:** Use a structured 4-stage retrieval pipeline rather than a simple top-k search.

**Stages:**
1. **Date filtering** — hard filter for explicit time windows ("last two years")
2. **Filing-type filtering** — prefer 10-K or 10-Q based on question language
3. **Per-entity diversification** — guaranteed minimum chunk slots per detected company
4. **Semantic ranking with topic boost** — cosine similarity + section-type boost

**Justification:**
- Simple top-k cosine similarity fails on multi-company questions: the semantically
  dominant company tends to fill all 15 slots, leaving others absent from context
- Hard date filtering for explicit time windows prevents "last two years" from surfacing
  2022 filings when 2024-2026 filings exist (the NVDA example)
- Recency weighting (soft decay) is reserved for vague temporal language only
- Temporal anchor uses `max(corpus filing dates)` rather than `date.today()` for
  reproducibility — the same corpus always produces the same weights
- Section-type boost (+10%) gently prioritizes Risk Factors chunks for risk questions
  without overriding strongly-scored chunks from other sections

**Trade-off considered:** A two-pass LLM reranker would give better relevance but adds
a second LLM call (violates the single-call constraint for the answer, and adds latency).
The current pipeline achieves reasonable diversity without any additional API calls.

**Status:** [ASSUMPTION — dynamic budget allocations (per_entity_min, global_slots) are
reasonable defaults; should be tested on all 10 evaluation questions]

---

## 6. Sector expansion: coverage-based company selection

**Decision:** For sector queries, select up to 6 companies by corpus coverage score
(filing_count × recency_score), computed at index build time.

**Justification:**
- Coverage-based selection is deterministic and requires no additional computation at
  query time — the coverage scores are precomputed and stored in `corpus_meta.json`
- Companies with more filings have richer retrieval signal; selecting them produces
  better-supported answers for demo stability
- The alternative (semantic centroid ranking) requires embedding company descriptions
  or centroid vectors, adding indexing complexity for marginal gain given that the
  sector sizes in this corpus are small (Healthcare: 7 companies, Tech: 14)
- For sectors with ≤6 members (Energy: XOM, CVX), all companies are always included
  regardless of coverage score

**Optional enhancement:** At retrieval time, for each sector company, run a FAISS sub-query
and select the 6 with the highest best-chunk cosine score. This is semantically better
but adds one FAISS scan per candidate company per query.

**Status:** [ASSUMPTION — coverage-based selection is pragmatic; the optional semantic
selection may produce noticeably better results for heterogeneous sectors like Healthcare]

---

## Corpus Date Range Note

The manifest description claims "2023-2025" but the actual corpus includes:
- 2022 filings: AAPL, AMZN, DIS, GOOG, JNJ, KO, MSFT, NVDA, PFE, TSLA, UNH, XOM
- 2026 filings: Several companies (filed for FY2025)
- 2015 outlier: GE_10K_2015-02-27

**Policy:** All 246 files are indexed. The 2015 GE outlier is self-suppressing under
recency weighting for vague temporal queries, and the Gaps section will surface its
limited coverage for any GE question asking about "recent" strategy.

**[SUPPORTED]** — confirmed by inspecting the manifest.json filing dates.
