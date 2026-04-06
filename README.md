# SEC EDGAR RAG System

A retrieval-augmented generation system that answers business questions about SEC financial filings using a single LLM API call.

## Overview

```
Question ──► Entity/intent detection
             │
             ▼
        FAISS index (246 SEC filings, ~33K chunks)
             │
             ▼
        Dynamic retrieval (date filter → filing-type filter → per-entity budget → ranking)
             │
             ▼
        Prompt construction (retrieved excerpts + question)
             │
             ▼
        Single Claude API call  ──► Structured answer
                                    (Answer / Supporting Evidence / Gaps)
```

**Key design choices:**
- Voyage AI embeddings (`voyage-3-lite`) — Anthropic's recommended provider, works on any machine
- FAISS exact cosine search — no server, no configuration
- Section-aware chunking — preserves PART/ITEM document structure
- 4-stage retrieval — entity detection, date filtering, temporal diversification, semantic ranking
- Single final LLM call — satisfies the assessment constraint

Full design rationale: [docs/design_decisions.md](docs/design_decisions.md)

---

## Prerequisites

- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com/))
- A Voyage AI API key ([get one here](https://dash.voyageai.com/))
- The SEC EDGAR corpus (246 `.txt` files)

---

## Setup

**1. Install Git LFS** (required to pull the pre-built index)
```bash
brew install git-lfs
git lfs install
```

Git LFS must be installed before cloning so that the large index files
(`index.faiss`, `metadata.pkl`) are downloaded correctly. If you cloned first,
run `git lfs pull` afterwards.

**2. Clone and install dependencies**
```bash
cd personal_rag_project
pip install -r requirements.txt
```

If `faiss-cpu` fails to install on macOS:
```bash
pip install faiss-cpu --no-cache-dir
```

**3. Configure your API keys**
```bash
cp .env.example .env
# Edit .env and add your Anthropic and Voyage AI API keys
```

The pre-built index is already included in the repository via Git LFS — no
indexing step is required. Skip straight to **Running a Query**.

---

### Rebuilding the index (optional)

Only needed if you want to re-embed the corpus or the corpus changes.
This takes ~10 minutes and costs ~$0.15 in Voyage AI embedding API calls.

Point `--corpus-dir` at the folder containing the `.txt` filings:
```bash
python indexer.py --corpus-dir /path/to/edgar_corpus
```

Expected output:
```
INDEX BUILD COMPLETE
  Total chunks:        33,362
  Companies indexed:   54
  Corpus date range:   2015-02-27 → 2026-01-30
  Filing types:        {'10-K': 19093, '10-Q': 14269}
  Index saved to:      data/index/
  Total time:          ~1800s
```

---

## Running a Query

**CLI:**
```bash
python query.py "What are the primary risk factors facing Apple, Tesla, and JPMorgan?"
```

**With source list:**
```bash
python query.py --show-sources "How has NVIDIA's revenue changed over the last two years?"
```

**Self-contained example:**
```bash
python example_request.py
```

This runs the first assessment example question and shows retrieved sources + the full answer.

---

## Example Output

**Question:** What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?

**Answer**

Apple's primary risks center on supply chain concentration (heavy reliance on a small number of manufacturers in Asia), intensifying regulatory scrutiny around the App Store, and macroeconomic sensitivity affecting consumer hardware spending. Tesla faces unique exposure to autonomous driving regulatory approval timelines, raw material supply constraints for battery production, and competitive pressure from both legacy automakers and Chinese EV manufacturers. JPMorgan's dominant risks are credit quality deterioration in a high-rate environment, evolving capital adequacy requirements under Basel III endgame, and litigation exposure across multiple regulatory jurisdictions.

**Supporting Evidence**

- [AAPL · 10-K · 2024-11-01 · Item 1A: Risk Factors] — "a significant portion of our revenue comes from a limited number of customers and we are subject to risks associated with concentration..."
- [TSLA · 10-K · 2025-01-30 · Item 1A: Risk Factors] — "our ability to grow depends on... regulatory approval of full self-driving capabilities"
- [JPM · 10-K · 2026-02-13 · Item 1A: Risk Factors] — references Basel III endgame capital requirements and potential impact on returns

**Gaps**

None. All three companies are well-represented in the retrieved excerpts.

---

## Project Structure

```
personal_rag_project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── indexer.py               # One-time: build FAISS index
├── query.py                 # CLI: retrieve + answer
├── example_request.py       # Self-contained demo
├── config/
│   ├── sectors.yaml         # Sector → ticker assignments + keyword triggers
│   └── aliases.yaml         # Common name variants (JPMorgan → JPM, etc.)
├── rag/
│   ├── chunker.py           # Parse filings → Chunk objects
│   ├── embedder.py          # Voyage AI voyage-3-lite wrapper
│   ├── retriever.py         # FAISS index + 4-stage dynamic retrieval
│   ├── prompt.py            # Prompt template (V4 final)
│   └── llm.py               # Single Claude API call
├── data/
│   └── index/               # Pre-built index, tracked via Git LFS
│       ├── index.faiss
│       ├── metadata.pkl
│       └── corpus_meta.json
└── docs/
    ├── prompt_iterations.md  # V1-V4 prompt evolution log
    ├── design_decisions.md   # Architecture rationale
    └── evaluation_notes.md   # Quality assessment
```

---

## Corpus Coverage

The index covers 246 SEC EDGAR filings from 54 companies:

| Sector | Companies |
|--------|-----------|
| Technology | AAPL, MSFT, NVDA, GOOG, META, AMZN, ADBE, CRM, CSCO, IBM, INTC, AMD, ORCL, NFLX |
| Financial | JPM, BAC, GS, MS, BLK, AXP, MA, V, BRK |
| Healthcare | JNJ, PFE, MRK, ABBV, LLY, TMO, UNH |
| Consumer | WMT, TGT, COST, KO, PEP, MCD, SBUX, NKE, DIS |
| Energy | XOM, CVX |
| Industrial | CAT, DE, BA, LMT, RTX, GE, UPS |
| Telecom | VZ, T, CMCSA |

Filing date range: 2015 (GE outlier) + 2022–2026. Note: the manifest description says "2023–2025" but the actual files include 2022 and 2026 filings.

---

## Retrieval Design

The retriever classifies each question along four dimensions before retrieving:

1. **Temporal scope** — `explicit_window` ("last two years") applies hard date filters; `point_in_time` ("most recent") restricts to latest filings; `temporal_vague` uses soft recency weighting
2. **Filing-type preference** — "annual report" prefers 10-K; "quarterly" prefers 10-Q
3. **Entity scope** — detects company names/tickers; expands sector keywords to member companies
4. **Topic focus** — "risk" boosts `risk_factors` chunks; "revenue" boosts `md_and_a`

For multi-company questions, a per-entity minimum budget guarantees each company gets representation. Global slots fill remaining context from the highest-scoring chunks across all companies.

---

## Assumptions and Limitations

- Token counts use a ~4 chars/token estimate; actual token counts depend on the Claude tokenizer
- Chunk sizes target 700 tokens (2800 chars) — validated informally; optimal size may vary by query type
- Recency decay rate (0.14/year) is a starting estimate; tune based on evaluation results
- Financial tables (Item 8) embed with lower quality than prose — numeric queries may miss specific figures
- Companies with only 1 filing (COST, MCD, BLK, etc.) have limited temporal coverage — the Gaps section will flag this

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/prompt_iterations.md](docs/prompt_iterations.md) | V1-V4 prompt evolution log with reasoning |
| [docs/design_decisions.md](docs/design_decisions.md) | Architecture choices with justification |
| [docs/evaluation_notes.md](docs/evaluation_notes.md) | 10-question evaluation plan and results |
