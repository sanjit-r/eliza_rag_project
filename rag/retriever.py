"""
retriever.py — FAISS index management and dynamic retrieval with a 4-stage pipeline.

Retrieval stages (in order):
  1. Date filtering  — hard filter for explicit time windows / point-in-time queries
  2. Filing-type filtering — prefer 10-K or 10-Q based on question language
  3. Per-entity diversification — guaranteed minimum slots per detected company
  4. Semantic ranking with topic-aware section boost

Query intent classification uses lightweight keyword matching (no LLM call).
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import yaml

from rag.embedder import Embedder

logger = logging.getLogger(__name__)

TOTAL_BUDGET = 15   # max chunks passed to Claude


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an exact inner-product FAISS index (cosine sim with L2-norm vecs)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, metadata: list[dict], corpus_meta: dict,
               corpus_latest_date: str, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    with open(path / "metadata.pkl", "wb") as f:
        pickle.dump({
            "chunks": metadata,
            "corpus_latest_date": corpus_latest_date,
        }, f)
    with open(path / "corpus_meta.json", "w") as f:
        json.dump(corpus_meta, f, indent=2)
    logger.info("Index saved to %s (%d chunks)", path, len(metadata))


def load_index(path: Path) -> tuple[faiss.Index, list[dict], dict, str]:
    """Returns (index, chunk_metadata_list, corpus_meta, corpus_latest_date)."""
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "metadata.pkl", "rb") as f:
        saved = pickle.load(f)
    chunks = saved["chunks"]
    corpus_latest_date = saved.get("corpus_latest_date", "2026-01-01")
    with open(path / "corpus_meta.json") as f:
        corpus_meta = json.load(f)
    logger.info("Index loaded: %d chunks, latest date %s", len(chunks), corpus_latest_date)
    return index, chunks, corpus_meta, corpus_latest_date


# ---------------------------------------------------------------------------
# Entity and sector detection
# ---------------------------------------------------------------------------

def build_entity_lookup(corpus_meta: dict, aliases_path: Path) -> dict[str, str]:
    """
    Build {name_variant → ticker} lookup from corpus_meta + aliases.yaml.
    """
    lookup: dict[str, str] = {}

    # From corpus metadata: exact company name and ticker
    for ticker, info in corpus_meta.items():
        lookup[ticker.upper()] = ticker
        lookup[ticker.lower()] = ticker
        company_name = info.get("company", "")
        if company_name:
            lookup[company_name] = ticker
            lookup[company_name.lower()] = ticker

    # From aliases.yaml: add common name variants
    if aliases_path.exists():
        with open(aliases_path) as f:
            alias_data = yaml.safe_load(f)
        for ticker, variants in (alias_data.get("aliases") or {}).items():
            for v in (variants or []):
                lookup[str(v)] = ticker
                lookup[str(v).lower()] = ticker

    return lookup


def detect_companies(question: str, entity_lookup: dict[str, str]) -> list[str]:
    """Return list of unique tickers mentioned in the question."""
    found: dict[str, bool] = {}

    # Sort by length descending so "JPMorgan Chase" is tested before "Morgan"
    for name, ticker in sorted(entity_lookup.items(), key=lambda x: -len(x[0])):
        if ticker in found:
            continue
        # Use word-boundary matching to avoid false positives from short names.
        # e.g. "MA" (Mastercard) must not match inside "primary".
        # Allow trailing possessive ('s) so "NVIDIA's" matches the "NVIDIA" alias.
        # Using word boundary before the name; after, either a word boundary or 's.
        pattern = r"\b" + re.escape(name) + r"(?:'s)?\b"
        if re.search(pattern, question, re.IGNORECASE):
            found[ticker] = True

    return list(found.keys())


def build_sector_lookup(sectors_path: Path) -> dict[str, list[str]]:
    """Return {sector_name → [ticker, ...]} and {keyword → sector_name}."""
    with open(sectors_path) as f:
        data = yaml.safe_load(f)
    sectors: dict[str, list[str]] = {}
    keywords: dict[str, str] = {}

    for sector, tickers in (data.get("sectors") or {}).items():
        sectors[sector] = [t for t in (tickers or [])]

    for sector, kws in (data.get("sector_keywords") or {}).items():
        for kw in (kws or []):
            keywords[kw.lower()] = sector

    return sectors, keywords


def detect_sector(question: str, sector_keywords: dict[str, str]) -> Optional[str]:
    """Return sector name if a sector keyword is found in the question."""
    q_lower = question.lower()
    for kw, sector in sector_keywords.items():
        if kw in q_lower:
            return sector
    return None


def select_sector_companies(
    sector_name: str,
    sectors: dict[str, list[str]],
    corpus_meta: dict,
    max_companies: int = 6,
) -> list[str]:
    """
    Select up to max_companies from a sector using corpus coverage score
    (filing_count × recency_score). Coverage-based = deterministic, no extra computation.
    """
    members = sectors.get(sector_name, [])
    # Only include companies that are actually in the corpus
    members = [t for t in members if t in corpus_meta]

    if len(members) <= max_companies:
        return members

    # Score by coverage: filing_count × recency_score (both stored in corpus_meta)
    def coverage_score(ticker: str) -> float:
        info = corpus_meta.get(ticker, {})
        return info.get("coverage_score", 0.0)

    return sorted(members, key=coverage_score, reverse=True)[:max_companies]


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

# Explicit time-window patterns
_WINDOW_PATTERNS = [
    re.compile(r"last\s+(\w+)\s+years?", re.IGNORECASE),
    re.compile(r"past\s+(\w+)\s+years?", re.IGNORECASE),
    re.compile(r"since\s+(\d{4})", re.IGNORECASE),
    re.compile(r"from\s+(\d{4})\s+to\s+(\d{4})", re.IGNORECASE),
    re.compile(r"(\d{4})\s+(?:through|to)\s+(\d{4})", re.IGNORECASE),
    re.compile(r"over\s+the\s+(?:last|past)\s+(\w+)\s+years?", re.IGNORECASE),
]

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_POINT_IN_TIME_WORDS = [
    "most recent", "latest", "current", "newest", "last annual", "last quarterly",
    "most recent annual", "most recent quarterly"
]
_VAGUE_TEMPORAL_WORDS = ["recent", "recently", "changed", "evolved", "grown", "declined",
                          "shifted", "trend", "over time", "historically", "progress"]

_ANNUAL_WORDS = ["annual report", "annual reports", "10-k", "10k", "year-end",
                 "fiscal year", "year over year", "year-over-year"]
_QUARTERLY_WORDS = ["quarterly", "quarter", "10-q", "10q", "q1", "q2", "q3",
                    "quarter-over-quarter", "quarter over quarter"]

_TOPIC_MAP = {
    "risk_factors":          ["risk", "risks", "risk factor", "risk factors", "threat", "uncertainty"],
    "md_and_a":              ["revenue", "revenues", "margin", "margins", "earnings", "profit",
                              "income", "financial performance", "operating", "results of operations",
                              "outlook", "guidance", "growth"],
    "financial_statements":  ["balance sheet", "cash flow", "cash position", "liquidity",
                              "debt", "capital expenditure", "capex", "assets", "liabilities"],
    "business":              ["strategy", "business model", "competitive", "market position",
                              "products", "services", "operations"],
    "legal_proceedings":     ["legal", "litigation", "lawsuit", "settlement", "regulatory",
                              "compliance", "investigation"],
    "risk_factors,legal_proceedings": ["regulatory risk", "regulatory risks", "regulation",
                                        "regulations", "enforcement"],
    "cybersecurity":         ["cybersecurity", "cyber", "data breach", "data security", "hacking"],
}


def classify_intent(question: str) -> dict:
    """
    Classify a question into temporal_scope, filing_type_pref, and topic_focus.
    Returns a dict with keys: temporal_scope, time_window, filing_type_pref, topic_focus.
    """
    q_lower = question.lower()
    result = {
        "temporal_scope": "unspecified",
        "time_window": None,            # (start_date_str, end_date_str) or None
        "filing_type_pref": None,       # "10-K", "10-Q", or None
        "topic_focus": None,            # item_semantic string or None
    }

    # --- Temporal scope ---
    # Check point-in-time first
    for phrase in _POINT_IN_TIME_WORDS:
        if phrase in q_lower:
            result["temporal_scope"] = "point_in_time"
            break

    # Check explicit window
    if result["temporal_scope"] == "unspecified":
        window = _parse_explicit_window(question)
        if window:
            result["temporal_scope"] = "explicit_window"
            result["time_window"] = window

    # Check vague temporal
    if result["temporal_scope"] == "unspecified":
        for phrase in _VAGUE_TEMPORAL_WORDS:
            if phrase in q_lower:
                result["temporal_scope"] = "temporal_vague"
                break

    # --- Filing type preference ---
    for phrase in _ANNUAL_WORDS:
        if phrase in q_lower:
            result["filing_type_pref"] = "10-K"
            break
    if result["filing_type_pref"] is None:
        for phrase in _QUARTERLY_WORDS:
            if phrase in q_lower:
                result["filing_type_pref"] = "10-Q"
                break

    # --- Topic focus ---
    for semantic, keywords in _TOPIC_MAP.items():
        for kw in keywords:
            if kw in q_lower:
                # Handle compound keys like "risk_factors,legal_proceedings"
                result["topic_focus"] = semantic.split(",")[0]
                break
        if result["topic_focus"]:
            break

    return result


def _parse_explicit_window(question: str) -> Optional[tuple[str, str]]:
    """
    Parse explicit time windows from question text.
    Returns (start_date_iso, end_date_iso) or None.
    Anchor: use today's date for relative phrases.
    """
    q_lower = question.lower()
    today = date.today()

    # "last N years" / "past N years" / "over the last N years"
    for pat in [
        re.compile(r"(?:last|past|over\s+the\s+(?:last|past))\s+(\w+)\s+years?", re.IGNORECASE)
    ]:
        m = pat.search(question)
        if m:
            n_str = m.group(1).lower()
            n = _NUM_WORDS.get(n_str, None)
            if n is None:
                try:
                    n = int(n_str)
                except ValueError:
                    continue
            start = date(today.year - n, today.month, today.day)
            return start.isoformat(), today.isoformat()

    # "since YYYY"
    m = re.search(r"since\s+(\d{4})", question, re.IGNORECASE)
    if m:
        year = int(m.group(1))
        return f"{year}-01-01", today.isoformat()

    # "from YYYY to YYYY"
    m = re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", question, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(2)}-12-31"

    # "YYYY through/to YYYY"
    m = re.search(r"(\d{4})\s+(?:through|to)\s+(\d{4})", question, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(2)}-12-31"

    return None


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def recency_weight(filing_date_str: str, corpus_latest_date_str: str) -> float:
    """
    Soft recency boost anchored to the latest filing date in the corpus.
    Used ONLY for temporal_vague and unspecified queries.
    Decay: 1.0 at age=0, 0.3 at age=5+ years.
    """
    try:
        filing = date.fromisoformat(filing_date_str)
        latest = date.fromisoformat(corpus_latest_date_str)
    except (ValueError, TypeError):
        return 1.0
    age_years = max(0.0, (latest - filing).days / 365.25)
    return max(0.3, 1.0 - age_years * 0.14)


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    embedder: Embedder,
    index: faiss.Index,
    all_chunks: list[dict],
    corpus_meta: dict,
    corpus_latest_date: str,
    entity_lookup: dict[str, str],
    sectors: dict[str, list[str]],
    sector_keywords: dict[str, str],
    k: int = TOTAL_BUDGET,
) -> list[dict]:
    """
    4-stage retrieval pipeline. Returns up to k chunks ordered by adjusted score.
    Each returned dict = chunk metadata + "score" key.
    """
    intent = classify_intent(question)
    temporal_scope = intent["temporal_scope"]
    time_window = intent["time_window"]
    filing_type_pref = intent["filing_type_pref"]
    topic_focus = intent["topic_focus"]

    detected_tickers = detect_companies(question, entity_lookup)
    detected_sector = detect_sector(question, sector_keywords)

    logger.debug("Intent: %s | tickers: %s | sector: %s | topic: %s",
                 temporal_scope, detected_tickers, detected_sector, topic_focus)

    # Expand sector if no explicit companies detected
    sector_expanded = False
    if not detected_tickers and detected_sector:
        detected_tickers = select_sector_companies(
            detected_sector, sectors, corpus_meta, max_companies=6
        )
        sector_expanded = True
        logger.debug("Sector expansion: %s → %s", detected_sector, detected_tickers)

    # --- Stage 1: Date filtering ---
    # For sector queries, restrict candidate pool to sector members so global slots
    # don't pull in unrelated companies.
    if sector_expanded and detected_tickers:
        candidate_pool = [c for c in all_chunks if c.get("ticker") in detected_tickers]
    else:
        candidate_pool = all_chunks
    if temporal_scope == "explicit_window" and time_window:
        start_str, end_str = time_window
        filtered = [c for c in candidate_pool
                    if start_str <= c.get("filing_date", "") <= end_str]
        if len(filtered) < 10:
            logger.warning(
                "Explicit window [%s, %s] yields only %d chunks; using filtered pool as-is",
                start_str, end_str, len(filtered)
            )
        candidate_pool = filtered if filtered else candidate_pool

    elif temporal_scope == "point_in_time":
        # Find most recent filing date per entity (or globally)
        if detected_tickers:
            candidate_pool = _filter_to_most_recent_per_entity(
                candidate_pool, detected_tickers, filing_type_pref
            )
        else:
            # Global most recent
            most_recent = max(
                (c.get("filing_date", "") for c in candidate_pool), default=""
            )
            candidate_pool = [c for c in candidate_pool
                              if c.get("filing_date", "") == most_recent]

    # --- Stage 2: Filing-type filtering ---
    if filing_type_pref:
        ft_filtered = [c for c in candidate_pool
                       if c.get("filing_type") == filing_type_pref]
        if len(ft_filtered) >= 5:
            candidate_pool = ft_filtered
        else:
            logger.debug("Filing-type filter %s yields < 5 chunks; allowing both types",
                         filing_type_pref)

    # --- Embed query once ---
    q_vec = embedder.embed_query(question)   # shape (1, dim)

    # --- Stage 3: Per-entity diversification ---
    if detected_tickers:
        selected = _diversified_retrieve(
            question=question,
            q_vec=q_vec,
            detected_tickers=detected_tickers,
            candidate_pool=candidate_pool,
            all_chunks=all_chunks,
            topic_focus=topic_focus,
            temporal_scope=temporal_scope,
            time_window=time_window,
            k=k,
        )
    else:
        # No entities: global semantic search over candidate pool
        selected = _semantic_search(q_vec, candidate_pool, top_k=k)

    # --- Stage 4: Section-type boost and final ranking ---
    if topic_focus:
        preferred_labels = set(topic_focus.split(","))
        for item in selected:
            if item.get("item_semantic") in preferred_labels:
                item["score"] = item.get("score", 0.0) * 1.10

    # Apply recency weighting for vague/unspecified queries
    if temporal_scope in ("temporal_vague", "unspecified"):
        for item in selected:
            w = recency_weight(item.get("filing_date", ""), corpus_latest_date)
            item["score"] = item.get("score", 0.0) * w

    # Sort by adjusted score descending
    selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Deduplicate by (source_file, chunk_index)
    seen: set[tuple] = set()
    final: list[dict] = []
    for item in selected:
        key = (item.get("source_file"), item.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            final.append(item)
        if len(final) >= k:
            break

    return final


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _semantic_search(
    q_vec: np.ndarray,
    pool: list[dict],
    top_k: int,
) -> list[dict]:
    """Run a FAISS search over a subset of chunks (pool)."""
    if not pool:
        return []

    texts = [c["text"] for c in pool]
    # Build a temporary index over this pool
    dim = q_vec.shape[1]
    tmp_index = faiss.IndexFlatIP(dim)

    # We need embeddings for the pool. We store them in chunk metadata under "embedding".
    embs = np.stack([c["embedding"] for c in pool]).astype(np.float32)
    tmp_index.add(embs)

    scores, indices = tmp_index.search(q_vec, min(top_k, len(pool)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(pool[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def _diversified_retrieve(
    question: str,
    q_vec: np.ndarray,
    detected_tickers: list[str],
    candidate_pool: list[dict],
    all_chunks: list[dict],
    topic_focus: Optional[str],
    temporal_scope: str,
    time_window: Optional[tuple[str, str]],
    k: int,
) -> list[dict]:
    """
    Allocate a minimum per-entity budget then fill remaining slots globally.
    """
    N = len(detected_tickers)
    per_entity_min = max(2, k // (N + 2))
    entity_slots = per_entity_min * N
    global_slots = k - entity_slots

    selected: list[dict] = []
    selected_keys: set[tuple] = set()

    preferred_labels = set(topic_focus.split(",")) if topic_focus else set()

    for ticker in detected_tickers:
        entity_pool = [c for c in candidate_pool if c.get("ticker") == ticker]

        if not entity_pool:
            logger.debug("No chunks for %s in candidate pool", ticker)
            continue

        # Apply topic filter if requested
        if preferred_labels:
            topic_filtered = [c for c in entity_pool
                              if c.get("item_semantic") in preferred_labels]
            if len(topic_filtered) >= per_entity_min:
                entity_pool = topic_filtered

        # For temporal questions: ensure date spread if possible
        if temporal_scope in ("explicit_window", "temporal_vague"):
            entity_pool = _spread_by_date(entity_pool, per_entity_min * 2)

        # Score and take top per_entity_min
        scored = _semantic_search(q_vec, entity_pool, top_k=per_entity_min * 2)
        for item in scored[:per_entity_min]:
            key = (item.get("source_file"), item.get("chunk_index"))
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)

    # Fill global slots from full candidate pool
    if global_slots > 0:
        global_scored = _semantic_search(q_vec, candidate_pool, top_k=global_slots * 3)
        for item in global_scored:
            key = (item.get("source_file"), item.get("chunk_index"))
            if key not in selected_keys and len(selected) < k:
                selected.append(item)
                selected_keys.add(key)

    return selected


def _spread_by_date(pool: list[dict], target: int) -> list[dict]:
    """
    Return up to target chunks from pool, preferring temporal spread.
    Groups by filing_date and interleaves from oldest and newest periods.
    Falls back to the full pool if only one date is available.
    """
    from collections import defaultdict
    by_date: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_date[c.get("filing_date", "")].append(c)

    dates = sorted(by_date.keys())
    if len(dates) <= 1:
        return pool  # only one date; caller logs limited coverage

    result: list[dict] = []
    left, right = 0, len(dates) - 1
    toggle = True
    while len(result) < target and left <= right:
        if toggle:
            date_key = dates[left]
            left += 1
        else:
            date_key = dates[right]
            right -= 1
        toggle = not toggle
        chunks_for_date = by_date[date_key]
        result.append(chunks_for_date[0])
        by_date[date_key] = chunks_for_date[1:]
        if not by_date[date_key]:
            # Remove exhausted date
            pass

    return result


def _filter_to_most_recent_per_entity(
    pool: list[dict],
    tickers: list[str],
    filing_type_pref: Optional[str],
) -> list[dict]:
    """
    For point-in-time queries: keep only chunks from each entity's most recent filing.
    """
    result: list[dict] = []
    for ticker in tickers:
        entity_chunks = [c for c in pool if c.get("ticker") == ticker]
        if filing_type_pref:
            typed = [c for c in entity_chunks if c.get("filing_type") == filing_type_pref]
            if len(typed) >= 3:
                entity_chunks = typed
        if not entity_chunks:
            continue
        most_recent_date = max(c.get("filing_date", "") for c in entity_chunks)
        result.extend(c for c in entity_chunks if c.get("filing_date") == most_recent_date)
    return result
