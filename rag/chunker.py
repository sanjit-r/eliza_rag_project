"""
chunker.py — Parse SEC EDGAR .txt filings into metadata-rich Chunk objects.

Actual file structure (verified by inspection):
  1. Lines 1-8: Header (key: value fields, then ===== separator)
  2. First line after separator: XBRL blob (dense, no spaces, ~1000-20000 chars)
  3. Short lines (~100-500 chars): Cover page boilerplate
  4. Short pipe-delimited lines: TOC — "Part I", "Item 1. | Business | 4"
  5. Very long lines (>5000 chars): Body content with items concatenated inline,
     e.g. "Item 7. Management's Discussion...Item 7A. Market Risk..."
  6. Short pipe-delimited lines: Table rows interspersed with body content

Strategy:
  A. Parse header fields by name (no line number assumptions).
  B. Skip XBRL blob: first line after separator with < 5% space density.
  C. Parse TOC for item→part mapping (handles both "| Part I |" and "Part I" formats).
  D. Find inline ITEM sections in the joined body text using a non-anchored regex.
  E. Assign PART by matching body items to TOC order (handles 10-Q duplicate item numbers).
  F. Sub-chunk large sections; fallback to recursive splitting if no sections found.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

SEPARATOR_RE   = re.compile(r"^={5,}")
FIELD_RE       = re.compile(r"^([A-Za-z ]+):\s*(.+)$")
XBRL_MARKER_RE = re.compile(r"xbrli:|iso4217:|us-gaap:|xbrl", re.IGNORECASE)

# TOC PART line: "| Part I |", "PART I. |", "PartI", "| Part I – Financial information | Page"
# Use \s* (not \s+) to handle "PartI" (no space); don't anchor end.
TOC_PART_RE = re.compile(
    r"^\s*\|?\s*PART\s*(I{1,3}|IV)\b",
    re.IGNORECASE,
)

# Valid SEC item IDs: 1-9 with optional letter suffix, or 10-16 with optional letter
# This prevents matching table values like "60L" or "17" that are not SEC item numbers
_VALID_ITEM_ID = r"((?:1[0-6]|[1-9])[A-Z]?)"

# TOC ITEM line: "Item 1. | Business | 4", "| Item 1A. | Risk Factors | 9"
# Also handles leading pipe (CAT format). Requires a pipe somewhere on the line.
TOC_ITEM_RE = re.compile(
    r"^\s*\|?\s*Item\s+" + _VALID_ITEM_ID + r"\.?\s*\|",
    re.IGNORECASE,
)

# Inline ITEM in body text (NOT a TOC entry — no pipe after label)
# Matches: "Item 7. Management's Discussion..." but NOT "Item 1. | Business | 4"
INLINE_ITEM_RE = re.compile(
    r"Item\s+" + _VALID_ITEM_ID + r"\.?\s+(?!\s*\|)(.{0,150})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Item → semantic label mapping
# ---------------------------------------------------------------------------

_10K_ITEM_MAP: dict[str, str] = {
    "1":   "business",
    "1A":  "risk_factors",
    "1B":  "unresolved_staff_comments",
    "1C":  "cybersecurity",
    "2":   "properties",
    "3":   "legal_proceedings",
    "4":   "mine_safety",
    "5":   "equity_market_info",
    "6":   "selected_financial_data",
    "7":   "md_and_a",
    "7A":  "market_risk",
    "8":   "financial_statements",
    "9":   "accounting_changes",
    "9A":  "controls_procedures",
    "9B":  "other_information",
    "9C":  "foreign_jurisdiction",
    "10":  "directors_officers",
    "11":  "executive_compensation",
    "12":  "security_ownership",
    "13":  "related_transactions",
    "14":  "principal_accountant",
    "15":  "exhibits",
    "16":  "form_summary",
}

# 10-Q: PART is required because item numbers repeat across Part I and Part II
_10Q_ITEM_MAP: dict[tuple[str, str], str] = {
    ("I",  "1"):   "financial_statements",
    ("I",  "2"):   "md_and_a",
    ("I",  "3"):   "market_risk",
    ("I",  "4"):   "controls_procedures",
    ("II", "1"):   "legal_proceedings",
    ("II", "1A"):  "risk_factors",
    ("II", "2"):   "unregistered_sales",
    ("II", "3"):   "defaults",
    ("II", "4"):   "mine_safety",
    ("II", "5"):   "other_information",
    ("II", "6"):   "exhibits",
}


def _item_semantic(filing_type: str, part: Optional[str], item_id: Optional[str]) -> str:
    if item_id is None:
        return "unknown"
    item_id = item_id.upper()
    if filing_type == "10-K":
        return _10K_ITEM_MAP.get(item_id, "unknown")
    elif filing_type == "10-Q":
        if part is None:
            return "unknown"
        return _10Q_ITEM_MAP.get((part.upper(), item_id), "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    ticker: str
    company: str
    filing_type: str
    filing_date: str
    fiscal_year: int
    fiscal_quarter: Optional[int]
    part: Optional[str]
    item_id: Optional[str]
    item_semantic: str
    item_title: str
    chunk_index: int
    parse_method: str
    source_file: str

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_header(lines: list[str]) -> dict:
    meta: dict[str, str] = {}
    for line in lines[:30]:
        if SEPARATOR_RE.match(line.strip()):
            break
        m = FIELD_RE.match(line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            meta[key] = m.group(2).strip()
    return meta


def _find_separator_idx(lines: list[str]) -> int:
    for i, line in enumerate(lines[:30]):
        if SEPARATOR_RE.match(line.strip()):
            return i
    return 0


def _derive_fiscal_info(meta: dict, filename: str) -> tuple[int, Optional[int]]:
    filing_date = meta.get("filing_date", "")
    quarter_str = meta.get("quarter", "")
    fiscal_year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else 0
    fiscal_quarter: Optional[int] = None
    # From "Quarter" field
    m = re.search(r"Q(\d)", quarter_str, re.IGNORECASE)
    if m:
        fiscal_quarter = int(m.group(1))
    # Fallback from filename (e.g. NVDA_10Q_2023Q4_...)
    if fiscal_quarter is None:
        m = re.search(r"\d{4}Q(\d)", filename)
        if m:
            fiscal_quarter = int(m.group(1))
    return fiscal_year, fiscal_quarter


# ---------------------------------------------------------------------------
# XBRL skip + content extraction
# ---------------------------------------------------------------------------

def _get_content_lines(lines: list[str], sep_idx: int) -> list[str]:
    """
    Return lines after the separator, skipping the XBRL blob.
    XBRL blob = first long line (> 300 chars) with < 5% space density.
    """
    result: list[str] = []
    xbrl_skipped = False
    for line in lines[sep_idx + 1:]:
        if not xbrl_skipped:
            if len(line) > 300 and (line.count(" ") / max(len(line), 1)) < 0.05:
                xbrl_skipped = True
                continue  # skip XBRL blob
        result.append(line)
    return result


# ---------------------------------------------------------------------------
# TOC parsing → item→part assignment queue
# ---------------------------------------------------------------------------

def _build_toc_part_assignments(content_lines: list[str]) -> dict[str, deque]:
    """
    Parse the TOC to build an ordered {item_id → deque(parts)} mapping.
    The deque preserves order so that the N-th occurrence of item_id in the body
    gets the N-th part from the TOC (critical for 10-Q disambiguation).
    """
    current_part: Optional[str] = None
    assignments: dict[str, deque] = defaultdict(deque)

    for line in content_lines:
        if len(line) > 5000:
            break  # past TOC into body content

        pm = TOC_PART_RE.match(line.strip())
        if pm:
            current_part = pm.group(1).upper()
            # Some filings put PART and first ITEM on the same line (CAT format):
            # "Part I | Item 1. | Business | 1"
            im_inline = TOC_ITEM_RE.search(line)
            if im_inline and current_part:
                assignments[im_inline.group(1).upper()].append(current_part)
            continue

        im = TOC_ITEM_RE.match(line)
        if im:
            item_id = im.group(1).upper()
            if current_part:
                assignments[item_id].append(current_part)

    return assignments


# ---------------------------------------------------------------------------
# Body section extraction (inline item detection)
# ---------------------------------------------------------------------------

def _parse_sections(
    content_lines: list[str],
    filing_type: str,
) -> list[tuple[Optional[str], str, str, str]]:
    """
    Find item sections embedded inline in the body content.
    Returns list of (part, item_id, item_title, body_text) tuples.

    Steps:
    1. Build item→part assignment queue from TOC lines.
    2. Join body lines (from first long line onward) into one text.
    3. Find all inline ITEM markers using INLINE_ITEM_RE.
    4. Match each body item to the next TOC assignment (in order).
    5. Extract body_text as text between consecutive item markers.
    """
    # Step 1: build part assignment queue from TOC
    part_assignments = _build_toc_part_assignments(content_lines)

    # Step 2: find where body content starts (first long line > 500 chars that is
    # not the XBRL blob — XBRL was already stripped in _get_content_lines)
    body_start_idx = 0
    for i, line in enumerate(content_lines):
        if len(line) > 500:
            body_start_idx = i
            break

    body_lines = content_lines[body_start_idx:]
    full_text = "\n".join(body_lines)

    if not full_text.strip():
        return []

    # Step 3: find all inline item markers
    all_matches = list(INLINE_ITEM_RE.finditer(full_text))
    if not all_matches:
        return []

    # Step 4: filter inline matches
    valid_matches: list[re.Match] = []

    if part_assignments:
        # TOC-guided: accept only the N-th occurrence where N = TOC count.
        # This prevents cross-references from creating spurious section splits.
        expected_counts: dict[str, int] = {k: len(v) for k, v in part_assignments.items()}
        seen_counts: dict[str, int] = defaultdict(int)
        for m in all_matches:
            item_id = m.group(1).upper()
            expected = expected_counts.get(item_id, 0)
            if expected == 0:
                continue  # not in TOC → likely a cross-reference, skip
            if seen_counts[item_id] < expected:
                valid_matches.append(m)
                seen_counts[item_id] += 1
    else:
        # No TOC info (e.g. JNJ "PartI" format, XOM no-pipe format).
        # Fall back to first-occurrence heuristic: accept first mention of each item_id.
        seen_ids: set[str] = set()
        for m in all_matches:
            item_id = m.group(1).upper()
            if item_id not in seen_ids:
                valid_matches.append(m)
                seen_ids.add(item_id)

    if not valid_matches:
        return []

    # Step 5: capture pre-match text if significant
    # (e.g. JPM 10-Q where financial statements appear without "Item N." markers)
    pre_sections: list[tuple] = []
    pre_text = full_text[:valid_matches[0].start()].strip()
    if len(pre_text) > _CHUNK_TARGET_CHARS:
        pre_sections.append((None, None, "", pre_text))

    # Step 6: assign parts and extract section bodies
    queues: dict[str, deque] = {k: deque(v) for k, v in part_assignments.items()}
    sections: list[tuple[Optional[str], str, str, str]] = []

    for i, m in enumerate(valid_matches):
        item_id = m.group(1).upper()
        item_title = m.group(2).strip()[:80]

        # Pop the PART for this occurrence (preserves order)
        queue = queues.get(item_id, deque())
        part = queue.popleft() if queue else None
        if item_id in queues:
            queues[item_id] = queue

        # Body = text from this match to next valid match
        body_start = m.start()
        body_end = valid_matches[i + 1].start() if i + 1 < len(valid_matches) else len(full_text)
        body_text = full_text[body_start:body_end].strip()

        if len(body_text) < 50:
            continue

        sections.append((part, item_id, item_title, body_text))

    return pre_sections + sections


# ---------------------------------------------------------------------------
# Text sub-chunking
# ---------------------------------------------------------------------------

_CHUNK_TARGET_CHARS = 700 * 4   # ~700 tokens
_CHUNK_MAX_CHARS    = 800 * 4   # ~800 tokens
_CHUNK_OVERLAP_CHARS = 100 * 4  # ~100 tokens overlap


def _split_text(text: str) -> list[str]:
    """Split text into ~700-token chunks with 100-token overlap."""
    if len(text) <= _CHUNK_MAX_CHARS:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + _CHUNK_TARGET_CHARS, text_len)

        if end < text_len:
            for delimiter in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(delimiter, start + _CHUNK_TARGET_CHARS // 2, end)
                if pos > start + _CHUNK_TARGET_CHARS // 2:
                    end = pos + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break  # reached end — no more content

        next_start = end - _CHUNK_OVERLAP_CHARS
        start = max(next_start, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def chunk_filing(filepath: Path | str) -> list[Chunk]:
    """Parse a single SEC EDGAR .txt filing into a list of Chunk objects."""
    filepath = Path(filepath)
    filename  = filepath.name

    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("Cannot read %s: %s", filepath, e)
        return []

    lines = raw.splitlines()

    # ── Metadata ──
    meta        = _parse_header(lines)
    ticker      = meta.get("ticker") or _ticker_from_filename(filename)
    company     = meta.get("company") or ticker
    filing_type = _normalize_filing_type(meta.get("filing_type") or _ftype_from_filename(filename))
    filing_date = _normalize_date(meta.get("filing_date") or "")
    fiscal_year, fiscal_quarter = _derive_fiscal_info(meta, filename)

    fn_ticker = _ticker_from_filename(filename)
    if fn_ticker and fn_ticker != ticker:
        logger.warning("%s: filename ticker %s != header ticker %s; using header",
                       filename, fn_ticker, ticker)

    # ── Content lines (XBRL skipped) ──
    sep_idx       = _find_separator_idx(lines)
    content_lines = _get_content_lines(lines, sep_idx)

    # ── Parse sections ──
    sections = _parse_sections(content_lines, filing_type)

    # ── Build chunks ──
    chunks: list[Chunk] = []

    if sections:
        for part, item_id, item_title, body in sections:
            semantic = _item_semantic(filing_type, part, item_id)
            for idx, sub_text in enumerate(_split_text(body)):
                if _is_xbrl_leak(sub_text):
                    logger.warning("%s: XBRL leak in chunk; skipping", filename)
                    continue
                chunks.append(Chunk(
                    text=sub_text, ticker=ticker, company=company,
                    filing_type=filing_type, filing_date=filing_date,
                    fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
                    part=part, item_id=item_id, item_semantic=semantic,
                    item_title=item_title, chunk_index=idx,
                    parse_method="section_aware", source_file=filename,
                ))
    else:
        logger.warning("%s: no inline item sections found; using fallback_recursive", filename)
        full_text = "\n".join(content_lines).strip()
        for idx, sub_text in enumerate(_split_text(full_text)):
            if _is_xbrl_leak(sub_text):
                continue
            chunks.append(Chunk(
                text=sub_text, ticker=ticker, company=company,
                filing_type=filing_type, filing_date=filing_date,
                fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
                part=None, item_id=None, item_semantic="unknown",
                item_title="", chunk_index=idx,
                parse_method="fallback_recursive", source_file=filename,
            ))

    if not chunks:
        logger.warning("%s: produced 0 chunks", filename)

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticker_from_filename(filename: str) -> str:
    return filename.split("_")[0] if filename else ""


def _ftype_from_filename(filename: str) -> str:
    for part in filename.split("_"):
        if part in ("10K", "10Q"):
            return "10-" + part[2]
    return ""


def _normalize_filing_type(raw: str) -> str:
    if "10-K" in raw or "10K" in raw:
        return "10-K"
    if "10-Q" in raw or "10Q" in raw:
        return "10-Q"
    return raw.strip()


def _normalize_date(raw: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else ""


def _is_xbrl_leak(text: str) -> bool:
    return bool(XBRL_MARKER_RE.search(text[:300]))
