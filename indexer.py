#!/usr/bin/env python3
"""
indexer.py — Build the FAISS retrieval index from the SEC EDGAR corpus.

Usage:
    python indexer.py --corpus-dir /path/to/edgar_corpus --index-dir data/index

This is a one-time operation. Run it before using query.py.
Expected time: ~5-15 minutes on CPU for 246 files.

What this script does:
  1. Parse all .txt filings in the corpus directory into Chunks
  2. Embed all chunk texts using BAAI/bge-small-en-v1.5
  3. Build a FAISS IndexFlatIP (exact cosine similarity)
  4. Save: index.faiss, metadata.pkl, corpus_meta.json
  5. Compute corpus_meta (company coverage scores) for sector expansion
"""

import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

# Configure logging before imports that may log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")

from rag.chunker import chunk_filing
from rag.embedder import Embedder
from rag.retriever import build_index, save_index


def compute_corpus_meta(
    chunk_metadata: list[dict],
    sectors_path: Path,
    corpus_latest_date: str,
) -> dict:
    """
    Build corpus_meta.json: per-company info including coverage score.

    coverage_score = filing_count * recency_score
    recency_score = fraction of filings in the last 2 years (from corpus_latest_date)
    """
    import yaml
    from datetime import date

    # Load sector assignments
    sector_by_ticker: dict[str, str] = {}
    if sectors_path.exists():
        with open(sectors_path) as f:
            sector_data = yaml.safe_load(f)
        for sector, tickers in (sector_data.get("sectors") or {}).items():
            for t in (tickers or []):
                sector_by_ticker[t] = sector

    latest = date.fromisoformat(corpus_latest_date)
    two_years_ago = latest.replace(year=latest.year - 2)

    # Aggregate per ticker
    by_ticker: dict[str, dict] = defaultdict(lambda: {
        "company": "",
        "filing_dates": [],
        "recent_count": 0,
    })

    for chunk in chunk_metadata:
        ticker = chunk["ticker"]
        fd = chunk.get("filing_date", "")
        info = by_ticker[ticker]
        if not info["company"]:
            info["company"] = chunk.get("company", ticker)
        if fd and fd not in info["filing_dates"]:
            info["filing_dates"].append(fd)
            try:
                if date.fromisoformat(fd) >= two_years_ago:
                    info["recent_count"] += 1
            except ValueError:
                pass

    corpus_meta: dict = {}
    for ticker, info in by_ticker.items():
        filing_count = len(info["filing_dates"])
        recent_count = info["recent_count"]
        # Coverage score: total filings + bonus for recency
        coverage_score = filing_count + (recent_count * 0.5)
        corpus_meta[ticker] = {
            "company":        info["company"],
            "sector":         sector_by_ticker.get(ticker, "Unknown"),
            "filing_dates":   sorted(info["filing_dates"]),
            "filing_count":   filing_count,
            "coverage_score": round(coverage_score, 2),
        }

    return corpus_meta


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS retrieval index from SEC EDGAR corpus"
    )
    parser.add_argument(
        "--corpus-dir", required=True,
        help="Directory containing the .txt filing files"
    )
    parser.add_argument(
        "--index-dir", default="data/index",
        help="Output directory for index files (default: data/index)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Embedding batch size (default: 64)"
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    index_dir = Path(args.index_dir)

    if not corpus_dir.exists():
        logger.error("Corpus directory does not exist: %s", corpus_dir)
        sys.exit(1)

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", corpus_dir)
        sys.exit(1)

    logger.info("Found %d .txt files in %s", len(txt_files), corpus_dir)

    # --- Step 1: Chunk all files ---
    t0 = time.time()
    all_chunks: list[dict] = []
    fallback_count = 0
    zero_chunk_files: list[str] = []

    try:
        from tqdm import tqdm
        file_iter = tqdm(txt_files, desc="Parsing filings", unit="file")
    except ImportError:
        file_iter = txt_files
        logger.info("tqdm not available; progress bars disabled")

    for fp in file_iter:
        chunks = chunk_filing(fp)
        if not chunks:
            zero_chunk_files.append(fp.name)
            continue
        for c in chunks:
            chunk_dict = c.to_dict()
            all_chunks.append(chunk_dict)
            if c.parse_method == "fallback_recursive":
                fallback_count += 1

    parse_time = time.time() - t0
    logger.info(
        "Parsed %d chunks from %d files in %.1fs "
        "(fallback: %d chunks, zero-output files: %d)",
        len(all_chunks), len(txt_files) - len(zero_chunk_files),
        parse_time, fallback_count, len(zero_chunk_files)
    )

    if zero_chunk_files:
        logger.warning("Files with zero chunks: %s", zero_chunk_files)

    if not all_chunks:
        logger.error("No chunks produced. Aborting.")
        sys.exit(1)

    # --- Step 2: Embed all chunks ---
    logger.info("Embedding %d chunks (this may take 5-15 minutes on CPU) ...", len(all_chunks))
    embedder = Embedder()
    t0 = time.time()

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed(texts, batch_size=args.batch_size, show_progress=True)

    embed_time = time.time() - t0
    logger.info("Embedded %d chunks in %.1fs (%.1f chunks/sec)",
                len(all_chunks), embed_time, len(all_chunks) / embed_time)

    # Store embeddings in chunk dicts for retrieval sub-searches
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i]

    # --- Step 3: Build FAISS index ---
    logger.info("Building FAISS IndexFlatIP (%d chunks, %d dims) ...",
                len(all_chunks), embeddings.shape[1])
    index = build_index(embeddings)

    # --- Step 4: Compute corpus metadata ---
    corpus_latest_date = max(
        (c.get("filing_date", "") for c in all_chunks), default="2026-01-01"
    )
    logger.info("Corpus latest filing date: %s", corpus_latest_date)

    config_dir = Path(__file__).parent / "config"
    sectors_path = config_dir / "sectors.yaml"
    corpus_meta = compute_corpus_meta(all_chunks, sectors_path, corpus_latest_date)

    logger.info("Corpus covers %d companies", len(corpus_meta))

    # --- Step 5: Save everything ---
    save_index(index, all_chunks, corpus_meta, corpus_latest_date, index_dir)

    # Print summary
    item_semantics = defaultdict(int)
    filing_types = defaultdict(int)
    for c in all_chunks:
        item_semantics[c.get("item_semantic", "unknown")] += 1
        filing_types[c.get("filing_type", "?")] += 1

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total chunks:        {len(all_chunks):,}")
    print(f"  Companies indexed:   {len(corpus_meta)}")
    print(f"  Corpus date range:   {min(c.get('filing_date','') for c in all_chunks if c.get('filing_date'))} "
          f"→ {corpus_latest_date}")
    print(f"  Filing types:        {dict(filing_types)}")
    print(f"  Fallback chunks:     {fallback_count}")
    print(f"  Index saved to:      {index_dir.resolve()}")
    print(f"  Total time:          {parse_time + embed_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
