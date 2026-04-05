#!/usr/bin/env python3
"""
example_request.py — A self-contained example demonstrating the full RAG pipeline.

This script is ready to run after completing setup (see README.md).
It uses one of the sample questions from the assessment brief.

Usage:
    python example_request.py
"""

import logging
import os
import sys
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration ────────────────────────────────────────────────────────────

QUESTION = (
    "What are the primary risk factors facing Apple, Tesla, and JPMorgan, "
    "and how do they compare?"
)

INDEX_DIR  = Path("data/index")
CONFIG_DIR = Path("config")
MODEL      = "claude-sonnet-4-6"
TOP_K      = 15

# ─────────────────────────────────────────────────────────────────────────────


def run():
    if not INDEX_DIR.exists():
        print("ERROR: Index not found at 'data/index/'.")
        print("Please run:  python indexer.py --corpus-dir /path/to/edgar_corpus")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("Copy .env.example to .env and add your Anthropic API key.")
        sys.exit(1)

    from rag.retriever import (
        load_index, build_entity_lookup, build_sector_lookup, retrieve
    )
    from rag.embedder import Embedder
    from rag.llm import generate_answer

    # ── Load index ──
    print("Loading index ...", end=" ", flush=True)
    index, all_chunks, corpus_meta, corpus_latest_date = load_index(INDEX_DIR)
    print(f"done ({len(all_chunks):,} chunks across {len(corpus_meta)} companies)\n")

    # ── Build lookup tables ──
    entity_lookup = build_entity_lookup(corpus_meta, CONFIG_DIR / "aliases.yaml")
    sectors, sector_keywords = build_sector_lookup(CONFIG_DIR / "sectors.yaml")
    embedder = Embedder()

    # ── Retrieve ──
    print(f"Question: {QUESTION}\n")
    print("Retrieving relevant filing excerpts ...", end=" ", flush=True)
    chunks = retrieve(
        question=QUESTION,
        embedder=embedder,
        index=index,
        all_chunks=all_chunks,
        corpus_meta=corpus_meta,
        corpus_latest_date=corpus_latest_date,
        entity_lookup=entity_lookup,
        sectors=sectors,
        sector_keywords=sector_keywords,
        k=TOP_K,
    )
    print(f"done ({len(chunks)} chunks)\n")

    # ── Show retrieved sources ──
    print("Retrieved sources:")
    print("-" * 60)
    for i, c in enumerate(chunks, 1):
        ticker   = c.get("ticker", "?")
        ft       = c.get("filing_type", "?")
        fd       = c.get("filing_date", "?")
        semantic = c.get("item_semantic", "?")
        part     = c.get("part")
        part_str = f" Part {part}" if part else ""
        score    = c.get("score", 0.0)
        print(f"  {i:2}. {ticker} {ft} {fd}{part_str} — {semantic}  (score={score:.3f})")
    print()

    # ── Generate answer (single API call) ──
    print("Generating answer (single Claude API call) ...")
    print("=" * 70)

    answer, usage = generate_answer(question=QUESTION, chunks=chunks, model=MODEL)

    print(answer)
    print("=" * 70)
    print(f"\nTokens: {usage['input_tokens']} in / {usage['output_tokens']} out")
    print(f"Model:  {usage['model']}")


if __name__ == "__main__":
    run()
