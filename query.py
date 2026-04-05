#!/usr/bin/env python3
"""
query.py — CLI interface for the SEC EDGAR RAG system.

Usage:
    python query.py "What are the primary risk factors facing Apple, Tesla, and JPMorgan?"

Options:
    --index-dir     Path to index directory (default: data/index)
    --model         Claude model ID (default: claude-sonnet-4-6)
    --show-sources  Print retrieved source list before the answer
    --k             Number of chunks to retrieve (default: 15)
"""

import argparse
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

logging.basicConfig(
    level=logging.WARNING,  # Quiet by default; use --verbose for more
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Query the SEC EDGAR RAG system"
    )
    parser.add_argument("question", help="The business question to answer")
    parser.add_argument(
        "--index-dir", default="data/index",
        help="Path to the index directory (default: data/index)"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude model ID (default: claude-sonnet-4-6)"
    )
    parser.add_argument(
        "--show-sources", action="store_true",
        help="Print the list of retrieved sources before the answer"
    )
    parser.add_argument(
        "--k", type=int, default=15,
        help="Number of chunks to retrieve (default: 15)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"Error: index directory '{index_dir}' not found.", file=sys.stderr)
        print("Run: python indexer.py --corpus-dir /path/to/edgar_corpus", file=sys.stderr)
        sys.exit(1)

    # Load index
    from rag.retriever import load_index, build_entity_lookup, build_sector_lookup, retrieve
    from rag.embedder import Embedder
    from rag.llm import generate_answer

    config_dir = Path(__file__).parent / "config"

    print("Loading index...", end=" ", flush=True)
    index, all_chunks, corpus_meta, corpus_latest_date = load_index(index_dir)
    print(f"done ({len(all_chunks):,} chunks)")

    entity_lookup = build_entity_lookup(corpus_meta, config_dir / "aliases.yaml")
    sectors, sector_keywords = build_sector_lookup(config_dir / "sectors.yaml")
    embedder = Embedder()

    # Retrieve
    print("Retrieving relevant excerpts...", end=" ", flush=True)
    chunks = retrieve(
        question=args.question,
        embedder=embedder,
        index=index,
        all_chunks=all_chunks,
        corpus_meta=corpus_meta,
        corpus_latest_date=corpus_latest_date,
        entity_lookup=entity_lookup,
        sectors=sectors,
        sector_keywords=sector_keywords,
        k=args.k,
    )
    print(f"done ({len(chunks)} chunks retrieved)")

    if not chunks:
        print("No relevant excerpts found. The index may be empty or the question may "
              "not match any indexed content.", file=sys.stderr)
        sys.exit(1)

    # Optionally show sources
    if args.show_sources:
        print("\n--- Retrieved Sources ---")
        for i, c in enumerate(chunks, 1):
            score = c.get("score", 0.0)
            ticker = c.get("ticker", "?")
            ft = c.get("filing_type", "?")
            fd = c.get("filing_date", "?")
            semantic = c.get("item_semantic", "?")
            part = c.get("part")
            part_str = f" Part {part}" if part else ""
            print(f"  {i:2}. [{ticker} {ft} {fd}{part_str}] {semantic}  (score: {score:.3f})")
        print()

    # Generate answer (single API call)
    print("Generating answer...\n")
    answer, usage = generate_answer(
        question=args.question,
        chunks=chunks,
        model=args.model,
    )

    print("=" * 70)
    print(answer)
    print("=" * 70)
    print(f"\n[Tokens used: {usage['input_tokens']} in / {usage['output_tokens']} out | "
          f"Model: {usage['model']}]")


if __name__ == "__main__":
    main()
