"""
RAG Talent Search Engine - Build Vector Index Script

Indexes resumes into a local ChromaDB collection using sentence-transformers.
Supports re-indexing with the --rebuild flag.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --rebuild
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, VECTOR_DB_PATH
from src.data_loader import load_raw_data, get_dataset_stats
from src.resume_parser import parse_all_resumes
from src.vector_store import build_index, index_exists, get_collection_stats


def main():
    parser = argparse.ArgumentParser(
        description="Build or update the ChromaDB vector index for RAG Talent Search."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the index from scratch, deleting existing embeddings."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to the resume JSON dataset."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("       RAG Talent Search Engine - Index Builder")
    print("=" * 60)

    # Check if index already exists
    if index_exists() and not args.rebuild:
        stats = get_collection_stats()
        print(f"\n[Index Check] Vector store already exists at: {VECTOR_DB_PATH}")
        print(f"Indexed documents count: {stats.get('document_count', 0)}")
        print("\nTo force re-embedding and overwrite the index, run:")
        print("  python scripts/build_index.py --rebuild")
        print("=" * 60)
        return

    # Load dataset
    print(f"\n1. Loading dataset from: {args.data_path or DATA_PATH}")
    start_time = time.perf_counter()
    try:
        raw_records = load_raw_data(args.data_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    stats = get_dataset_stats(raw_records)
    print(f"   Loaded: {stats['total_resumes']} records")
    print(f"   Avg text length: {stats['avg_text_length']:.0f} chars")

    # Parse resumes into normalized profiles
    print("\n2. Parsing resume text and entity annotations...")
    profiles = parse_all_resumes(raw_records)
    print(f"   Normalized {len(profiles)} candidate profiles")

    # Build vector index
    print(f"\n3. Generating embeddings & building persistent ChromaDB at:\n   {VECTOR_DB_PATH}")
    if args.rebuild:
        print("   (--rebuild flag passed: wiping existing collection)")

    doc_count = build_index(profiles, rebuild=args.rebuild)

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"[OK] Indexing completed successfully in {elapsed:.2f} seconds!")
    print(f"  Total candidates indexed: {doc_count}")
    print(f"  ChromaDB directory: {VECTOR_DB_PATH}")
    print(f"  Ready for semantic queries & Streamlit UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
