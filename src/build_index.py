"""
Build index from document directory.

This script:
1. Scans data directory for documents
2. Parses all supported formats
3. Chunks documents
4. Builds hybrid index (BM25 + FAISS)
5. Saves index to disk

Usage:
    python build_index.py --data-dir ./data --index-dir ./indexes
"""

import argparse
from pathlib import Path
import time

from parsers import DocumentParserFactory
from chunking import chunk_documents
from hybrid_index import HybridRAGIndex
from config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from documents")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing source documents"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory to save index"
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="hierarchical",
        choices=["semantic", "hierarchical", "sliding_window"],
        help="Chunking strategy"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between chunks in characters"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories"
    )
    
    args = parser.parse_args()
    settings = get_settings()
    
    # Use settings defaults if not provided
    data_dir = args.data_dir or settings.data_dir
    index_dir = args.index_dir or settings.index_dir
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap or settings.chunk_overlap
    
    print("=" * 60)
    print("RAG INDEX BUILDER")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Index directory: {index_dir}")
    print(f"Chunk strategy: {args.chunk_strategy}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    # Step 1: Parse documents
    print("STEP 1: Parsing documents...")
    print("-" * 60)
    
    parser_factory = DocumentParserFactory()
    documents = parser_factory.parse_directory(data_dir, recursive=args.recursive)
    
    if not documents:
        print("\n❌ No documents found or parsed successfully!")
        print(f"   Check that {data_dir} contains supported files:")
        print("   - PDF (.pdf)")
        print("   - HTML (.html, .htm)")
        print("   - Word (.docx)")
        print("   - Text (.txt, .md)")
        print("   - Confluence JSON (.json)")
        return
    
    print(f"\n✓ Parsed {len(documents)} documents")
    parse_time = time.time() - start_time
    print(f"  Time: {parse_time:.2f}s")
    
    # Step 2: Chunk documents
    print("\n" + "=" * 60)
    print("STEP 2: Chunking documents...")
    print("-" * 60)
    
    chunk_start = time.time()
    chunks = chunk_documents(
        documents=documents,
        strategy=args.chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"  Average chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
    chunk_time = time.time() - chunk_start
    print(f"  Time: {chunk_time:.2f}s")
    
    # Step 3: Build index
    print("\n" + "=" * 60)
    print("STEP 3: Building hybrid index...")
    print("-" * 60)
    
    index_start = time.time()
    
    index = HybridRAGIndex()
    index.ingest_chunks(chunks)
    index.build_all_indexes()
    
    index_time = time.time() - index_start
    print(f"  Time: {index_time:.2f}s")
    
    # Step 4: Save index
    print("\n" + "=" * 60)
    print("STEP 4: Saving index...")
    print("-" * 60)
    
    save_start = time.time()
    index.save(index_dir)
    save_time = time.time() - save_start
    print(f"  Time: {save_time:.2f}s")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ INDEX BUILD COMPLETE!")
    print("=" * 60)
    print(f"Documents parsed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Index saved to: {index_dir}")
    print(f"\nTotal time: {total_time:.2f}s")
    print("=" * 60)
    
    # Show stats
    stats = index.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 Ready to run! Start the Chainlit app:")
    print(f"   chainlit run app.py -w")


if __name__ == "__main__":
    main()
