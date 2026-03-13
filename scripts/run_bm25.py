from __future__ import annotations

import argparse
from pathlib import Path

from ragforge.retrieval.pipeline import BM25Pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a small CLI for querying the BM25 pipeline on real documents."""
    parser = argparse.ArgumentParser(description="Run BM25 retrieval on local text documents.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of .txt files.")
    parser.add_argument("--query", required=True, help="Search query to run against the chunk index.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked results to return.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = BM25Pipeline.from_directory(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    results = pipeline.search(query=args.query, top_k=args.top_k)

    print(f"Loaded documents: {len(pipeline.documents)}")
    print(f"Built chunks: {len(pipeline.chunks)}")
    print(f"Query: {args.query}")
    print("-" * 80)

    if not results:
        print("No retrieval results returned.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Document ID: {result.chunk.doc_id}")
        print(f"Chunk ID: {result.chunk.chunk_id}")
        print(f"Metadata: {result.chunk.metadata}")
        print(f"Text: {result.chunk.text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
