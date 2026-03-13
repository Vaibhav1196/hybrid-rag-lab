from __future__ import annotations

import argparse
from pathlib import Path

from ragforge.retrieval.pipeline import HybridPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI for querying the hybrid pipeline on local documents."""
    parser = argparse.ArgumentParser(description="Run hybrid retrieval on local text documents.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of .txt files.")
    parser.add_argument("--query", required=True, help="Search query to run against the hybrid index.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked results to return.")
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name to use for dense embeddings.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant. Higher values flatten rank differences.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = HybridPipeline.from_directory(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_name=args.model_name,
        rrf_k=args.rrf_k,
    )
    results = pipeline.search(query=args.query, top_k=args.top_k)

    print(f"Loaded documents: {len(pipeline.documents)}")
    print(f"Built chunks: {len(pipeline.chunks)}")
    print(f"Query: {args.query}")
    print(f"Dense model: {args.model_name}")
    print(f"RRF k: {args.rrf_k}")
    print("-" * 80)

    if not results:
        print("No retrieval results returned.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Fusion Score: {result.score:.4f}")
        print(f"Source: {result.source}")
        print(f"Document ID: {result.chunk.doc_id}")
        print(f"Chunk ID: {result.chunk.chunk_id}")
        print(f"Metadata: {result.chunk.metadata}")
        print(f"Text: {result.chunk.text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
