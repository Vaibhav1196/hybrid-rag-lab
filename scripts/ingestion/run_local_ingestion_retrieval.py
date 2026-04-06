from __future__ import annotations

import argparse
from pathlib import Path

from ragforge.ingestion.loader import load_documents
from ragforge.retrieval.pipeline import BM25Pipeline, DensePipeline, HybridPipeline, RerankedHybridPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI for testing retrieval pipelines on explicit uploaded-style files."""
    parser = argparse.ArgumentParser(
        description="Run a retrieval pipeline on explicit TXT, PDF, and DOCX files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more input files or directories containing supported documents.",
    )
    parser.add_argument("--query", required=True, help="Search query to run.")
    parser.add_argument(
        "--pipeline",
        choices=["bm25", "dense", "hybrid", "reranked"],
        default="hybrid",
        help="Retrieval pipeline to build.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked results to return.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Dense embedding model name.")
    parser.add_argument(
        "--reranker-model-name",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=10, help="Hybrid shortlist size before reranking.")
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal Rank Fusion constant.")
    return parser


def build_pipeline(args: argparse.Namespace, documents: list) -> object:
    """Build the requested retrieval pipeline from loaded documents."""
    common = {
        "documents": documents,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }

    if args.pipeline == "bm25":
        return BM25Pipeline.from_documents(**common)
    if args.pipeline == "dense":
        return DensePipeline.from_documents(**common, model_name=args.model_name)
    if args.pipeline == "hybrid":
        return HybridPipeline.from_documents(**common, model_name=args.model_name, rrf_k=args.rrf_k)
    return RerankedHybridPipeline.from_documents(
        **common,
        model_name=args.model_name,
        rrf_k=args.rrf_k,
        reranker_model_name=args.reranker_model_name,
        candidate_top_k=args.candidate_top_k,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    documents = load_documents(args.input)
    pipeline = build_pipeline(args, documents)
    results = pipeline.search(query=args.query, top_k=args.top_k)

    print(f"Pipeline: {args.pipeline}")
    print(f"Loaded documents: {len(documents)}")
    print(f"Loaded files:")
    for document in documents:
        print(
            f"- {document.metadata['filename']} "
            f"(type={document.metadata['file_type']}, doc_id={document.doc_id})"
        )
    print("-" * 80)
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
