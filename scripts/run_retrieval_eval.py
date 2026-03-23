from __future__ import annotations

import argparse
from pathlib import Path

from ragforge.evaluation import evaluate_retrieval, load_retrieval_samples
from ragforge.retrieval.pipeline import BM25Pipeline, DensePipeline, HybridPipeline, RerankedHybridPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI for evaluating retrieval pipelines on labeled queries."""
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on a labeled dataset.")
    parser.add_argument("--pipeline", choices=["bm25", "dense", "hybrid", "reranked"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of .txt files.")
    parser.add_argument("--eval-path", type=Path, required=True, help="Path to JSONL retrieval labels.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved items to evaluate.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Dense embedding model name.")
    parser.add_argument(
        "--reranker-model-name",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal Rank Fusion constant.")
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=10,
        help="Shortlist size to send into reranking when pipeline=reranked.",
    )
    return parser


def build_pipeline(args: argparse.Namespace) -> object:
    """Build the requested retrieval pipeline."""
    common = {
        "data_dir": args.data_dir,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }

    if args.pipeline == "bm25":
        return BM25Pipeline.from_directory(**common)
    if args.pipeline == "dense":
        return DensePipeline.from_directory(**common, model_name=args.model_name)
    if args.pipeline == "hybrid":
        return HybridPipeline.from_directory(**common, model_name=args.model_name, rrf_k=args.rrf_k)
    return RerankedHybridPipeline.from_directory(
        **common,
        model_name=args.model_name,
        rrf_k=args.rrf_k,
        reranker_model_name=args.reranker_model_name,
        candidate_top_k=args.candidate_top_k,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = build_pipeline(args)
    samples = load_retrieval_samples(args.eval_path)
    report = evaluate_retrieval(pipeline, samples=samples, top_k=args.top_k)

    print(f"Pipeline: {args.pipeline}")
    print(f"Queries evaluated: {report.metrics.queries_evaluated}")
    print(f"Top-k: {report.metrics.top_k}")
    print(f"Hit Rate@{report.metrics.top_k}: {report.metrics.hit_rate:.4f}")
    print(f"MRR@{report.metrics.top_k}: {report.metrics.mean_reciprocal_rank:.4f}")
    print(f"Recall@{report.metrics.top_k}: {report.metrics.recall_at_k:.4f}")
    print("-" * 80)

    for result in report.query_results:
        print(f"Query ID: {result.query_id}")
        print(f"Query: {result.query}")
        print(f"Matched: {result.matched}")
        print(f"Reciprocal Rank: {result.reciprocal_rank:.4f}")
        print(f"Recall: {result.recall:.4f}")
        print(f"Matched Doc IDs: {result.matched_doc_ids}")
        print(f"Matched Chunk IDs: {result.matched_chunk_ids}")
        print(f"Returned Doc IDs: {result.returned_doc_ids}")
        print(f"Returned Chunk IDs: {result.returned_chunk_ids}")
        print("-" * 80)


if __name__ == "__main__":
    main()
