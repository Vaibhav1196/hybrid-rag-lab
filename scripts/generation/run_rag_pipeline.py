from __future__ import annotations

import argparse
from pathlib import Path

from ragforge.generation import ContextBuilder, ExtractiveFallbackLLM, OpenAICompatibleLLM, RAGPipeline
from ragforge.retrieval.pipeline import RerankedHybridPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI for end-to-end RAG generation."""
    parser = argparse.ArgumentParser(description="Run the end-to-end hybrid RAG pipeline.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of .txt files.")
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Shortlist size before generation.")
    parser.add_argument("--candidate-top-k", type=int, default=10, help="Hybrid shortlist size before reranking.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Dense embedding model name.")
    parser.add_argument(
        "--reranker-model-name",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal Rank Fusion constant.")
    parser.add_argument("--llm-mode", choices=["fallback", "openai"], default="fallback")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI-compatible model name.")
    parser.add_argument("--llm-base-url", default=None, help="Optional OpenAI-compatible base URL.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retrieval_pipeline = RerankedHybridPipeline.from_directory(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_name=args.model_name,
        rrf_k=args.rrf_k,
        reranker_model_name=args.reranker_model_name,
        candidate_top_k=args.candidate_top_k,
    )

    if args.llm_mode == "openai":
        llm = OpenAICompatibleLLM(
            model_name=args.llm_model,
            base_url=args.llm_base_url,
        )
    else:
        llm = ExtractiveFallbackLLM()

    pipeline = RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(),
        llm=llm,
    )
    response = pipeline.answer(query=args.query, retrieval_top_k=args.retrieval_top_k)

    print(f"Query: {response.query}")
    print(f"LLM Model: {response.llm_response.model}")
    print(f"Total Duration (ms): {response.trace.total_duration_ms:.2f}")
    print("-" * 80)
    print("Answer:")
    print(response.answer)
    print("-" * 80)
    print("Context:")
    print(response.context.prompt_context or "No context available.")
    print("-" * 80)
    print("Stage Timings:")
    for item in response.trace.stage_timings:
        print(f"{item.stage}: {item.duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
