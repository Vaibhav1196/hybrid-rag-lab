from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ragforge.evaluation import evaluate_answer_heuristics, evaluate_answer_with_judge, load_answer_evaluation_samples
from ragforge.generation import ContextBuilder, ExtractiveFallbackLLM, OpenAICompatibleLLM, RAGPipeline
from ragforge.retrieval.pipeline import RerankedHybridPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI for answer-level RAG evaluation."""
    parser = argparse.ArgumentParser(description="Run answer-level evaluation for the RAG pipeline.")
    parser.add_argument("--eval-path", type=Path, required=True, help="Path to the JSONL answer eval dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory of .txt files.")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--llm-mode", choices=["fallback", "openai"], default="fallback")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge after heuristic evaluation.")
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

    llm = (
        OpenAICompatibleLLM(model_name=args.llm_model, base_url=args.llm_base_url)
        if args.llm_mode == "openai"
        else ExtractiveFallbackLLM()
    )
    if args.judge and args.llm_mode != "openai":
        raise ValueError("--judge currently requires --llm-mode openai.")

    rag_pipeline = RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(),
        llm=llm,
    )

    samples = load_answer_evaluation_samples(args.eval_path)

    for sample in samples:
        response = rag_pipeline.answer(query=sample.query, retrieval_top_k=args.retrieval_top_k)
        heuristic = evaluate_answer_heuristics(sample, response)
        print(
            json.dumps(
                {
                    "query_id": sample.query_id,
                    "heuristic": asdict(heuristic),
                    "answer": response.answer,
                },
                indent=2,
            )
        )

        if args.judge:
            judge_result = evaluate_answer_with_judge(llm, sample, response)
            print(json.dumps({"query_id": sample.query_id, "judge": asdict(judge_result)}, indent=2))


if __name__ == "__main__":
    main()
