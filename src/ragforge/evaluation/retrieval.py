from __future__ import annotations

from typing import Protocol

from ragforge.core.schemas import RetrievalResult
from ragforge.evaluation.schemas import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
    RetrievalQueryResult,
    RetrievalSample,
)


class SearchPipeline(Protocol):
    """Protocol for retrieval pipelines that can be evaluated."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run retrieval for a query."""


def _first_relevant_rank(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    relevant_chunk_ids: set[str],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in relevant_chunk_ids:
            return rank
        if result.chunk.doc_id in relevant_doc_ids:
            return rank
    return None


def _matched_ids(
    results: list[RetrievalResult],
    relevant_doc_ids: set[str],
    relevant_chunk_ids: set[str],
) -> tuple[list[str], list[str]]:
    matched_chunk_ids: list[str] = []
    matched_doc_ids: list[str] = []

    for result in results:
        if result.chunk.chunk_id in relevant_chunk_ids:
            matched_chunk_ids.append(result.chunk.chunk_id)
        if result.chunk.doc_id in relevant_doc_ids:
            matched_doc_ids.append(result.chunk.doc_id)

    return matched_chunk_ids, matched_doc_ids


def evaluate_retrieval(
    pipeline: SearchPipeline,
    samples: list[RetrievalSample],
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate a retrieval pipeline on labeled queries."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if not samples:
        raise ValueError("At least one retrieval sample is required for evaluation.")

    query_results: list[RetrievalQueryResult] = []

    for sample in samples:
        query = sample.query.strip()
        if not query:
            raise ValueError(f"Retrieval sample {sample.query_id!r} has a blank query.")

        relevant_doc_ids = set(sample.relevant_doc_ids)
        relevant_chunk_ids = set(sample.relevant_chunk_ids)
        if not relevant_doc_ids and not relevant_chunk_ids:
            raise ValueError(
                f"Retrieval sample {sample.query_id!r} must define relevant_doc_ids or relevant_chunk_ids."
            )

        results = pipeline.search(query=query, top_k=top_k)
        first_relevant_rank = _first_relevant_rank(results, relevant_doc_ids, relevant_chunk_ids)
        matched_chunk_ids, matched_doc_ids = _matched_ids(results, relevant_doc_ids, relevant_chunk_ids)

        total_relevant = len(relevant_chunk_ids) + len(relevant_doc_ids)
        total_matches = len(set(matched_chunk_ids)) + len(set(matched_doc_ids))

        query_results.append(
            RetrievalQueryResult(
                query_id=sample.query_id,
                query=sample.query,
                matched=first_relevant_rank is not None,
                reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
                recall=0.0 if total_relevant == 0 else total_matches / total_relevant,
                matched_chunk_ids=matched_chunk_ids,
                matched_doc_ids=matched_doc_ids,
                returned_chunk_ids=[result.chunk.chunk_id for result in results],
                returned_doc_ids=[result.chunk.doc_id for result in results],
            )
        )

    metrics = RetrievalMetrics(
        queries_evaluated=len(query_results),
        top_k=top_k,
        hit_rate=sum(1.0 for result in query_results if result.matched) / len(query_results),
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in query_results) / len(query_results),
        recall_at_k=sum(result.recall for result in query_results) / len(query_results),
    )

    return RetrievalEvaluationReport(metrics=metrics, query_results=query_results)
