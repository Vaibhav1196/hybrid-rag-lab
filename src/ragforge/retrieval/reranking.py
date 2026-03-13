from __future__ import annotations

from typing import Protocol

import numpy as np

from ragforge.core.schemas import RetrievalResult


class QueryDocumentScorer(Protocol):
    """Protocol for scoring query-document text pairs for reranking."""

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return one score per query-document pair."""


class CrossEncoderScorer:
    """Thin adapter around sentence-transformers CrossEncoder for reranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score query-document pairs with a cross-encoder."""
        if not pairs:
            return np.zeros((0,), dtype=np.float32)

        scores = self.model.predict(pairs)
        return np.asarray(scores, dtype=np.float32)


class RetrievalReranker:
    """Rerank an existing shortlist of retrieval results."""

    def __init__(self, scorer: QueryDocumentScorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results for a query and return the top-k items."""
        query = query.strip()
        if not query:
            return []
        if top_k <= 0:
            return []
        if not results:
            return []

        scores = np.asarray(
            self.scorer.score([(query, result.chunk.text) for result in results]),
            dtype=np.float32,
        ).reshape(-1)

        if scores.shape[0] != len(results):
            raise ValueError("Reranker requires exactly one score per retrieval result.")

        ranked_indices = sorted(
            range(len(results)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                chunk=results[index].chunk,
                score=float(scores[index]),
                source="hybrid_reranked",
            )
            for index in ranked_indices
        ]
