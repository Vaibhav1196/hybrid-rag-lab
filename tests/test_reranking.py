from __future__ import annotations

import numpy as np

from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.retrieval.reranking import RetrievalReranker


class FakeScorer:
    def __init__(self, mapping: dict[tuple[str, str], float]) -> None:
        self.mapping = mapping

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray([self.mapping[pair] for pair in pairs], dtype=np.float32)


def make_result(chunk_id: str, text: str, score: float, source: str = "hybrid_rrf") -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"{chunk_id}-doc",
            text=text,
            metadata={},
        ),
        score=score,
        source=source,
    )


def test_reranker_reorders_candidates_by_cross_encoder_score() -> None:
    reranker = RetrievalReranker(
        FakeScorer(
            {
                ("python query", "capital city of france"): 0.1,
                ("python query", "python backend systems"): 0.9,
            }
        )
    )
    results = [
        make_result("c1", "capital city of france", 0.8),
        make_result("c2", "python backend systems", 0.7),
    ]

    reranked = reranker.rerank("python query", results, top_k=2)

    assert [result.chunk.chunk_id for result in reranked] == ["c2", "c1"]
    assert reranked[0].source == "hybrid_reranked"


def test_reranker_returns_empty_for_blank_query_or_non_positive_top_k() -> None:
    reranker = RetrievalReranker(FakeScorer({("query", "text"): 0.5}))
    results = [make_result("c1", "text", 0.8)]

    assert reranker.rerank("   ", results, top_k=1) == []
    assert reranker.rerank("query", results, top_k=0) == []
    assert reranker.rerank("query", results, top_k=-1) == []


def test_reranker_returns_empty_for_empty_candidates() -> None:
    reranker = RetrievalReranker(FakeScorer({}))

    assert reranker.rerank("query", [], top_k=3) == []
