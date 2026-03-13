from __future__ import annotations

from dataclasses import dataclass

from ragforge.core.schemas import RetrievalResult


@dataclass(slots=True)
class FusionResult:
    """Internal fused ranking entry used to aggregate retriever outputs."""

    result: RetrievalResult
    score: float


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    top_k: int = 5,
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Fuse multiple ranked result lists with Reciprocal Rank Fusion.

    RRF combines ranks instead of raw scores, which makes it robust when
    underlying retrievers use different scoring scales.
    """
    if top_k <= 0:
        return []
    if k <= 0:
        raise ValueError("RRF constant k must be > 0.")

    fused_by_chunk_id: dict[str, FusionResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            fused_score = 1.0 / (k + rank)
            chunk_id = result.chunk.chunk_id

            if chunk_id not in fused_by_chunk_id:
                fused_by_chunk_id[chunk_id] = FusionResult(result=result, score=0.0)

            fused_by_chunk_id[chunk_id].score += fused_score

    fused_results = sorted(
        fused_by_chunk_id.values(),
        key=lambda item: (-item.score, item.result.chunk.chunk_id),
    )[:top_k]

    return [
        RetrievalResult(
            chunk=item.result.chunk,
            score=item.score,
            source="hybrid_rrf",
        )
        for item in fused_results
    ]
