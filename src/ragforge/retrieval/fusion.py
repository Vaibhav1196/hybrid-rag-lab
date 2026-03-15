"""
This code implements Reciprocal Rank Fusion (RRF)
Common algorithm used in hybrid search systems to combine results from multiple retrievers (e.g., dense + BM25).

Instead of combining scores, it combines ranks, 
which makes it robust when different retrievers use different scoring scales.

This code:
1. Takes multiple ranked result lists
2. Computes a reciprocal rank score for each result
3. Aggregates scores per chunk
4. Sorts chunks by fused score
5. Returns the top results

RRF_score = Σ (1 / (k + rank_i))

rank_i is the rank each retriever

"""
from __future__ import annotations

from dataclasses import dataclass

from ragforge.core.schemas import RetrievalResult


#----------------------------------------------------------------------------------

@dataclass(slots=True)
class FusionResult:
    """Internal fused ranking entry used to aggregate retriever outputs."""
    result: RetrievalResult
    score: float


#----------------------------------------------------------------------------------

# This function combines multiple ranked lists.
# result_lists: list[list[RetrievalResult]] 
# example : [dense_results, bm25_results]
# top_k: int = 5
# k: int = 60 => This is the RRF constant
# The formula used later is : score = 1 / (k + rank)
# This keeps the ranking smooth and prevents top results from dominating too strongly.
# returns: list[RetrievalResult]
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

    # Initialize a dictionary to store the fused results.
    # Key: chunk_id
    # Value: FusionResult (contains the result and the fused score)
    fused_by_chunk_id: dict[str, FusionResult] = {}

    for results in result_lists:
        # Here we will loop through index starting from 1
        # rank = 1, 2, 3, ...
        for rank, result in enumerate(results, start=1):
            # This is the Reciprocal Rank Fusion formula.
            # Compute RRF score for this result
            fused_score = 1.0 / (k + rank)
            # get the chunk_id for the corresponding result
            chunk_id = result.chunk.chunk_id

            # Store the result if not already present
            if chunk_id not in fused_by_chunk_id:
                fused_by_chunk_id[chunk_id] = FusionResult(result=result, score=0.0)

            # If a chunk appears in multiple retrievers, its score adds up.
            fused_by_chunk_id[chunk_id].score += fused_score


    # We now have one entry per chunk.
    # So now we sort the fused results by score in descending order.
    # Sorting logic : 
    # - highest score first : -item.score
    # - tie breaker : chunk_id : item.result.chunk.chunk_id
    fused_results = sorted(
        fused_by_chunk_id.values(),
        key=lambda item: (-item.score, item.result.chunk.chunk_id),
    )[:top_k]


    # Convert back to RetrievalResult objects
    return [
        RetrievalResult(
            chunk=item.result.chunk,
            score=item.score,
            source="hybrid_rrf",
        )
        for item in fused_results
    ]


#----------------------------------------------------------------------------------