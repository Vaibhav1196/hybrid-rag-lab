from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.retrieval.fusion import reciprocal_rank_fusion


def make_result(chunk_id: str, score: float, source: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"{chunk_id}-doc",
            text=f"text for {chunk_id}",
            metadata={},
        ),
        score=score,
        source=source,
    )


def test_rrf_promotes_overlapping_hits() -> None:
    sparse = [
        make_result("c1", 10.0, "bm25"),
        make_result("c2", 9.0, "bm25"),
    ]
    dense = [
        make_result("c2", 0.9, "dense"),
        make_result("c1", 0.8, "dense"),
    ]

    fused = reciprocal_rank_fusion([sparse, dense], top_k=2, k=60)

    assert len(fused) == 2
    assert fused[0].chunk.chunk_id == "c1"
    assert fused[1].chunk.chunk_id == "c2"
    assert fused[0].source == "hybrid_rrf"


def test_rrf_handles_one_sided_results() -> None:
    sparse = [make_result("c1", 10.0, "bm25")]
    dense: list[RetrievalResult] = []

    fused = reciprocal_rank_fusion([sparse, dense], top_k=2, k=60)

    assert [result.chunk.chunk_id for result in fused] == ["c1"]


def test_rrf_returns_empty_for_non_positive_top_k() -> None:
    sparse = [make_result("c1", 10.0, "bm25")]

    assert reciprocal_rank_fusion([sparse], top_k=0) == []
    assert reciprocal_rank_fusion([sparse], top_k=-1) == []


def test_rrf_rejects_invalid_rrf_constant() -> None:
    sparse = [make_result("c1", 10.0, "bm25")]

    try:
        reciprocal_rank_fusion([sparse], top_k=1, k=0)
        assert False, "Expected ValueError for invalid RRF constant"
    except ValueError as exc:
        assert "must be > 0" in str(exc)
