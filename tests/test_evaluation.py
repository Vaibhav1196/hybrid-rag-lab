from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.evaluation import evaluate_retrieval, load_retrieval_samples
from ragforge.evaluation.schemas import RetrievalSample


class FakePipeline:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results_by_query.get(query, [])[:top_k]


def make_result(chunk_id: str, doc_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=f"text for {chunk_id}",
            metadata={},
        ),
        score=score,
        source="test",
    )


def test_evaluate_retrieval_computes_metrics_across_queries() -> None:
    samples = [
        RetrievalSample(query_id="q1", query="python", relevant_doc_ids=["doc-1"]),
        RetrievalSample(query_id="q2", query="france", relevant_chunk_ids=["chunk-3"]),
    ]
    pipeline = FakePipeline(
        {
            "python": [
                make_result("chunk-1", "doc-2", 0.8),
                make_result("chunk-2", "doc-1", 0.7),
            ],
            "france": [
                make_result("chunk-3", "doc-3", 0.9),
            ],
        }
    )

    report = evaluate_retrieval(pipeline, samples=samples, top_k=2)

    assert report.metrics.queries_evaluated == 2
    assert report.metrics.hit_rate == 1.0
    assert report.metrics.mean_reciprocal_rank == pytest.approx((0.5 + 1.0) / 2)
    assert report.metrics.recall_at_k == 1.0


def test_evaluate_retrieval_handles_missed_queries() -> None:
    samples = [RetrievalSample(query_id="q1", query="python", relevant_doc_ids=["doc-1"])]
    pipeline = FakePipeline({"python": [make_result("chunk-1", "doc-2", 0.5)]})

    report = evaluate_retrieval(pipeline, samples=samples, top_k=1)

    assert report.metrics.hit_rate == 0.0
    assert report.metrics.mean_reciprocal_rank == 0.0
    assert report.metrics.recall_at_k == 0.0
    assert report.query_results[0].matched is False


def test_evaluate_retrieval_rejects_invalid_inputs() -> None:
    pipeline = FakePipeline({})

    with pytest.raises(ValueError, match="At least one retrieval sample"):
        evaluate_retrieval(pipeline, samples=[], top_k=5)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        evaluate_retrieval(
            pipeline,
            samples=[RetrievalSample(query_id="q1", query="python", relevant_doc_ids=["doc-1"])],
            top_k=0,
        )

    with pytest.raises(ValueError, match="blank query"):
        evaluate_retrieval(
            pipeline,
            samples=[RetrievalSample(query_id="q1", query="   ", relevant_doc_ids=["doc-1"])],
            top_k=5,
        )

    with pytest.raises(ValueError, match="must define relevant_doc_ids or relevant_chunk_ids"):
        evaluate_retrieval(
            pipeline,
            samples=[RetrievalSample(query_id="q1", query="python")],
            top_k=5,
        )


def test_load_retrieval_samples_reads_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        '\n'.join(
            [
                '{"query_id":"q1","query":"python","relevant_doc_ids":["doc-1"]}',
                '{"query_id":"q2","query":"france","relevant_chunk_ids":["chunk-2"]}',
            ]
        ),
        encoding="utf-8",
    )

    samples = load_retrieval_samples(dataset_path)

    assert len(samples) == 2
    assert samples[0].query_id == "q1"
    assert samples[0].relevant_doc_ids == ["doc-1"]
    assert samples[1].relevant_chunk_ids == ["chunk-2"]
