from __future__ import annotations

import numpy as np

from ragforge.core.schemas import Chunk
from ragforge.retrieval.dense import DenseRetriever


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        metadata={},
    )


def test_dense_retriever_returns_semantically_ranked_results() -> None:
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "paris france capital": [0.0, 1.0],
            "programming with python": [0.8, 0.2],
        }
    )
    retriever = DenseRetriever(
        chunks=[
            make_chunk("c1", "python backend systems"),
            make_chunk("c2", "paris france capital"),
        ],
        embedder=embedder,
    )

    results = retriever.search("programming with python", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].source == "dense"
    assert results[0].score >= results[1].score


def test_dense_retriever_returns_empty_for_blank_query() -> None:
    embedder = FakeEmbedder({"chunk text": [1.0, 0.0]})
    retriever = DenseRetriever([make_chunk("c1", "chunk text")], embedder=embedder)

    assert retriever.search("   ", top_k=3) == []


def test_dense_retriever_returns_empty_for_non_positive_top_k() -> None:
    embedder = FakeEmbedder(
        {
            "chunk text": [1.0, 0.0],
            "query text": [1.0, 0.0],
        }
    )
    retriever = DenseRetriever([make_chunk("c1", "chunk text")], embedder=embedder)

    assert retriever.search("query text", top_k=0) == []
    assert retriever.search("query text", top_k=-1) == []


def test_dense_retriever_filters_out_zero_score_results() -> None:
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "paris france capital": [0.0, 1.0],
            "basketball sports": [0.0, 0.0],
        }
    )
    retriever = DenseRetriever(
        chunks=[
            make_chunk("c1", "python backend systems"),
            make_chunk("c2", "paris france capital"),
        ],
        embedder=embedder,
    )

    assert retriever.search("basketball sports", top_k=2) == []


def test_dense_retriever_raises_for_empty_chunk_list() -> None:
    embedder = FakeEmbedder({})

    try:
        DenseRetriever([], embedder=embedder)
        assert False, "Expected ValueError for empty chunk list"
    except ValueError as exc:
        assert "at least one chunk" in str(exc)
