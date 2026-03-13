from ragforge.core.schemas import Chunk
from ragforge.retrieval.bm25 import BM25Retriever



def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        metadata={}
    )


def test_bm25_returns_results_for_relevant_query() -> None:
    chunks = [
        make_chunk("c1", "Python is a programming language."),
        make_chunk("c2", "Paris is the capital of France."),
        make_chunk("c3", "Basketball is played on a court."),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("What is Python?", top_k=2)

    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].source == "bm25"


def test_bm25_respects_top_k() -> None:
    chunks = [
        make_chunk("c1", "apple orange banana"),
        make_chunk("c2", "car bus train"),
        make_chunk("c3", "dog cat bird"),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("apple", top_k=1)

    assert len(results) == 1


def test_bm25_returns_empty_for_blank_query() -> None:
    chunks = [
        make_chunk("c1", "some text here"),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("   ", top_k=5)

    assert results == []


def test_bm25_raises_for_empty_chunk_list() -> None:
    try:
        BM25Retriever([])
        assert False, "Expected ValueError for empty chunk list"
    except ValueError as exc:
        assert "at least one chunk" in str(exc)