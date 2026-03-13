from __future__ import annotations

from pathlib import Path

import pytest

from ragforge.core.schemas import Document
from ragforge.retrieval.pipeline import BM25Pipeline


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_bm25_pipeline_builds_from_directory_and_returns_ranked_results(tmp_path: Path) -> None:
    write_text_file(
        tmp_path / "python.txt",
        "Python is widely used for backend systems, data work, and AI applications.",
    )
    write_text_file(
        tmp_path / "retrieval.txt",
        "BM25 is a sparse retrieval algorithm for ranking relevant text documents.",
    )
    write_text_file(tmp_path / "empty.txt", "   ")

    pipeline = BM25Pipeline.from_directory(tmp_path, chunk_size=80, overlap=10)
    results = pipeline.search("sparse retrieval algorithm", top_k=2)

    assert len(pipeline.documents) == 2
    assert len(pipeline.chunks) >= 2
    assert len(results) == 1
    assert results[0].chunk.doc_id == "retrieval"
    assert results[0].source == "bm25"
    assert results[0].score > 0


def test_bm25_pipeline_builds_from_documents() -> None:
    documents = [
        Document(
            doc_id="doc-1",
            text="LangChain helps build applications with language models and retrieval chains.",
            metadata={"source": "memory"},
        ),
        Document(
            doc_id="doc-2",
            text="LangGraph focuses on durable orchestration for stateful agent workflows.",
            metadata={"source": "memory"},
        ),
    ]

    pipeline = BM25Pipeline.from_documents(documents, chunk_size=80, overlap=10)
    results = pipeline.search("durable orchestration workflows", top_k=1)

    assert len(pipeline.documents) == 2
    assert len(pipeline.chunks) >= 2
    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc-2"


def test_bm25_pipeline_rejects_directories_without_non_empty_documents(tmp_path: Path) -> None:
    write_text_file(tmp_path / "blank.txt", "   ")

    with pytest.raises(ValueError, match="No non-empty text documents found"):
        BM25Pipeline.from_directory(tmp_path)


def test_bm25_pipeline_rejects_empty_chunk_output() -> None:
    documents = [
        Document(doc_id="doc-1", text="   ", metadata={"source": "memory"}),
    ]

    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Pipeline.from_documents(documents)
