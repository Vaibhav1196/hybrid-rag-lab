from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragforge.core.schemas import Document
from ragforge.retrieval.pipeline import HybridPipeline


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_hybrid_pipeline_builds_from_documents_and_fuses_results() -> None:
    documents = [
        Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"}),
        Document(doc_id="doc-2", text="semantic vector search", metadata={"source": "memory"}),
        Document(doc_id="doc-3", text="capital city of france", metadata={"source": "memory"}),
    ]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "semantic vector search": [0.7, 0.3],
            "capital city of france": [0.0, 1.0],
            "python search": [0.95, 0.05],
        }
    )

    pipeline = HybridPipeline.from_documents(
        documents,
        embedder=embedder,
        chunk_size=80,
        overlap=10,
        rrf_k=60,
    )
    results = pipeline.search("python search", top_k=3)

    assert len(pipeline.documents) == 3
    assert len(pipeline.chunks) == 3
    assert len(results) >= 1
    assert results[0].chunk.doc_id == "doc-1"
    assert results[0].source == "hybrid_rrf"


def test_hybrid_pipeline_handles_one_retriever_returning_no_results() -> None:
    documents = [
        Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"}),
        Document(doc_id="doc-2", text="capital city of france", metadata={"source": "memory"}),
    ]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "capital city of france": [0.0, 1.0],
            "software engineering": [1.0, 0.0],
        }
    )

    pipeline = HybridPipeline.from_documents(
        documents,
        embedder=embedder,
        chunk_size=80,
        overlap=10,
    )
    results = pipeline.search("software engineering", top_k=2)

    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc-1"


def test_hybrid_pipeline_returns_empty_for_non_positive_top_k() -> None:
    documents = [
        Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"}),
    ]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "python": [1.0, 0.0],
        }
    )

    pipeline = HybridPipeline.from_documents(documents, embedder=embedder)

    assert pipeline.search("python", top_k=0) == []
    assert pipeline.search("python", top_k=-1) == []


def test_hybrid_pipeline_rejects_directories_without_non_empty_documents(tmp_path: Path) -> None:
    write_text_file(tmp_path / "blank.txt", "   ")
    embedder = FakeEmbedder({})

    with pytest.raises(ValueError, match="No non-empty text documents found"):
        HybridPipeline.from_directory(tmp_path, embedder=embedder)


def test_hybrid_pipeline_rejects_empty_chunk_output() -> None:
    documents = [Document(doc_id="doc-1", text="   ", metadata={"source": "memory"})]
    embedder = FakeEmbedder({})

    with pytest.raises(ValueError, match="at least one chunk"):
        HybridPipeline.from_documents(documents, embedder=embedder)
