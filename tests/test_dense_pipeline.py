from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragforge.core.schemas import Document
from ragforge.retrieval.pipeline import DensePipeline


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_dense_pipeline_builds_from_directory_and_returns_ranked_results(tmp_path: Path) -> None:
    write_text_file(tmp_path / "python.txt", "python backend systems")
    write_text_file(tmp_path / "paris.txt", "paris france capital")
    write_text_file(tmp_path / "empty.txt", "   ")

    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "paris france capital": [0.0, 1.0],
            "python programming": [0.9, 0.1],
        }
    )

    pipeline = DensePipeline.from_directory(
        tmp_path,
        embedder=embedder,
        chunk_size=80,
        overlap=10,
    )
    results = pipeline.search("python programming", top_k=2)

    assert len(pipeline.documents) == 2
    assert len(pipeline.chunks) == 2
    assert len(results) == 2
    assert results[0].chunk.doc_id == "python"
    assert results[0].source == "dense"


def test_dense_pipeline_builds_from_documents() -> None:
    documents = [
        Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"}),
        Document(doc_id="doc-2", text="paris france capital", metadata={"source": "memory"}),
    ]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "paris france capital": [0.0, 1.0],
            "python programming": [0.9, 0.1],
        }
    )

    pipeline = DensePipeline.from_documents(
        documents,
        embedder=embedder,
        chunk_size=80,
        overlap=10,
    )
    results = pipeline.search("python programming", top_k=1)

    assert len(pipeline.documents) == 2
    assert len(pipeline.chunks) == 2
    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc-1"


def test_dense_pipeline_rejects_directories_without_non_empty_documents(tmp_path: Path) -> None:
    write_text_file(tmp_path / "blank.txt", "   ")
    embedder = FakeEmbedder({})

    with pytest.raises(ValueError, match="No non-empty supported documents found"):
        DensePipeline.from_directory(tmp_path, embedder=embedder)


def test_dense_pipeline_rejects_empty_chunk_output() -> None:
    documents = [Document(doc_id="doc-1", text="   ", metadata={"source": "memory"})]
    embedder = FakeEmbedder({})

    with pytest.raises(ValueError, match="at least one chunk"):
        DensePipeline.from_documents(documents, embedder=embedder)
