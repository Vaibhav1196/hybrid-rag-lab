from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragforge.core.schemas import Document
from ragforge.retrieval.pipeline import RerankedHybridPipeline


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


class FakeScorer:
    def __init__(self, mapping: dict[tuple[str, str], float]) -> None:
        self.mapping = mapping

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray([self.mapping[pair] for pair in pairs], dtype=np.float32)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_reranked_pipeline_reranks_the_hybrid_shortlist() -> None:
    documents = [
        Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"}),
        Document(doc_id="doc-2", text="python tutorial guide", metadata={"source": "memory"}),
        Document(doc_id="doc-3", text="capital city of france", metadata={"source": "memory"}),
    ]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "python tutorial guide": [0.8, 0.2],
            "capital city of france": [0.0, 1.0],
            "python guide": [0.9, 0.1],
        }
    )
    scorer = FakeScorer(
        {
            ("python guide", "python backend systems"): 0.3,
            ("python guide", "python tutorial guide"): 0.95,
        }
    )

    pipeline = RerankedHybridPipeline.from_documents(
        documents,
        embedder=embedder,
        scorer=scorer,
        candidate_top_k=2,
        chunk_size=80,
        overlap=10,
    )
    results = pipeline.search("python guide", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.doc_id == "doc-2"
    assert results[0].source == "hybrid_reranked"


def test_reranked_pipeline_returns_empty_for_non_positive_top_k() -> None:
    documents = [Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"})]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "python": [1.0, 0.0],
        }
    )
    scorer = FakeScorer({("python", "python backend systems"): 1.0})

    pipeline = RerankedHybridPipeline.from_documents(
        documents,
        embedder=embedder,
        scorer=scorer,
        candidate_top_k=1,
    )

    assert pipeline.search("python", top_k=0) == []
    assert pipeline.search("python", top_k=-1) == []


def test_reranked_pipeline_rejects_invalid_candidate_top_k() -> None:
    documents = [Document(doc_id="doc-1", text="python backend systems", metadata={"source": "memory"})]
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
        }
    )
    scorer = FakeScorer({})

    with pytest.raises(ValueError, match="candidate_top_k must be > 0"):
        RerankedHybridPipeline.from_documents(
            documents,
            embedder=embedder,
            scorer=scorer,
            candidate_top_k=0,
        )


def test_reranked_pipeline_rejects_directories_without_non_empty_documents(tmp_path: Path) -> None:
    write_text_file(tmp_path / "blank.txt", "   ")
    embedder = FakeEmbedder({})
    scorer = FakeScorer({})

    with pytest.raises(ValueError, match="No non-empty supported documents found"):
        RerankedHybridPipeline.from_directory(
            tmp_path,
            embedder=embedder,
            scorer=scorer,
        )
