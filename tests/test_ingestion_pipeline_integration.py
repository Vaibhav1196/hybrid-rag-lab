from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from docx import Document as DocxDocument

from ragforge.ingestion.loader import load_documents
from ragforge.retrieval.pipeline import HybridPipeline


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def write_pdf(path: Path, text: str) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def write_docx(path: Path, text: str) -> None:
    doc = DocxDocument()
    doc.add_paragraph(text)
    doc.save(path)


def test_mixed_format_documents_flow_into_hybrid_pipeline(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("python backend systems", encoding="utf-8")
    write_pdf(tmp_path / "guide.pdf", "capital city of france")
    write_docx(tmp_path / "manual.docx", "hybrid retrieval with bm25 and dense vectors")

    documents = load_documents(tmp_path)
    embedder = FakeEmbedder(
        {
            "python backend systems": [1.0, 0.0],
            "capital city of france": [0.0, 1.0],
            "hybrid retrieval with bm25 and dense vectors": [0.8, 0.2],
            "bm25 dense retrieval": [0.75, 0.25],
        }
    )

    pipeline = HybridPipeline.from_documents(
        documents=documents,
        embedder=embedder,
        chunk_size=120,
        overlap=10,
    )
    results = pipeline.search("bm25 dense retrieval", top_k=3)

    assert len(documents) == 3
    assert len(results) >= 1
    assert results[0].chunk.doc_id == "manual"
