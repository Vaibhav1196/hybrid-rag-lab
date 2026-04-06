from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from docx import Document as DocxDocument

from ragforge.ingestion.loader import load_documents, load_text_documents


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


def test_load_documents_supports_txt_pdf_and_docx(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("plain text document", encoding="utf-8")
    write_pdf(tmp_path / "report.pdf", "pdf document content")
    write_docx(tmp_path / "memo.docx", "docx document content")

    documents = load_documents(tmp_path)

    assert [document.doc_id for document in documents] == ["memo", "note", "report"]
    assert {document.metadata["file_type"] for document in documents} == {"docx", "pdf", "txt"}


def test_load_documents_skips_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    assert load_documents(tmp_path) == []


def test_load_documents_supports_explicit_file_paths(tmp_path: Path) -> None:
    text_path = tmp_path / "note.txt"
    text_path.write_text("plain text document", encoding="utf-8")

    documents = load_documents([text_path])

    assert len(documents) == 1
    assert documents[0].doc_id == "note"
    assert documents[0].metadata["file_type"] == "txt"


def test_load_documents_rejects_unsupported_explicit_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents(path)


def test_load_text_documents_keeps_text_only_behavior(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("plain text document", encoding="utf-8")
    write_pdf(tmp_path / "report.pdf", "pdf document content")

    documents = load_text_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].doc_id == "note"
    assert documents[0].metadata["file_type"] == "txt"
