from __future__ import annotations

from pathlib import Path

import fitz

from ragforge.core.schemas import Document


def load_pdf_file(file_path: str | Path) -> Document | None:
    """Load a `.pdf` file into a `Document` using PyMuPDF."""
    path = Path(file_path)
    with fitz.open(path) as pdf:
        pages = [page.get_text("text").strip() for page in pdf]

    text = "\n\n".join(page for page in pages if page)
    if not text.strip():
        return None

    return Document(
        doc_id=path.stem,
        text=text.strip(),
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_type": "pdf",
            "parser": "pymupdf",
            "page_count": len(pages),
        },
    )
