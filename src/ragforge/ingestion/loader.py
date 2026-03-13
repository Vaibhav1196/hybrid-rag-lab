from __future__ import annotations

from pathlib import Path

from ragforge.core.schemas import Document


def load_text_documents(data_dir: str | Path) -> list[Document]:
    """Load non-empty `.txt` files from a directory into `Document` objects."""
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory of text files: {data_path}")

    documents: list[Document] = []

    for file_path in sorted(data_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                doc_id=file_path.stem,
                text=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                },
            )
        )

    return documents
