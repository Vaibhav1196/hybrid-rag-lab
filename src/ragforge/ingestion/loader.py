from __future__ import annotations
from genericpath import exists

from pathlib import Path
from typing import List

from ragforge.core.schemas import Document


def load_text_documents(data_dir: str | Path) -> List[Document]:
    """
    Load all .txt files from a directory into Document objects
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    documents: List[Document] = []

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
                }
            )
        )

    return documents

