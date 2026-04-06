from __future__ import annotations

from pathlib import Path

from ragforge.core.schemas import Document


def load_text_file(file_path: str | Path) -> Document | None:
    """Load a `.txt` file into a `Document`."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    return Document(
        doc_id=path.stem,
        text=text,
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_type": "txt",
            "parser": "text",
        },
    )
