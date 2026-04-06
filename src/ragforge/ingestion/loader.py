from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ragforge.core.schemas import Document
from ragforge.ingestion.parsers import load_docx_file, load_pdf_file, load_text_file


SUPPORTED_FILE_TYPES = {".txt", ".pdf", ".docx"}


def _coerce_paths(path_or_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Normalize a path input into a concrete list of filesystem paths."""
    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    return [Path(path) for path in path_or_paths]


def _iter_supported_files(path_or_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Collect supported files from explicit paths or directories."""
    files: list[Path] = []

    for path in _coerce_paths(path_or_paths):
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_FILE_TYPES:
                    files.append(child)
            continue

        if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")
        files.append(path)

    return files


def _load_file(file_path: Path) -> Document | None:
    """Dispatch a file to the appropriate parser."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(file_path)
    if suffix == ".pdf":
        return load_pdf_file(file_path)
    if suffix == ".docx":
        return load_docx_file(file_path)

    raise ValueError(f"Unsupported file type: {suffix or '<none>'}")


def load_documents(path_or_paths: str | Path | Iterable[str | Path]) -> list[Document]:
    """Load supported document files into `Document` objects."""
    documents: list[Document] = []
    for file_path in _iter_supported_files(path_or_paths):
        document = _load_file(file_path)
        if document is not None:
            documents.append(document)
    return documents


def load_text_documents(data_dir: str | Path) -> list[Document]:
    """Backward-compatible wrapper for text-only directory loading."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory of documents: {path}")

    return [document for document in load_documents(path) if document.metadata.get("file_type") == "txt"]
