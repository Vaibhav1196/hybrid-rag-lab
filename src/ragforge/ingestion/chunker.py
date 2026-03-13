from __future__ import annotations

from ragforge.core.schemas import Chunk, Document


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: The text to chunk.
        chunk_size: The size of each chunk.
        overlap: The number of overlapping characters between chunks.

    Returns:
        A list of chunked strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """Convert documents into chunks while preserving source metadata."""
    all_chunks: list[Chunk] = []

    for doc in documents:
        text_chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_text_value in enumerate(text_chunks):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text_value,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                    },
                )
            )

    return all_chunks
