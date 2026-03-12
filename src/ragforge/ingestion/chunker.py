
from __future__ import annotations
from typing import List

from ragforge.core.schemas import Document, Chunk

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: The text to chunk.
        chunk_size: The size of each chunk
        overlap: The number of overlapping character between chunks

    Return :
        List of chunked strings
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >=0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    
    text = text.strip()
    if not text:
        return []
    
    chunks: List[str] = []

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
    documents: List[Document],
    chunk_size: int = 300,
    overlap: int = 50,
    ) -> List[Chunk]:
    """
    Convert a list of Documents into a list of Chunks.
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        text_chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_text_value in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{idx}",
                doc_id=doc.doc_id,
                text = chunk_text_value,
                metadata={
                    **doc.metadata,
                    "chunk_index": str(idx),
                },
            )

            all_chunks.append(chunk)
    
    return all_chunks

