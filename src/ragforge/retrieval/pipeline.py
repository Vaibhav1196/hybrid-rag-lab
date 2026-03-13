from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragforge.core.schemas import Chunk, Document, RetrievalResult
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.loader import load_text_documents
from ragforge.retrieval.bm25 import BM25Retriever


@dataclass(slots=True)
class BM25Pipeline:
    """End-to-end sparse retrieval pipeline built on the ingestion layer."""

    documents: list[Document]
    chunks: list[Chunk]
    retriever: BM25Retriever

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> BM25Pipeline:
        """Build a BM25 pipeline from already-loaded documents."""
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("BM25Pipeline requires at least one chunk to build an index.")

        return cls(
            documents=list(documents),
            chunks=chunks,
            retriever=BM25Retriever(chunks),
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> BM25Pipeline:
        """Load `.txt` documents, chunk them, and build a BM25 index."""
        documents = load_text_documents(data_dir)
        if not documents:
            raise ValueError(f"No non-empty text documents found in: {Path(data_dir)}")

        return cls.from_documents(
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run a query against the indexed chunks."""
        return self.retriever.search(query=query, top_k=top_k)
