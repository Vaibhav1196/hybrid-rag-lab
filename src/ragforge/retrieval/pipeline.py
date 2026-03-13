from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragforge.core.schemas import Chunk, Document, RetrievalResult
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.loader import load_text_documents
from ragforge.retrieval.bm25 import BM25Retriever
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.embeddings import SentenceTransformerEmbedder, TextEmbedder


def _build_chunks(
    documents: list[Document],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Build chunks from documents and validate that the result is non-empty."""
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Retrieval pipeline requires at least one chunk to build an index.")
    return chunks


def _load_documents(data_dir: str | Path) -> list[Document]:
    """Load and validate text documents for retrieval."""
    documents = load_text_documents(data_dir)
    if not documents:
        raise ValueError(f"No non-empty text documents found in: {Path(data_dir)}")
    return documents


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
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)

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
        documents = _load_documents(data_dir)

        return cls.from_documents(
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run a query against the indexed chunks."""
        return self.retriever.search(query=query, top_k=top_k)


@dataclass(slots=True)
class DensePipeline:
    """End-to-end dense retrieval pipeline built on the ingestion layer."""

    documents: list[Document]
    chunks: list[Chunk]
    retriever: DenseRetriever

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedder: TextEmbedder | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> DensePipeline:
        """Build a dense pipeline from already-loaded documents."""
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        resolved_embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        return cls(
            documents=list(documents),
            chunks=chunks,
            retriever=DenseRetriever(chunks=chunks, embedder=resolved_embedder),
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        embedder: TextEmbedder | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> DensePipeline:
        """Load `.txt` documents, chunk them, and build a dense index."""
        documents = _load_documents(data_dir)

        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run a query against the dense index."""
        return self.retriever.search(query=query, top_k=top_k)
