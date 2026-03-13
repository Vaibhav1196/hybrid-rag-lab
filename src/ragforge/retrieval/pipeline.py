from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragforge.core.schemas import Chunk, Document, RetrievalResult
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.loader import load_text_documents
from ragforge.retrieval.bm25 import BM25Retriever
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.embeddings import SentenceTransformerEmbedder, TextEmbedder
from ragforge.retrieval.fusion import reciprocal_rank_fusion
from ragforge.retrieval.reranking import CrossEncoderScorer, QueryDocumentScorer, RetrievalReranker


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


@dataclass(slots=True)
class HybridPipeline:
    """End-to-end hybrid retrieval pipeline using BM25, dense retrieval, and RRF."""

    documents: list[Document]
    chunks: list[Chunk]
    sparse_retriever: BM25Retriever
    dense_retriever: DenseRetriever
    rrf_k: int = 60

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedder: TextEmbedder | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ) -> HybridPipeline:
        """Build a hybrid pipeline from already-loaded documents."""
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        resolved_embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        return cls(
            documents=list(documents),
            chunks=chunks,
            sparse_retriever=BM25Retriever(chunks),
            dense_retriever=DenseRetriever(chunks=chunks, embedder=resolved_embedder),
            rrf_k=rrf_k,
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        embedder: TextEmbedder | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ) -> HybridPipeline:
        """Load `.txt` documents, chunk them, and build a hybrid index."""
        documents = _load_documents(data_dir)

        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
            rrf_k=rrf_k,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run sparse and dense retrieval, then fuse the rankings with RRF."""
        if top_k <= 0:
            return []

        sparse_results = self.sparse_retriever.search(query=query, top_k=top_k)
        dense_results = self.dense_retriever.search(query=query, top_k=top_k)
        return reciprocal_rank_fusion(
            [sparse_results, dense_results],
            top_k=top_k,
            k=self.rrf_k,
        )


@dataclass(slots=True)
class RerankedHybridPipeline:
    """Hybrid retrieval pipeline with a reranking stage over the fused shortlist."""

    documents: list[Document]
    chunks: list[Chunk]
    hybrid_pipeline: HybridPipeline
    reranker: RetrievalReranker
    candidate_top_k: int = 10

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedder: TextEmbedder | None = None,
        scorer: QueryDocumentScorer | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_top_k: int = 10,
    ) -> RerankedHybridPipeline:
        """Build a reranked hybrid pipeline from already-loaded documents."""
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be > 0.")

        hybrid_pipeline = HybridPipeline.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
            rrf_k=rrf_k,
        )
        resolved_scorer = scorer or CrossEncoderScorer(model_name=reranker_model_name)

        return cls(
            documents=list(documents),
            chunks=hybrid_pipeline.chunks,
            hybrid_pipeline=hybrid_pipeline,
            reranker=RetrievalReranker(resolved_scorer),
            candidate_top_k=candidate_top_k,
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        embedder: TextEmbedder | None = None,
        scorer: QueryDocumentScorer | None = None,
        chunk_size: int = 300,
        overlap: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_top_k: int = 10,
    ) -> RerankedHybridPipeline:
        """Load documents and build a reranked hybrid retrieval pipeline."""
        documents = _load_documents(data_dir)

        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            scorer=scorer,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
            rrf_k=rrf_k,
            reranker_model_name=reranker_model_name,
            candidate_top_k=candidate_top_k,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve with the hybrid pipeline, then rerank the fused shortlist."""
        if top_k <= 0:
            return []

        candidates = self.hybrid_pipeline.search(
            query=query,
            top_k=max(top_k, self.candidate_top_k),
        )
        return self.reranker.rerank(query=query, results=candidates, top_k=top_k)
