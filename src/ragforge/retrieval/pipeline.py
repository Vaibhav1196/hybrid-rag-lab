
"""
This is the file orchestration layer of the retrieval system.

The earlier files defined the building blocks
- load documents
- split them into chunks 
- BM25 retrieval 
- Dense retrieval 
- Fuse rankings with RRF
- Rerank with cross-encoder

This file ties all of that together into four ready-to-use pipelines:
- BM25Pipeline
- DensePipeline
- HybridPipeline
- RerankedHybridPipeline

"""



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragforge.core.schemas import Chunk, Document, RetrievalResult
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.loader import load_documents
from ragforge.retrieval.bm25 import BM25Retriever
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.embeddings import SentenceTransformerEmbedder, TextEmbedder
from ragforge.retrieval.fusion import reciprocal_rank_fusion
from ragforge.retrieval.reranking import CrossEncoderScorer, QueryDocumentScorer, RetrievalReranker



#----------------------------------------------------------------------------------

# Helper functions takes :
# - takes documents
# - chunks them 
# - verifies that the result is non-empty
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

#----------------------------------------------------------------------------------

# Helper function takes :
# loads supported documents from a directory
# checks that something was loaded
# returns list of documents
def _load_documents(data_dir: str | Path) -> list[Document]:
    """Load and validate supported documents for retrieval."""
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No non-empty supported documents found in: {Path(data_dir)}")
    return documents


#----------------------------------------------------------------------------------

# This is a complete sparse retrieval pipeline.
# It bundles together : 
# - Loaded documents 
# - Generated chunks
# - A bm25 retriever
@dataclass(slots=True)
class BM25Pipeline:
    """End-to-end sparse retrieval pipeline built on the ingestion layer."""

    # So an instance stores eveything needed for a BM25 retrieval.
    documents: list[Document]
    chunks: list[Chunk]
    retriever: BM25Retriever

    # This builds pipeline from document already in memmory
    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> BM25Pipeline:
        """Build a BM25 pipeline from already-loaded documents."""
        # Creates chunks and validates non-empty output.
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)

        # Returns a new instance of the class with the documents, chunks, and a BM25 retriever.
        # This creates and returns a BM25Pipeline instance.
        return cls(
            documents=list(documents),
            chunks=chunks,
            retriever=BM25Retriever(chunks),
        )


    # This builds the same pipeline, but starting from a folder path.x
    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> BM25Pipeline:
        """Load `.txt` documents, chunk them, and build a BM25 index."""

        # load documents
        documents = _load_documents(data_dir)

        # delegate to from_documents
        return cls.from_documents(
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )


    # This runs a query against the indexed chunks.
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run a query against the indexed chunks."""
        # This just forwards the call to the underlying retriever :
        # So the pipeline behaves like a ready-to-use search engine.
        return self.retriever.search(query=query, top_k=top_k)


#----------------------------------------------------------------------------------


# This is the same idea, but for dense retrieval.
# It stores : documents, chunks, a DenseRetriever
@dataclass(slots=True)
class DensePipeline:
    """End-to-end dense retrieval pipeline built on the ingestion layer."""

    documents: list[Document]
    chunks: list[Chunk]
    retriever: DenseRetriever


    # This builds a dense retrieval pipeline.
    # New things here:
    #  - optional custom embedder
    #  - default embedding model name
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

        # chunk documents
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)

        # resolve embedder
        # This means:
        # - if user supplied an embedder, use it
        # - otherwise create a default SentenceTransformerEmbedder
        # So this pipeline is customizable. 
        resolved_embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        
        # build dense retriever & return pipeline
        return cls(
            documents=list(documents),
            chunks=chunks,
            retriever=DenseRetriever(chunks=chunks, embedder=resolved_embedder),
        )


    # Same pattern as BM25
    # load docs from disk
    # call from_documents(...)
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

        # build dense retriever & return pipeline
        return cls.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
        )

    # Just forwards to dense search.
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run a query against the dense index."""
        return self.retriever.search(query=query, top_k=top_k)

#----------------------------------------------------------------------------------


# This combines:
# - BM25 retrieval
# - dense retrieval
# - Reciprocal Rank Fusion (RRF)
# So this is your hybrid retriever.
@dataclass(slots=True)
class HybridPipeline:
    """End-to-end hybrid retrieval pipeline using BM25, dense retrieval, and RRF."""

    # It stores both retrievers plus the RRF constant.
    documents: list[Document]
    chunks: list[Chunk]
    sparse_retriever: BM25Retriever
    dense_retriever: DenseRetriever
    rrf_k: int = 60

    # This builds the hybrid pipeline.
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

        # chunk documents
        chunks = _build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        # resolve embedder
        resolved_embedder = embedder or SentenceTransformerEmbedder(model_name=model_name)

        # build both retrievers , store RRF constant
        return cls(
            documents=list(documents),
            chunks=chunks,
            sparse_retriever=BM25Retriever(chunks),
            dense_retriever=DenseRetriever(chunks=chunks, embedder=resolved_embedder),
            rrf_k=rrf_k,
        )


    # Same pattern again:
    # load documents
    # delegate to from_documents(...)
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


    # This is where hybrid retrieval actually happens.
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run sparse and dense retrieval, then fuse the rankings with RRF."""
        if top_k <= 0:
            return []

        # sparse retrieval
        sparse_results = self.sparse_retriever.search(query=query, top_k=top_k)
        # dense retrieval
        dense_results = self.dense_retriever.search(query=query, top_k=top_k)
        # fuse them with RRF
        # This combines the two ranked lists using RRF.
        return reciprocal_rank_fusion(
            [sparse_results, dense_results],
            top_k=top_k,
            k=self.rrf_k,
        )

#----------------------------------------------------------------------------------


# This is the most advanced pipeline in the file.
# It does:
# - hybrid retrieval
# - shortlist creation
# - reranking with a cross-encoder
# This is the most complete retrieval pipeline here.
@dataclass(slots=True)
class RerankedHybridPipeline:
    """Hybrid retrieval pipeline with a reranking stage over the fused shortlist."""
    # hybrid_pipeline: used to retrieve candidate results
    # reranker: reorders those candidates
    # candidate_top_k: how many candidates to rerank
    documents: list[Document]
    chunks: list[Chunk]
    hybrid_pipeline: HybridPipeline
    reranker: RetrievalReranker
    candidate_top_k: int = 10


    # This builds the full hybrid + reranking system.
    # It introduces two customizable components:
    # - embedder
    # - scorer
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

        # validate candidate pool size & You must rerank at least one candidate.
        if candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be > 0.")

        # build hybrid pipeline
        # This internally builds:
        # - chunks
        # - BM25 retriever
        # - dense retriever
        # - fusion config
        hybrid_pipeline = HybridPipeline.from_documents(
            documents=documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
            rrf_k=rrf_k,
        )

        # resolve scorer
        # This means:
        # - if a scorer was provided, use it
        # - otherwise create a default cross-encoder scorer
        resolved_scorer = scorer or CrossEncoderScorer(model_name=reranker_model_name)


        # build reranker & return pipeline
        # It reuses the chunks already created by the hybrid pipeline instead of rebuilding them.
        return cls(
            documents=list(documents),
            chunks=hybrid_pipeline.chunks,
            hybrid_pipeline=hybrid_pipeline,
            reranker=RetrievalReranker(resolved_scorer),
            candidate_top_k=candidate_top_k,
        )


    # Same pattern again:
    # load documents
    # delegate to from_documents(...)
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


    # This is the final retrieval flow.
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve with the hybrid pipeline, then rerank the fused shortlist."""
        if top_k <= 0:
            return []


        # retrieve candidate shortlist
        # This is important.
        # It asks the hybrid pipeline for : max(top_k, self.candidate_top_k)
        # Why ? Because reranking usually needs a larger candidate pool than the final number of results.
        '''
        Example:
        final top_k = 5
        candidate_top_k = 10
        Then hybrid search returns 10 candidates, and reranker chooses the best 5.

        If top_k is already larger than candidate_top_k, it uses top_k.

        Example:
        final top_k = 20
        candidate_top_k = 10
        Then it retrieves 20 candidates, because returning fewer than 20 would make it impossible to output 20 final results.
        '''
        candidates = self.hybrid_pipeline.search(
            query=query,
            top_k=max(top_k, self.candidate_top_k),
        )

        # rerank candidates
        # The reranker scores each (query, chunk_text) pair with the cross-encoder and returns the top final results.
        return self.reranker.rerank(query=query, results=candidates, top_k=top_k)

#----------------------------------------------------------------------------------
