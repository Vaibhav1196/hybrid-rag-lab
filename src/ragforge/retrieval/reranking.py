"""

This file represents the reranking stage of the retrieval system.

In mordern RAG the typical flow is :

Query -> Retriever (fast, approximate) -> Top ~50-100 documents -> Reranker (slow, accurate) -> Top-K documents 

Our code implements the reranker layer using a "cross-encoder model".
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ragforge.core.schemas import RetrievalResult


#----------------------------------------------------------------------------------

class QueryDocumentScorer(Protocol):
    """Protocol for scoring query-document text pairs for reranking."""

    # Input  -> [(query, document_text), ...]
    # Output -> [s1, s2, s3, ...]
    # Example : [
    # ("what is RAG?", "Retrieval augmented generation..."),
    # ("what is RAG?", "Python is a programming language...")
    # ] -> [0.98, 0.12]
    # Higher score → more relevant.
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return one score per query-document pair."""



#----------------------------------------------------------------------------------


# This class implements the QueryDocumentScorer protocol using a cross-encoder model.
# A cross-encoder processes: [query + document] -> together in the same transformer.
# Example input to the model : [CLS] query tokens [SEP] document tokens [SEP]
# This allows deep interaction between query and document tokens.
# That makes cross-encoders much more accurate than bi-encoders (embeddings).
# But they are also much slower, which is why they are used only for reranking.
class CrossEncoderScorer:
    """Thin adapter around sentence-transformers CrossEncoder for reranking."""

    # cross-encoder/ms-marco-MiniLM-L-6-v2 most common rerank models
    # Properties:
    #  - trained on MS MARCO
    #  - optimized for search ranking
    #  - small and fast        
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:

        # avoids loading heavy dependencies unless needed -> lazy loading
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        # This loads the transformer model.
        self.model = CrossEncoder(model_name)


    # This method scores query-document pairs.
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score query-document pairs with a cross-encoder."""

        # If no pairs are provided, return an empty array.
        # shape (0,)
        if not pairs:
            return np.zeros((0,), dtype=np.float32)


        scores = self.model.predict(pairs)
        # shape (num_pairs,)
        # Convert to float32 and ensures consistent numpy format
        return np.asarray(scores, dtype=np.float32)


#----------------------------------------------------------------------------------


# This class takes a shortlist of retrieved documents and reranks them.
# It does not retrieve documents itself.
class RetrievalReranker:
    """Rerank an existing shortlist of retrieval results."""

    # The reranker takes a scorer object implementing the protocol.
    # Example:
    # scorer = CrossEncoderScorer()
    # reranker = RetrievalReranker(scorer)  
    # Because of the Protocol, you could also plug in:
    #  - OpenAI reranker
    #  - Cohere reranker
    #  - custom transformer model  
    def __init__(self, scorer: QueryDocumentScorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results for a query and return the top-k items."""
        query = query.strip()
        if not query:
            return []
        if top_k <= 0:
            return []
        if not results:
            return []


        # Build the query-document pairs for the scorer.
        # [(query, result.chunk.text) for result in results]
        # Example: [("what is RAG?", "Retrieval augmented generation..."), ...]
        # Score the pairs : example output -> [0.92, 0.17] -> Convert to numpy ->  np.asarray(...).reshape(-1)
        # This ensures shape : (n_results,)
        scores = np.asarray(
            self.scorer.score([(query, result.chunk.text) for result in results]),
            dtype=np.float32,
        ).reshape(-1)

        # Each result must have exactly one score.
        # 5 documents → 5 scores
        # # Example: [0.92, 0.17] 
        if scores.shape[0] != len(results):
            raise ValueError("Reranker requires exactly one score per retrieval result.")


        # Sort results by score in descending order
        # scores = [0.2, 0.9, 0.5] -> indices = [1, 2, 0] 
        # sorted by score -> [1,2,0]
        ranked_indices = sorted(
            range(len(results)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        # Return the top-k results
        # Example: [1,2,0] -> results[1], results[2], results[0]
        return [
            RetrievalResult(
                chunk=results[index].chunk,
                score=float(scores[index]),
                source="hybrid_reranked",
            )
            for index in ranked_indices
        ]

#----------------------------------------------------------------------------------