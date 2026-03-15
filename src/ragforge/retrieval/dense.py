"""
This code defines a dense retrievr for RAG, what it does is
- Convert each chunk into embedding vector
- Convert the query into embedding vector
- Compare the cosine similarity between the query vector and each chunk vector
- Rank chunks by similarity
- Return the top k results

So this is a simple semantic search component

"""

from __future__ import annotations

import numpy as np

from ragforge.core.schemas import Chunk, RetrievalResult
from ragforge.retrieval.embeddings import TextEmbedder



#----------------------------------------------------------------------------------

# Helper function 1 : _as_2d_float32
# This makes sure the embeddings are in a clean predictable format :
# - Numpy array 
# - float32
# - 2D shape
def _as_2d_float32(array: np.ndarray) -> np.ndarray:
    """Convert encoder output into a 2D float32 matrix."""

    # This ensures the data is a NumPy array and uses float32, which is common for embeddings.
    # input could be a list, tuple or a numpy array
    # the output will become numpy array of float32 
    matrix = np.asarray(array, dtype=np.float32)
    # If 1D turn it into a single-row 2D matrix
    # If the embedding looks like this: [0.1, 0.2, 0.3], its shape is (3,), which is 1D.
    # This reshapes it to: [[0.1, 0.2, 0.3]] with shape (1, 3).
    # Why? Because the rest of the code expects a batch-style 2D matrix :
    # - rows = embeddings
    # - columns = dimensions
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    # Reject anything that still isn’t 2D
    # This protects against malformed encoder output.
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    return matrix



#----------------------------------------------------------------------------------



def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows for cosine similarity while keeping zero rows safe."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms




#----------------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever that ranks chunks with cosine similarity."""

    def __init__(self, chunks: list[Chunk], embedder: TextEmbedder) -> None:
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk.")

        self.chunks = chunks
        self.embedder = embedder

        chunk_embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        self.chunk_embeddings = _normalize_rows(_as_2d_float32(chunk_embeddings))

        if self.chunk_embeddings.shape[0] != len(chunks):
            raise ValueError("DenseRetriever requires one embedding per chunk.")

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search the dense index and return ranked retrieval results."""
        query = query.strip()
        if not query:
            return []
        if top_k <= 0:
            return []

        query_embedding = self.embedder.encode([query])
        query_matrix = _normalize_rows(_as_2d_float32(query_embedding))

        if query_matrix.shape[0] != 1:
            raise ValueError("DenseRetriever query encoding must produce exactly one vector.")
        if query_matrix.shape[1] != self.chunk_embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match chunk embeddings.")

        scores = self.chunk_embeddings @ query_matrix[0]
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                source="dense",
            )
            for index in ranked_indices
        ]

#----------------------------------------------------------------------------------