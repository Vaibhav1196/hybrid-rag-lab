"""
This code defines a dense retrievr for RAG, what it does is
- Convert each chunk into embedding vector
- Convert the query into embedding vector
- Compare the cosine similarity between the query vector and each chunk vector
- Rank chunks by similarity
- Return the top k results


Whats happening here ?

“Take all document chunks, embed them, normalize them, and store them.
When a query comes in, embed and normalize it too.
Compare the query to every chunk using cosine similarity.
Keep the chunks with positive similarity, sort them from most similar to least similar, and return the best ones.”

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


# Helper function 2 : _normalize_rows
# This normalizes each row vector to unit length so that the dot product becomes cosine similarity.
def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows for cosine similarity while keeping zero rows safe."""

    # Compute the length of each row
    # If matrix is :
    # [[3, 4],
    #  [1, 2]]
    # norms will be :
    # [[5],
    # [sqrt(5)]]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # If a row is all zeros, its norm is 0. Dividing by 0 would break.
    # So zero norms are replaced with 1.0.
    # That means a zero row stays zero after division instead of crashing.
    safe_norms = np.where(norms == 0, 1.0, norms)
    # Divide each row by its norm
    # This makes every nonzero row have length 1.
    return matrix / safe_norms


#----------------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever that ranks chunks with cosine similarity."""

    def __init__(self, chunks: list[Chunk], embedder: TextEmbedder) -> None:
        if not chunks:
            raise ValueError("DenseRetriever requires at least one chunk.")

        # the text chunks to index
        self.chunks = chunks
        # the embedding model used to encode text
        self.embedder = embedder

        # This extracts the text from each chunk and passes the list of strings to the embedder.
        chunk_embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        # Convert and normalize chunk embeddings
        # After this, self.chunk_embeddings is a 2D float32 normalized matrix.
        self.chunk_embeddings = _normalize_rows(_as_2d_float32(chunk_embeddings))

        # Check that each chunk got one embedding
        # The number of rows in the embedding matrix should match the number of chunks.
        # self.chunk_embeddings.shape is (number_of_chunks, embedding_dimension)
        if self.chunk_embeddings.shape[0] != len(chunks):
            raise ValueError("DenseRetriever requires one embedding per chunk.")



    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search the dense index and return ranked retrieval results."""
        query = query.strip()
        # If query is empty return no result
        if not query:
            return []
        # If top_k is less than or equal to 0 return no result
        if top_k <= 0:
            return []

        # The query is passed as a list containing one string.
        # The embedder should return one embedding vector.
        query_embedding = self.embedder.encode([query])
        # Convert and normalize query embedding
        # This ensures 2D float32 matrix and normalized row vector
        # Since there is only one query, shape should be : (1, embedding_dimension)
        query_matrix = _normalize_rows(_as_2d_float32(query_embedding))

        # The code expects exactly one query vector.
        if query_matrix.shape[0] != 1:
            raise ValueError("DenseRetriever query encoding must produce exactly one vector.")
        # Validate embedding dimension matches chunks
        # If chunk embeddings are dimension 768, query embedding must also be 768.
        if query_matrix.shape[1] != self.chunk_embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match chunk embeddings.")

        # This is the core retrieval step.
        # self.chunk_embeddings is shape (num_chunks, dim)
        # query_matrix[0] is shape (dim,)
        # Matrix-vector multiplication gives: (num_chunks,)
        scores = self.chunk_embeddings @ query_matrix[0]

        # Rank indices with positive scores only
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Return the top k results
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                source="dense",
            )
            for index in ranked_indices
        ]


#----------------------------------------------------------------------------------